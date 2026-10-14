"""Export match data for the static website (site/data/).

Probabilities come from model v2 (v1 plus the expected-runs-remaining features, src/v2_resources.py).
2024 matches use v2 trained on 2005-2023; 2025-26 matches use v2 trained on 2005-2024. v2 was
designed after the 2025-26 test had been scored, so the frozen v1 test score stays the headline.

site/data/index.json            : one entry per match (for the picker)
site/data/matches/<id>.json     : ball-by-ball probabilities, scores, events, biggest swings
site/data/summary.json          : headline results for the 'how good is it' section
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import log_loss

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
OUT = ROOT / "site" / "data"
COMP_NAMES = {"ipl": "IPL", "bbl": "Big Bash", "psl": "PSL", "cpl": "CPL", "sat": "SA20",
              "ntb": "T20 Blast", "sma": "Syed Mushtaq Ali", "t20s": "T20 International"}


def dismissal_text(r):
    if not r.is_wicket:
        return None
    k = r.dismissal_kind
    if k == "caught":
        return f"{r.player_out} caught off {r.bowler}"
    if k == "caught and bowled":
        return f"{r.player_out} caught and bowled {r.bowler}"
    if k in ("bowled", "lbw", "stumped", "hit wicket"):
        return f"{r.player_out} {k} by {r.bowler}" if k != "lbw" else f"{r.player_out} lbw to {r.bowler}"
    return f"{r.player_out} {k}"


def ball_text(r):
    w = dismissal_text(r)
    if w:
        return f"Wicket: {w}"
    if r.runs_batter == 6:
        return f"Six: {r.batter} off {r.bowler}"
    if r.runs_batter == 4:
        return f"Four: {r.batter} off {r.bowler}"
    if r.runs_total == 0:
        return f"Dot ball: {r.bowler} to {r.batter}"
    return f"{r.runs_total} run{'s' if r.runs_total != 1 else ''}: {r.batter} off {r.bowler}"


def main():
    (OUT / "matches").mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(PROC / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    m = m.set_index("match_id")

    v = pd.read_parquet(RES / "valid_predictions_v2model.parquet").rename(columns={"p_v2": "p"})
    t = pd.read_parquet(RES / "test_predictions_v2.parquet").rename(columns={"p_v2": "p"})
    preds = pd.concat([v[["match_id", "innings", "balls_bowled", "p"]],
                       t[["match_id", "innings", "balls_bowled", "p"]]], ignore_index=True)
    preds["match_id"] = preds["match_id"].astype(str)
    ids = set(preds["match_id"])

    d = pd.read_parquet(PROC / "deliveries.parquet", columns=[
        "match_id", "innings", "over", "ball_in_over", "legal_ball", "legal_balls_cum", "batter",
        "bowler", "runs_batter", "runs_total", "is_wicket", "dismissal_kind", "player_out",
        "runs_cum", "wickets_cum", "batting_team"])
    for c in ["match_id", "batter", "bowler", "player_out", "dismissal_kind", "batting_team"]:
        d[c] = d[c].astype(object)
    d["match_id"] = d["match_id"].astype(str)
    d = d[d["match_id"].isin(ids)]
    legal = d[d["legal_ball"]].copy()
    legal["balls_bowled"] = legal["legal_balls_cum"] - 1

    legal_by = {k: g.set_index(["innings", "balls_bowled"]) for k, g in legal.groupby("match_id", sort=False)}
    totals_all = d.groupby(["match_id", "innings"])[["runs_cum", "wickets_cum"]].max()
    index = []
    for mid, g in preds.groupby("match_id", sort=False):
        if mid not in legal_by:
            continue
        info = m.loc[mid]
        g = g.sort_values(["innings", "balls_bowled"])
        lg = legal_by[mid]
        team1 = info["team_batting_first"]
        team2 = info["team2"] if team1 == info["team1"] else info["team1"]
        # P(team batting first wins), before each legal ball
        p1 = np.where(g["innings"] == 1, g["p"], 1 - g["p"])
        balls = []
        lgd = {k: r for k, r in zip(lg.index, lg.itertuples())}
        for (inn, bb), p in zip(zip(g["innings"], g["balls_bowled"]), p1):
            r = lgd.get((inn, bb))
            if r is None:
                continue
            balls.append({"i": int(inn), "o": f"{int(r.over)}.{int(r.ball_in_over)}",
                          "p": round(float(p), 4),
                          "s": f"{int(r.runs_cum - r.runs_total)}/{int(r.wickets_cum - (1 if r.is_wicket else 0))}",
                          "r": int(r.runs_total), **({"w": 1} if r.is_wicket else {}),
                          **({"t": ball_text(r)} if (r.is_wicket or r.runs_batter in (4, 6)) else {}),
                          "b": f"{r.bowler} to {r.batter}"})
        if len(balls) < 20:
            continue
        final_p = 1.0 if info["winner"] == team1 else 0.0
        # swing of each ball = probability after it minus before it (for team batting first)
        nxt = [b["p"] for b in balls[1:]] + [final_p]
        for b, n in zip(balls, nxt):
            b["d"] = round(n - b["p"], 4)
        swings = sorted(range(len(balls)), key=lambda k: -abs(balls[k]["d"]))[:3]
        totals = totals_all.loc[mid]
        score = {int(k): f"{int(r.runs_cum)}/{int(r.wickets_cum)}" for k, r in totals.iterrows()}
        lead_changes = int(np.sum(np.diff(np.sign(np.array([b["p"] for b in balls] + [final_p]) - 0.5)) != 0))
        drama = round(float(np.mean([1 - abs(b["p"] - 0.5) * 2 for b in balls])), 3)
        rec = {"id": mid, "date": info["date"].strftime("%Y-%m-%d"),
               "comp": COMP_NAMES.get(info["competition"], info["competition"]),
               "event": info["event"], "venue": info["venue"],
               "team1": team1, "team2": team2, "winner": info["winner"],
               "score1": score.get(1), "score2": score.get(2),
               "sample": "2024 validation" if pd.Timestamp(info["date"]).year == 2024 else "2025-26 test"}
        names = sorted({n for b in balls for n in b["b"].split(" to ")})
        ni = {n: k for k, n in enumerate(names)}
        cols = {"inn": [b["i"] for b in balls], "over": [b["o"] for b in balls],
                "p": [b["p"] for b in balls], "d": [b["d"] for b in balls],
                "score": [b["s"] for b in balls], "runs": [b["r"] for b in balls],
                "wkt": [k for k, b in enumerate(balls) if b.get("w")],
                "bowler": [ni[b["b"].split(" to ")[0]] for b in balls],
                "batter": [ni[b["b"].split(" to ")[1]] for b in balls],
                "text": {k: b["t"] for k, b in enumerate(balls) if "t" in b}}
        detail = dict(rec, names=names, balls=cols, final=final_p, swings=swings)
        (OUT / "matches" / f"{mid}.json").write_text(json.dumps(detail, separators=(",", ":")))
        index.append(dict(rec, drama=drama, lead_changes=lead_changes,
                          max_swing=round(abs(balls[swings[0]]["d"]), 3),
                          low=round(float(min(b["p"] if final_p == 1 else 1 - b["p"] for b in balls)), 3)))
    index.sort(key=lambda r: r["date"], reverse=True)
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    test = pd.read_csv(RES / "test_overall.csv").set_index("model")
    mk = pd.read_csv(RES / "market_prematch.csv").set_index("slice")
    cal = pd.read_csv(RES / "test_calibration.csv")
    tv2 = pd.read_parquet(RES / "test_predictions_v2.parquet")
    summary = {
        "model_version": "v2",
        "v2_test_log_loss": round(log_loss(tv2["label"].to_numpy(), tv2["p_v2"].to_numpy()), 4),
        "test_matches": 1539,
        "test": {k: {"log_loss": round(r.log_loss, 4), "accuracy": round(r.accuracy, 4)}
                 for k, r in test.iterrows()},
        "calibration": [{"pred": round(r.pred, 4), "actual": round(r.actual, 4), "n": int(r.n)}
                        for r in cal.itertuples()],
        "market": {k: {"matches": int(r.matches), "model": round(r.logloss_model, 4),
                       "market": round(r.logloss_market, 4)} for k, r in mk.iterrows()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    size = sum(f.stat().st_size for f in (OUT / "matches").glob("*.json"))
    print(f"Exported {len(index):,} matches ({size/1e6:.1f} MB), index {len((OUT/'index.json').read_bytes())/1e3:.0f} KB")
    top = sorted(index, key=lambda r: r["low"])[:5]
    print("Biggest comebacks (winner's lowest probability):")
    for r in top:
        print(f"  {r['date']} {r['comp']}: {r['team1']} {r['score1']} v {r['team2']} {r['score2']}, "
              f"winner {r['winner']} was down to {r['low']:.1%}")


if __name__ == "__main__":
    main()

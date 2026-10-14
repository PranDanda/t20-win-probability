"""Player-quality features: bowling resources and batting resources still available.

All player career stats use ONLY matches on earlier dates, shrunk toward a typical player.

Ball-level outputs (joined onto features_v2 -> features_v3.parquet):
  bowl_now_econ, bowl_now_wkt     current bowler's career runs/ball and wickets/ball
  rem_bowl_econ, rem_bowl_wkt     expected runs/ball and wickets/ball over the rest of the innings,
                                  if the fielding side uses its best remaining bowling
                                  (each bowler capped at 4 overs)
  bat_res_avg                     sum of career batting averages of the batting side's
                                  players who are not yet out
  bat_res_sr                      their mean career strike rate (runs/ball)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC, JSON_DIR = ROOT / "data" / "processed", ROOT / "data" / "json"

# Shrinkage priors: (typical value, strength in balls / dismissals)
BAT_AVG_PRIOR, BAT_AVG_K = 18.0, 3          # runs per dismissal, worth 3 dismissals
BAT_SR_PRIOR, BAT_SR_K = 1.2, 100           # runs per ball, worth 100 balls
ECON_PRIOR, WKT_PRIOR, BOWL_K = 1.32, 0.055, 120   # ~7.9 an over, a wicket every ~18 balls
REPLACEMENT_ECON, REPLACEMENT_WKT = 1.55, 0.035    # part-timer filling unallocated overs
MIN_BOWLER_BALLS = 60      # counts as a "real" bowler with this much history
QUOTA = 24                 # 4 overs
NOT_BOWLER_WICKETS = {"run out", "retired hurt", "retired not out", "retired out",
                      "obstructing the field", "hit the ball twice"}


def read_rosters():
    rows = []
    for comp in sorted(p for p in JSON_DIR.iterdir() if p.is_dir()):
        for f in comp.glob("*.json"):
            d = json.loads(f.read_text())
            for team, players in d["info"].get("players", {}).items():
                rows += [{"match_id": f.stem, "team": team, "player": p} for p in players]
            for inn in d.get("innings", []):
                for ov in inn.get("overs", []):
                    for dl in ov["deliveries"]:
                        for r in dl.get("replacements", {}).get("match", []):
                            rows.append({"match_id": f.stem, "team": r["team"], "player": r["in"]})
    return pd.DataFrame(rows).drop_duplicates()


def career_stats(d, dates):
    """Per (player, date): career totals from strictly earlier dates."""
    d = d.merge(dates, on="match_id")
    faced = d[d["extra_wides"] == 0]
    bat = faced.groupby(["batter", "date"], observed=True).agg(
        runs=("runs_batter", "sum"), balls=("runs_batter", "size")).reset_index()
    bat = bat.rename(columns={"batter": "player"})
    outs = (d[d["player_out"].notna()].groupby(["player_out", "date"], observed=True)
            .size().rename("outs").reset_index().rename(columns={"player_out": "player"}))
    d["conceded"] = d["runs_total"] - d["extra_byes"] - d["extra_legbyes"]
    d["bowler_wkt"] = d["is_wicket"] & ~d["dismissal_kind"].isin(NOT_BOWLER_WICKETS)
    bowl = d.groupby(["bowler", "date"], observed=True).agg(
        conceded=("conceded", "sum"), bwkts=("bowler_wkt", "sum"),
        bballs=("legal_ball", "sum")).reset_index().rename(columns={"bowler": "player"})

    s = bat.merge(outs, on=["player", "date"], how="outer").merge(bowl, on=["player", "date"], how="outer")
    s = s.fillna(0).sort_values(["player", "date"])
    g = s.groupby("player")
    for c in ["runs", "balls", "outs", "conceded", "bwkts", "bballs"]:
        s[f"prior_{c}"] = g[c].cumsum()             # totals up to AND INCLUDING this date;
        # looked up later only for strictly LATER match dates (no same-day leakage)
    s["bat_avg"] = (s["prior_runs"] + BAT_AVG_PRIOR * BAT_AVG_K) / (s["prior_outs"] + BAT_AVG_K)
    s["bat_sr"] = (s["prior_runs"] + BAT_SR_PRIOR * BAT_SR_K) / (s["prior_balls"] + BAT_SR_K)
    s["econ"] = (s["prior_conceded"] + ECON_PRIOR * BOWL_K) / (s["prior_bballs"] + BOWL_K)
    s["wkt"] = (s["prior_bwkts"] + WKT_PRIOR * BOWL_K) / (s["prior_bballs"] + BOWL_K)
    return s[["player", "date", "bat_avg", "bat_sr", "econ", "wkt", "prior_bballs"]]


def player_on_date(rosters, stats, dates):
    """Stats for each rostered player as of the match date (players with no history -> priors)."""
    r = rosters.merge(dates, on="match_id")
    # For each match, take the player's cumulative totals as of their latest STRICTLY EARLIER
    # playing date.
    st = stats.sort_values("date")
    r = r.sort_values("date")
    out = pd.merge_asof(r, st, on="date", by="player", direction="backward",
                        allow_exact_matches=False)   # strictly earlier dates only
    return out


def main():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    dates = m[["match_id", "date"]]
    d = pd.read_parquet(PROC / "deliveries.parquet", columns=[
        "match_id", "innings", "over", "legal_ball", "legal_balls_cum", "batting_team",
        "bowling_team", "batter", "bowler", "non_striker", "runs_batter", "runs_total",
        "extra_wides", "extra_byes", "extra_legbyes", "is_wicket", "dismissal_kind", "player_out"])
    for c in ["match_id", "batting_team", "bowling_team", "batter", "bowler", "non_striker",
              "player_out", "dismissal_kind"]:
        d[c] = d[c].astype(object).where(d[c].notna(), None)
    d["match_id"] = d["match_id"].astype(str)

    print("Reading team sheets...", flush=True)
    rosters = read_rosters()
    print(f"  {len(rosters):,} player-match rows", flush=True)

    stats = career_stats(d, dates)
    ps = player_on_date(rosters, stats, dates)
    ps = ps.fillna({"bat_avg": BAT_AVG_PRIOR, "bat_sr": BAT_SR_PRIOR, "econ": ECON_PRIOR,
                    "wkt": WKT_PRIOR, "prior_bballs": 0})
    # two players can share a name in one match: keep one row per (match, name)
    q = ps.drop_duplicates(["match_id", "player"]).set_index(["match_id", "player"])

    # ---------- Current bowler (ball level)
    legal = d[d["legal_ball"]].copy()
    legal["balls_bowled"] = (legal["legal_balls_cum"] - 1).astype("int16")
    cb = q[["econ", "wkt"]].rename(columns={"econ": "bowl_now_econ", "wkt": "bowl_now_wkt"})
    legal = legal.join(cb, on=["match_id", "bowler"])
    legal["bowl_now_econ"] = legal["bowl_now_econ"].fillna(ECON_PRIOR)
    legal["bowl_now_wkt"] = legal["bowl_now_wkt"].fillna(WKT_PRIOR)

    # ---------- Remaining bowling resources (computed at the start of each over)
    print("Allocating remaining bowling overs...", flush=True)
    roster_by = rosters.groupby(["match_id", "team"])["player"].apply(list).to_dict()
    qd = q[["econ", "wkt", "prior_bballs"]].to_dict("index")
    over_bowler = (d[d["legal_ball"]].groupby(["match_id", "innings", "over"], observed=True)
                   .agg(bowler=("bowler", "first"), balls=("legal_ball", "sum"),
                        team=("bowling_team", "first")).reset_index())
    rows = []
    for (mid, inn), grp in over_bowler.groupby(["match_id", "innings"], sort=False):
        team = grp["team"].iloc[0]
        squad = roster_by.get((mid, team), [])
        used = defaultdict(int)
        legal_before = 0
        for r in grp.itertuples():
            remaining = max(120 - legal_before, 0)
            pool = []
            for p in set(squad) | set(used):
                st = qd.get((mid, p), {"econ": ECON_PRIOR, "wkt": WKT_PRIOR, "prior_bballs": 0})
                if st["prior_bballs"] >= MIN_BOWLER_BALLS or used[p] > 0:
                    left = max(QUOTA - used[p], 0)
                    if left:
                        pool.append((st["econ"], st["wkt"], left))
            pool.sort()                            # best (lowest economy) first
            need, runs_exp, wkts_exp = remaining, 0.0, 0.0
            for econ, wkt, left in pool:
                take = min(left, need)
                runs_exp += take * econ; wkts_exp += take * wkt; need -= take
                if need == 0:
                    break
            runs_exp += need * REPLACEMENT_ECON; wkts_exp += need * REPLACEMENT_WKT
            denom = max(remaining, 1)
            rows.append((mid, inn, r.over, runs_exp / denom, wkts_exp / denom))
            used[r.bowler] += int(r.balls)
            legal_before += int(r.balls)
    rem = pd.DataFrame(rows, columns=["match_id", "innings", "over", "rem_bowl_econ", "rem_bowl_wkt"])

    # ---------- Batting resources: sum of averages of players not yet out
    print("Batting resources...", flush=True)
    team_sum = (ps.groupby(["match_id", "team"])
                .agg(team_bat_avg=("bat_avg", "sum"), team_bat_sr=("bat_sr", "mean"),
                     n_players=("player", "size")).reset_index())
    d["out_avg"] = d.join(q["bat_avg"], on=["match_id", "player_out"])["bat_avg"]
    d.loc[d["player_out"].notna() & d["out_avg"].isna(), "out_avg"] = BAT_AVG_PRIOR
    d["out_avg"] = d["out_avg"].fillna(0.0)
    d["out_sr"] = d.join(q["bat_sr"], on=["match_id", "player_out"])["bat_sr"]
    d["out_n"] = d["player_out"].notna().astype(int)
    d["out_sr"] = d["out_sr"].where(d["player_out"].notna(), 0.0).fillna(BAT_SR_PRIOR)
    g = d.groupby(["match_id", "innings"], sort=False)
    d["dismissed_avg_before"] = g["out_avg"].cumsum() - d["out_avg"]
    d["dismissed_sr_before"] = g["out_sr"].cumsum() - d["out_sr"]
    d["dismissed_n_before"] = g["out_n"].cumsum() - d["out_n"]
    d = d.merge(team_sum, left_on=["match_id", "batting_team"], right_on=["match_id", "team"], how="left")
    d["bat_res_avg"] = d["team_bat_avg"] - d["dismissed_avg_before"]
    left_n = (d["n_players"] - d["dismissed_n_before"]).clip(lower=1)
    d["bat_res_sr"] = (d["team_bat_sr"] * d["n_players"] - d["dismissed_sr_before"]) / left_n

    batres = d[d["legal_ball"]].copy()
    batres["balls_bowled"] = (batres["legal_balls_cum"] - 1).astype("int16")
    ball = legal[["match_id", "innings", "balls_bowled", "over", "bowl_now_econ", "bowl_now_wkt"]].merge(
        batres[["match_id", "innings", "balls_bowled", "bat_res_avg", "bat_res_sr"]],
        on=["match_id", "innings", "balls_bowled"], how="left")
    ball = ball.merge(rem, on=["match_id", "innings", "over"], how="left").drop(columns="over")

    f = pd.read_parquet(PROC / "features_v2.parquet")
    f["match_id"] = f["match_id"].astype(str)
    n0 = len(f)
    f = f.merge(ball, on=["match_id", "innings", "balls_bowled"], how="left")
    assert len(f) == n0
    for c in ["bowl_now_econ", "bowl_now_wkt", "rem_bowl_econ", "rem_bowl_wkt", "bat_res_avg", "bat_res_sr"]:
        f[c] = f[c].astype("float32")
    f["match_id"] = f["match_id"].astype("category")
    f.to_parquet(PROC / "features_v3.parquet", index=False)

    new = ["bowl_now_econ", "bowl_now_wkt", "rem_bowl_econ", "rem_bowl_wkt", "bat_res_avg", "bat_res_sr"]
    print(f"\nRows {len(f):,}; missing share:", f[new].isna().mean().round(4).to_dict())
    print(f[f.split == "train"][new].describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()

"""Parse Cricsheet men's T20 JSON into two clean tables.

Outputs (in data/processed/):
  matches.parquet    - one row per match: teams, venue, date, toss, result, flags
  deliveries.parquet - one row per delivery (regulation innings only, no super overs)
  summary.csv        - match and delivery counts per competition

Data: Cricsheet (https://cricsheet.org), Open Data Commons Attribution Licence.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JSON_DIR = ROOT / "data" / "json"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_match(path: Path, competition: str):
    d = json.loads(path.read_text())
    info = d["info"]
    match_id = path.stem
    teams = info["teams"]
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    result = outcome.get("result")  # 'no result', 'tie', 'draw' or None
    if winner is None and result == "tie":
        winner = outcome.get("eliminator")  # super-over winner, if recorded
    method = outcome.get("method")  # e.g. 'D/L', 'Awarded'
    toss = info.get("toss", {})

    regulation = [inn for inn in d.get("innings", []) if not inn.get("super_over")]
    first_bat = regulation[0]["team"] if regulation else None

    match = {
        "match_id": match_id,
        "competition": competition,
        "event": info.get("event", {}).get("name"),
        "season": str(info.get("season")),
        "date": pd.to_datetime(info["dates"][0]),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0],
        "team2": teams[1],
        "team_batting_first": first_bat,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "result": result,
        "method": method,
        "overs_scheduled": info.get("overs", 20),
        "balls_per_over": info.get("balls_per_over", 6),
        "tied": result == "tie",
        "n_regulation_innings": len(regulation),
    }

    rows = []
    for inn_idx, inn in enumerate(regulation[:2], start=1):
        bat = inn["team"]
        bowl = teams[1] if bat == teams[0] else teams[0]
        target = inn.get("target", {})
        runs_cum = wkts_cum = legal_cum = 0
        seq = 0
        for over in inn.get("overs", []):
            ball_in_over = 0
            for dl in over["deliveries"]:
                seq += 1
                extras = dl.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                legal = not (is_wide or is_noball)
                if legal:
                    ball_in_over += 1
                    legal_cum += 1
                r = dl["runs"]
                wickets = dl.get("wickets", [])
                runs_cum += r["total"]
                wkts_cum += len(wickets)
                rows.append({
                    "match_id": match_id,
                    "innings": inn_idx,
                    "batting_team": bat,
                    "bowling_team": bowl,
                    "seq": seq,
                    "over": over["over"],            # 0-indexed over number
                    "ball_in_over": ball_in_over,    # legal balls so far this over
                    "legal_ball": legal,
                    "batter": dl["batter"],
                    "bowler": dl["bowler"],
                    "non_striker": dl["non_striker"],
                    "runs_batter": r["batter"],
                    "runs_extras": r["extras"],
                    "runs_total": r["total"],
                    "extra_wides": extras.get("wides", 0),
                    "extra_noballs": extras.get("noballs", 0),
                    "extra_byes": extras.get("byes", 0),
                    "extra_legbyes": extras.get("legbyes", 0),
                    "is_wicket": len(wickets) > 0,
                    "n_wickets": len(wickets),
                    "dismissal_kind": wickets[0]["kind"] if wickets else None,
                    "player_out": wickets[0]["player_out"] if wickets else None,
                    # state AFTER this delivery
                    "runs_cum": runs_cum,
                    "wickets_cum": wkts_cum,
                    "legal_balls_cum": legal_cum,
                    "target_runs": target.get("runs"),
                    "target_overs": target.get("overs"),
                })
    return match, rows


CAT_COLS = ["batting_team", "bowling_team", "batter", "bowler", "non_striker",
            "dismissal_kind", "player_out", "match_id"]


def compact(ddf: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        ddf[c] = ddf[c].astype("category")
    for c in ["innings", "over", "ball_in_over", "runs_batter", "runs_extras", "runs_total",
              "extra_wides", "extra_noballs", "extra_byes", "extra_legbyes", "n_wickets",
              "wickets_cum"]:
        ddf[c] = ddf[c].astype("int8")
    for c in ["seq", "runs_cum", "legal_balls_cum"]:
        ddf[c] = ddf[c].astype("int16")
    ddf["target_runs"] = ddf["target_runs"].astype("float32")
    ddf["target_overs"] = ddf["target_overs"].astype("float32")
    return ddf


def main():
    parts_dir = OUT_DIR / "deliveries_by_competition"
    parts_dir.mkdir(exist_ok=True)
    matches, errors = [], []
    for comp_dir in sorted(p for p in JSON_DIR.iterdir() if p.is_dir()):
        comp_rows, comp_matches = [], []
        for f in sorted(comp_dir.glob("*.json")):
            try:
                m, rows = parse_match(f, comp_dir.name)
            except Exception as e:  # keep going, report at the end
                errors.append((f.name, repr(e)))
                continue
            comp_matches.append(m)
            comp_rows.extend(rows)
        cm = pd.DataFrame(comp_matches)
        cd = pd.DataFrame(comp_rows)
        del comp_rows
        cd = cd.merge(cm[["match_id", "winner"]], on="match_id", how="left")
        won = (cd["batting_team"] == cd["winner"]).astype("int8")
        won[cd["winner"].isna()] = -1  # no winner (no result / unresolved tie)
        cd["batting_team_won"] = won
        cd = compact(cd.drop(columns="winner"))
        cd.insert(1, "competition", comp_dir.name)
        cd.to_parquet(parts_dir / f"{comp_dir.name}.parquet", index=False)
        print(f"{comp_dir.name}: {len(cm)} matches, {len(cd):,} deliveries", flush=True)
        matches.extend(comp_matches)
        del cd, cm

    mdf = pd.DataFrame(matches).drop_duplicates("match_id")
    ddf = pd.concat([pd.read_parquet(f) for f in sorted(parts_dir.glob("*.parquet"))],
                    ignore_index=True)
    for c in CAT_COLS + ["competition"]:
        ddf[c] = ddf[c].astype("category")

    # Older files omit the chase target: fill it as first-innings total + 1 over 20 overs.
    # (Where a target is recorded it is kept, so penalty-run adjustments survive.)
    first_total = (ddf[ddf["innings"] == 1].groupby("match_id", observed=True)["runs_cum"]
                   .last().rename("first_innings_total"))
    ddf = ddf.merge(first_total, on="match_id", how="left")
    ddf["match_id"] = ddf["match_id"].astype("category")
    ddf["target_filled"] = (ddf["innings"] == 2) & ddf["target_runs"].isna()
    ddf.loc[ddf["target_filled"], "target_runs"] = ddf["first_innings_total"] + 1
    ddf.loc[ddf["target_filled"], "target_overs"] = 20
    ddf = ddf.drop(columns="first_innings_total")

    # Flag matches usable for modelling: a winner, two innings, 20 x 6-ball overs,
    # no D/L or other method, and no reduced-overs target
    mdf["usable"] = (
        mdf["winner"].notna()
        & (mdf["n_regulation_innings"] == 2)
        & (mdf["overs_scheduled"] == 20)
        & (mdf["balls_per_over"] == 6)
        & mdf["method"].isna()
    )
    tgt = ddf[ddf["innings"] == 2].groupby("match_id", observed=True)["target_overs"].first()
    shortened = tgt[tgt.notna() & (tgt != 20)].index
    mdf.loc[mdf["match_id"].isin(shortened), "usable"] = False

    mdf.to_parquet(OUT_DIR / "matches.parquet", index=False)
    ddf.to_parquet(OUT_DIR / "deliveries.parquet", index=False)

    usable_ids = set(mdf.loc[mdf["usable"], "match_id"])
    ddf["usable"] = ddf["match_id"].isin(usable_ids)
    summary = (
        mdf.groupby("competition")
        .agg(matches=("match_id", "size"), usable_matches=("usable", "sum"),
             first_season=("season", "min"), last_season=("season", "max"))
        .join(ddf.groupby("competition", observed=True).size().rename("deliveries"))
        .join(ddf[ddf["usable"]].groupby("competition", observed=True).size()
              .rename("usable_deliveries"))
    )
    summary.loc["TOTAL"] = [summary["matches"].sum(), summary["usable_matches"].sum(), "", "",
                            summary["deliveries"].sum(), summary["usable_deliveries"].sum()]
    summary.to_csv(OUT_DIR / "summary.csv")
    print()
    print(summary.to_string())
    print(f"\nParse errors: {len(errors)}")
    for e in errors[:10]:
        print("  ", e)
    print("Excluded from modelling:")
    print("  no winner (no result / unresolved tie):", int(mdf["winner"].isna().sum()))
    print("  D/L or other method:", int(mdf["method"].notna().sum()))
    print("  not 2 innings:", int((mdf["n_regulation_innings"] != 2).sum()))
    print("  reduced-overs chase:", len(shortened))


if __name__ == "__main__":
    main()

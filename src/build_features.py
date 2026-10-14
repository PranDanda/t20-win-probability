"""Turn deliveries into one modelling row per legal ball, describing the match state
BEFORE that ball, plus the label (did the batting team win?) and the time-based split.

Output: data/processed/features.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TRAIN_END = pd.Timestamp("2023-12-31")   # train: matches up to end of 2023
VALID_END = pd.Timestamp("2024-12-31")   # validation: 2024; test: 2025 onwards
BALLS = 120


def main():
    m = pd.read_parquet(PROC / "matches.parquet")
    # Modelling set: usable matches, excluding ties decided by super overs
    keep = m[m["usable"] & ~m["tied"]][["match_id", "date", "competition"]]
    d = pd.read_parquet(
        PROC / "deliveries.parquet",
        columns=["match_id", "innings", "legal_ball", "over", "runs_total", "n_wickets",
                 "runs_cum", "wickets_cum", "legal_balls_cum", "target_runs",
                 "batting_team_won"],
    )
    d["match_id"] = d["match_id"].astype(str)
    d = d[d["match_id"].isin(set(keep["match_id"])) & d["legal_ball"]]

    # --- State BEFORE this ball (cumulative-after minus this ball's own contribution).
    # Wides/no-balls bowled before this ball are already inside the cumulative totals.
    f = pd.DataFrame({
        "match_id": d["match_id"].values,
        "innings": d["innings"].values.astype("int8"),
        "balls_bowled": (d["legal_balls_cum"] - 1).values.astype("int16"),
        "runs": (d["runs_cum"] - d["runs_total"]).values.astype("int16"),
        "wickets_lost": (d["wickets_cum"] - d["n_wickets"]).values.astype("int8"),
        "over": d["over"].values.astype("int8"),
        "target": d["target_runs"].values,
        "label": d["batting_team_won"].values.astype("int8"),
    })
    f["balls_left"] = (BALLS - f["balls_bowled"]).clip(lower=1).astype("int16")
    f["wickets_in_hand"] = (10 - f["wickets_lost"]).astype("int8")
    overs_bowled = f["balls_bowled"] / 6
    f["current_rr"] = np.where(f["balls_bowled"] > 0, f["runs"] / overs_bowled.replace(0, np.nan), 0.0)
    f["phase"] = pd.cut(f["over"], [-1, 5, 14, 30], labels=["powerplay", "middle", "death"])

    # Chase features (2nd innings only; NaN in the 1st innings)
    chase = f["innings"] == 2
    f["runs_needed"] = np.where(chase, f["target"] - f["runs"], np.nan)
    f["required_rr"] = np.where(chase, f["runs_needed"] / (f["balls_left"] / 6), np.nan)
    f["rr_gap"] = f["required_rr"] - f["current_rr"]
    f = f.drop(columns="target")

    # --- Attach date/competition and the time-based split
    f = f.merge(keep, on="match_id", how="left")
    f["split"] = np.select(
        [f["date"] <= TRAIN_END, f["date"] <= VALID_END], ["train", "valid"], "test")
    for c in ["match_id", "split", "competition"]:
        f[c] = f[c].astype("category")
    f["current_rr"] = f["current_rr"].astype("float32")
    for c in ["runs_needed", "required_rr", "rr_gap"]:
        f[c] = f[c].astype("float32")

    f.to_parquet(PROC / "features.parquet", index=False)

    # --- Report
    print(f"Rows (legal balls): {len(f):,}  from {f['match_id'].nunique():,} matches\n")
    s = (f.groupby(["split", "innings"], observed=True)
         .agg(rows=("label", "size"), matches=("match_id", "nunique"),
              batting_team_win_rate=("label", "mean"),
              first_date=("date", "min"), last_date=("date", "max")))
    print(s.round(3).to_string())
    print("\nExample: one IPL chase, every 24th ball")
    ex = f[(f["competition"] == "ipl") & (f["innings"] == 2) & (f["split"] == "test")]
    mid = ex["match_id"].iloc[0]
    cols = ["balls_left", "wickets_in_hand", "runs", "runs_needed", "required_rr",
            "current_rr", "phase", "label"]
    print(ex[ex["match_id"] == mid][cols].iloc[::24].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

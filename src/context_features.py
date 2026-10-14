"""Week 4: context features, each computed ONLY from matches played on earlier dates.

Match-level (one value per match, known before the first ball):
  elo_batting_first, elo_chasing        team strength (Elo), pre-match
  venue_par                              shrunk average first-innings total at the ground
  venue_chase_rate                       shrunk share of matches won by the chasing side there
  comp_level                             average first-innings total of the competition's
                                         previous 60 matches (the scoring era)
Ball-level:
  striker_rpb, non_striker_rpb           batters' career runs per ball before this match (shrunk)

Output: data/processed/features_v2.parquet (features.parquet + these columns)
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

ELO_START, ELO_K = 1500.0, 24.0
VENUE_K = 5           # shrink venue par toward the competition level, worth 5 matches
CHASE_K = 10          # shrink venue chase rate toward 50%, worth 10 matches
COMP_WINDOW = 60      # matches in the rolling scoring-era average
BAT_K, BAT_PRIOR = 100, 1.2   # shrink batters' runs/ball toward 1.2, worth 100 balls


def venue_key(v):
    # "Wankhede Stadium, Mumbai" and "Wankhede Stadium" are the same ground
    return str(v).split(",")[0].strip().lower()


def match_context(m, first_totals):
    m = m.sort_values(["date", "match_id"]).copy()
    m["first_total"] = m["match_id"].map(first_totals)
    m["venue_key"] = m["venue"].map(venue_key)

    elo = defaultdict(lambda: ELO_START)
    venue_tot, venue_n = defaultdict(float), defaultdict(int)
    venue_chase_w, venue_res_n = defaultdict(float), defaultdict(int)
    comp_recent = defaultdict(lambda: deque(maxlen=COMP_WINDOW))
    global_recent = deque(maxlen=COMP_WINDOW)
    out = []

    for date, day in m.groupby("date", sort=True):
        # 1) Features for every match on this date, from state BEFORE this date
        for r in day.itertuples():
            bat1 = r.team_batting_first
            chase = r.team2 if bat1 == r.team1 else r.team1
            cr = comp_recent[r.competition]
            comp_level = (np.mean(cr) if len(cr) >= 10
                          else np.mean(global_recent) if len(global_recent) >= 10 else 160.0)
            vn = venue_n[r.venue_key]
            par = (venue_tot[r.venue_key] + VENUE_K * comp_level) / (vn + VENUE_K)
            vr = venue_res_n[r.venue_key]
            chase_rate = (venue_chase_w[r.venue_key] + CHASE_K * 0.5) / (vr + CHASE_K)
            out.append({"match_id": r.match_id, "elo_batting_first": elo[bat1],
                        "elo_chasing": elo[chase], "venue_par": par,
                        "venue_chase_rate": chase_rate, "venue_prior_matches": vn,
                        "comp_level": comp_level})
        # 2) Then update ratings and averages with this date's results
        for r in day.itertuples():
            bat1 = r.team_batting_first
            if bat1 is None:
                continue
            chase = r.team2 if bat1 == r.team1 else r.team1
            if r.winner is not None and r.winner in (bat1, chase):
                exp1 = 1 / (1 + 10 ** ((elo[chase] - elo[bat1]) / 400))
                s1 = 1.0 if r.winner == bat1 else 0.0
                elo[bat1] += ELO_K * (s1 - exp1)
                elo[chase] -= ELO_K * (s1 - exp1)
                if r.method is None or pd.isna(r.method):
                    venue_chase_w[r.venue_key] += 1.0 - s1
                    venue_res_n[r.venue_key] += 1
            if r.usable and not pd.isna(r.first_total):
                venue_tot[r.venue_key] += r.first_total
                venue_n[r.venue_key] += 1
                comp_recent[r.competition].append(r.first_total)
                global_recent.append(r.first_total)
    return pd.DataFrame(out)


def batter_quality(d, m):
    """Career runs per ball for each batter BEFORE each match (strictly earlier dates)."""
    faced = d[d["extra_wides"] == 0]
    per = (faced.groupby(["match_id", "batter"], observed=True)
           .agg(runs=("runs_batter", "sum"), balls=("runs_batter", "size")).reset_index())
    per = per.merge(m[["match_id", "date"]], on="match_id")
    # sum per (batter, date) so double-headers on one day don't leak into each other
    per_day = per.groupby(["batter", "date"], observed=True)[["runs", "balls"]].sum().reset_index()
    per_day = per_day.sort_values(["batter", "date"])
    g = per_day.groupby("batter", observed=True)
    per_day["prior_runs"] = g["runs"].cumsum() - per_day["runs"]
    per_day["prior_balls"] = g["balls"].cumsum() - per_day["balls"]
    per_day["rpb"] = (per_day["prior_runs"] + BAT_K * BAT_PRIOR) / (per_day["prior_balls"] + BAT_K)
    return per_day[["batter", "date", "rpb", "prior_balls"]]


def main():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    d = pd.read_parquet(PROC / "deliveries.parquet",
                        columns=["match_id", "innings", "legal_ball", "legal_balls_cum",
                                 "runs_cum", "batter", "non_striker", "runs_batter", "extra_wides"])
    d["match_id"] = d["match_id"].astype(str)
    first_totals = d[d["innings"] == 1].groupby("match_id")["runs_cum"].last()

    ctx = match_context(m, first_totals)
    print(f"Match context rows: {len(ctx):,}")

    bq = batter_quality(d, m)
    legal = d[d["legal_ball"]].copy()
    legal["balls_bowled"] = (legal["legal_balls_cum"] - 1).astype("int16")
    legal = legal.merge(m[["match_id", "date"]], on="match_id")
    for role in ("batter", "non_striker"):
        q = bq.rename(columns={"batter": role, "rpb": f"{role}_rpb",
                               "prior_balls": f"{role}_prior_balls"})
        legal[role] = legal[role].astype(str)
        q[role] = q[role].astype(str)
        legal = legal.merge(q, on=[role, "date"], how="left")
    legal = legal.rename(columns={"batter_rpb": "striker_rpb",
                                  "batter_prior_balls": "striker_prior_balls"})
    ball_ctx = legal[["match_id", "innings", "balls_bowled", "striker_rpb", "non_striker_rpb",
                      "striker_prior_balls"]]

    f = pd.read_parquet(PROC / "features.parquet")
    f["match_id"] = f["match_id"].astype(str)
    n0 = len(f)
    f = f.merge(ctx, on="match_id", how="left")
    f = f.merge(ball_ctx, on=["match_id", "innings", "balls_bowled"], how="left")
    assert len(f) == n0, "merge changed the row count"

    first = f["innings"] == 1
    f["elo_diff"] = np.where(first, f["elo_batting_first"] - f["elo_chasing"],
                             f["elo_chasing"] - f["elo_batting_first"])   # batting side minus bowling side
    f["venue_par_vs_level"] = f["venue_par"] - f["comp_level"]
    f["pace_vs_par"] = np.where(first, f["runs"] + 8.0 * f["balls_left"] / 6 - f["venue_par"], np.nan)
    f["target_vs_par"] = np.where(first, np.nan, f["runs"] + f["runs_needed"] - 1 - f["venue_par"])
    for c in ["elo_batting_first", "elo_chasing", "elo_diff", "venue_par", "venue_chase_rate",
              "comp_level", "venue_par_vs_level", "pace_vs_par", "target_vs_par",
              "striker_rpb", "non_striker_rpb"]:
        f[c] = f[c].astype("float32")
    f["match_id"] = f["match_id"].astype("category")
    f.to_parquet(PROC / "features_v2.parquet", index=False)

    print(f"Rows: {len(f):,}; missing striker quality: {f['striker_rpb'].isna().mean():.2%}")
    print("\nContext feature summary (train split):")
    cols = ["elo_diff", "venue_par", "venue_chase_rate", "comp_level", "striker_rpb", "non_striker_rpb"]
    print(f[f["split"] == "train"][cols].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    # Does each feature, on its own, relate to winning? (first ball of each innings 1)
    s = f[(f["innings"] == 1) & (f["balls_bowled"] == 0) & (f["split"] == "train")].copy()
    s["elo_band"] = pd.qcut(s["elo_diff"], 5, duplicates="drop")
    print("\nBatting-first win rate by Elo difference (train, pre-match):")
    print(s.groupby("elo_band", observed=True)["label"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()

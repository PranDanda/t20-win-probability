"""Feature batch 2: four groups, all leakage-safe. Builds features_v4.parquet from features_v2.

Group A, set batters & momentum (state before each ball, this innings only):
  striker_faced, striker_runs_inn, non_striker_faced, partnership_runs, partnership_balls,
  runs_last18, wkts_last18
Group B, first-innings shape (chase rows only; known once the first innings ends):
  fi_wkts_lost, fi_dot_share, fi_last5_runs
Group C, toss & home familiarity (pre-match):
  bat_won_toss, toss_winner_chose_bat, bat_venue_share, bowl_venue_share, venue_familiarity_diff
Group D, Elo v2: home advantage, margin of victory, season regression, K tuned on
  TRAIN-period matches only (Elo's own predictions, 2010-2023):
  elo2_diff
"""
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from context_features import venue_key

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"


# ---------------------------------------------------------------- Group A
def group_a(d):
    d = d.sort_values(["match_id", "innings", "seq"]).copy()
    faced_ball = (d["extra_wides"] == 0).astype(int)
    d["faced_ball"] = faced_ball
    gk = ["match_id", "innings", "batter"]
    d["striker_faced"] = d.groupby(gk, observed=True)["faced_ball"].cumsum() - faced_ball
    d["striker_runs_inn"] = d.groupby(gk, observed=True)["runs_batter"].cumsum() - d["runs_batter"]
    # Non-striker's balls faced so far: look up their own running total at the last delivery
    # they faced before this one (as-of on the delivery sequence within the innings)
    d["key_bat"] = d["match_id"] + "|" + d["innings"].astype(str) + "|" + d["batter"]
    d["key_ns"] = d["match_id"] + "|" + d["innings"].astype(str) + "|" + d["non_striker"]
    d["faced_after"] = d["striker_faced"] + faced_ball
    d["gseq"] = d.groupby(["match_id", "innings"], observed=True).cumcount()
    left = d[["key_ns", "gseq"]].rename(columns={"key_ns": "key"}).reset_index()
    right = d[["key_bat", "gseq", "faced_after"]].rename(columns={"key_bat": "key"})
    right = right.sort_values("gseq"); left = left.sort_values("gseq")
    ns = pd.merge_asof(left, right, on="gseq", by="key", allow_exact_matches=False)
    d.loc[ns["index"], "non_striker_faced"] = ns["faced_after"].fillna(0).to_numpy()
    # Partnership since the last wicket
    wk_before = d.groupby(["match_id", "innings"], observed=True)["n_wickets"].cumsum() - d["n_wickets"]
    d["wk_before"] = wk_before
    pk = ["match_id", "innings", "wk_before"]
    d["partnership_runs"] = d.groupby(pk, observed=True)["runs_total"].cumsum() - d["runs_total"]
    d["partnership_balls"] = d.groupby(pk, observed=True)["legal_ball"].cumsum() - d["legal_ball"].astype(int)
    legal = d[d["legal_ball"]].copy()
    legal["balls_bowled"] = (legal["legal_balls_cum"] - 1).astype("int16")
    # Momentum over the previous 18 legal balls (state before this ball)
    legal["runs_before"] = legal["runs_cum"] - legal["runs_total"]
    legal["wkts_before"] = legal["wickets_cum"] - legal["n_wickets"]
    g = legal.groupby(["match_id", "innings"], observed=True)
    legal["runs_last18"] = legal["runs_before"] - g["runs_before"].shift(18).fillna(0)
    legal["wkts_last18"] = legal["wkts_before"] - g["wkts_before"].shift(18).fillna(0)
    cols = ["striker_faced", "striker_runs_inn", "non_striker_faced", "partnership_runs",
            "partnership_balls", "runs_last18", "wkts_last18"]
    return legal[["match_id", "innings", "balls_bowled"] + cols]


# ---------------------------------------------------------------- Group B
def group_b(d):
    i1 = d[d["innings"] == 1]
    g = i1.groupby("match_id", observed=True)
    out = pd.DataFrame({
        "fi_wkts_lost": g["n_wickets"].sum(),
        "fi_dot_share": i1[i1["legal_ball"]].assign(dot=lambda x: x["runs_total"] == 0)
                          .groupby("match_id", observed=True)["dot"].mean(),
        "fi_last5_runs": i1[i1["over"] >= 15].groupby("match_id", observed=True)["runs_total"].sum(),
    }).reset_index()
    out["fi_last5_runs"] = out["fi_last5_runs"].fillna(0)
    out["innings"] = 2
    return out


# ---------------------------------------------------------------- Group C
def group_c(m):
    m = m.sort_values(["date", "match_id"]).copy()
    m["vk"] = m["venue"].map(venue_key)
    team_n, team_venue_n = defaultdict(int), defaultdict(int)
    rows = []
    for date, day in m.groupby("date", sort=True):
        for r in day.itertuples():
            bat1 = r.team_batting_first
            chase = r.team2 if bat1 == r.team1 else r.team1
            sh = {t: team_venue_n[(t, r.vk)] / team_n[t] if team_n[t] else 0.0 for t in (bat1, chase)}
            cnt = {t: team_venue_n[(t, r.vk)] for t in (bat1, chase)}
            rows.append({"match_id": r.match_id, "bat1": bat1, "chase": chase,
                         "share_bat1": sh[bat1], "share_chase": sh[chase],
                         "cnt_bat1": cnt[bat1], "cnt_chase": cnt[chase],
                         "toss_winner": r.toss_winner, "toss_decision": r.toss_decision})
        for r in day.itertuples():
            for t in (r.team1, r.team2):
                team_n[t] += 1
                team_venue_n[(t, r.vk)] += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Group D
def margin_equiv(row):
    """Margin of victory in run-equivalents."""
    if row.winner == row.bat1:
        return max(row.first_total - row.second_total, 1)
    return max(1.25 * row.balls_left_end + 3.0 * row.wkts_left_end, 1)


def run_elo(m, K, hfa, mov, regress, record=False):
    elo = defaultdict(lambda: 1500.0)
    last_season = {}
    ll, n, pre = 0.0, 0, {}
    for r in m.itertuples():
        a, b = r.bat1, r.chase
        for t in (a, b):
            if regress and r.competition != "t20s" and last_season.get(t) not in (None, r.season):
                elo[t] = 1500 + (1 - regress) * (elo[t] - 1500)
            last_season[t] = r.season
        home = hfa * (1 if r.home_side == a else -1 if r.home_side == b else 0)
        ra, rb = elo[a] + home, elo[b]
        if record:
            pre[r.match_id] = (elo[a], elo[b], home)
        exp_a = 1 / (1 + 10 ** ((rb - ra) / 400))
        if r.winner not in (a, b):
            continue
        s = 1.0 if r.winner == a else 0.0
        if r.score_window:
            ll -= np.log(exp_a if s else 1 - exp_a); n += 1
        mult = np.log1p(r.margin) / np.log1p(20) if mov else 1.0
        elo[a] += K * mult * (s - exp_a)
        elo[b] -= K * mult * (s - exp_a)
    return (ll / max(n, 1)), pre


def group_d(m, c, d):
    tot = d.groupby(["match_id", "innings"], observed=True)["runs_cum"].last().unstack()
    end2 = d[d["innings"] == 2].groupby("match_id", observed=True).agg(
        lb=("legal_balls_cum", "max"), wk=("wickets_cum", "max"))
    mm = m.merge(c[["match_id", "bat1", "chase", "cnt_bat1", "cnt_chase"]], on="match_id")
    mm = mm.join(tot.rename(columns={1: "first_total", 2: "second_total"}), on="match_id")
    mm = mm.join(end2, on="match_id")
    mm["balls_left_end"] = (120 - mm["lb"]).clip(lower=0).fillna(0)
    mm["wkts_left_end"] = (10 - mm["wk"]).clip(lower=0).fillna(0)
    mm = mm.dropna(subset=["bat1"])
    mm["margin"] = mm.apply(margin_equiv, axis=1).fillna(1)
    # Home side: clearly the more familiar team at this ground (>=3 prior matches, 2x the other)
    hs = np.where((mm["cnt_bat1"] >= 3) & (mm["cnt_bat1"] >= 2 * mm["cnt_chase"] + 1), mm["bat1"],
         np.where((mm["cnt_chase"] >= 3) & (mm["cnt_chase"] >= 2 * mm["cnt_bat1"] + 1), mm["chase"], None))
    mm["home_side"] = hs
    mm["score_window"] = (mm["date"] >= "2010-01-01") & (mm["date"] <= "2023-12-31")  # TRAIN years
    mm = mm.sort_values(["date", "match_id"])

    grid = []
    for K, hfa, mov, reg in itertools.product([16, 24, 32, 48], [0, 30, 60], [False, True], [0, 0.25, 0.5]):
        ll, _ = run_elo(mm, K, hfa, mov, reg)
        grid.append({"K": K, "home_adv": hfa, "margin": mov, "season_regress": reg, "train_logloss": ll})
    grid = pd.DataFrame(grid).sort_values("train_logloss")
    grid.to_csv(RES / "elo_tuning.csv", index=False)
    base = grid[(grid.K == 24) & (grid.home_adv == 0) & (~grid.margin) & (grid.season_regress == 0)]
    print("Elo tuning on TRAIN-period matches (Elo's own pre-match log loss):")
    print(f"  week 4 settings (K=24, no home, no margin, no regression): {base.train_logloss.iloc[0]:.4f}")
    print(grid.head(5).round(4).to_string(index=False))
    best = grid.iloc[0]
    _, pre = run_elo(mm, best.K, best.home_adv, bool(best.margin), best.season_regress, record=True)
    e = pd.DataFrame([(k, *v) for k, v in pre.items()],
                     columns=["match_id", "elo2_bat1", "elo2_chase", "home_pts_bat1"])
    e["elo2_diff_bat1"] = e["elo2_bat1"] + e["home_pts_bat1"] - e["elo2_chase"]
    return e[["match_id", "elo2_diff_bat1"]]


def main():
    m = pd.read_parquet(PROC / "matches.parquet")
    m["match_id"] = m["match_id"].astype(str)
    d = pd.read_parquet(PROC / "deliveries.parquet", columns=[
        "match_id", "innings", "seq", "over", "legal_ball", "legal_balls_cum", "batter",
        "non_striker", "runs_batter", "runs_total", "extra_wides", "n_wickets",
        "runs_cum", "wickets_cum"])
    for c_ in ["match_id", "batter", "non_striker"]:
        d[c_] = d[c_].astype(str)

    print("Group A...", flush=True); A = group_a(d)
    print("Group B...", flush=True); B = group_b(d)
    print("Group C...", flush=True); C = group_c(m)
    print("Group D...", flush=True); D = group_d(m, C, d)

    f = pd.read_parquet(PROC / "features_v2.parquet")
    f["match_id"] = f["match_id"].astype(str)
    n0 = len(f)
    f = f.merge(A, on=["match_id", "innings", "balls_bowled"], how="left")
    f = f.merge(B, on=["match_id", "innings"], how="left")
    f = f.merge(C, on="match_id", how="left").merge(D, on="match_id", how="left")
    assert len(f) == n0
    first = f["innings"] == 1
    bat_team = np.where(first, f["bat1"], f["chase"])
    f["bat_won_toss"] = (f["toss_winner"] == bat_team).astype("float32")
    f["toss_winner_chose_bat"] = (f["toss_decision"] == "bat").astype("float32")
    f["bat_venue_share"] = np.where(first, f["share_bat1"], f["share_chase"]).astype("float32")
    f["bowl_venue_share"] = np.where(first, f["share_chase"], f["share_bat1"]).astype("float32")
    f["venue_familiarity_diff"] = f["bat_venue_share"] - f["bowl_venue_share"]
    f["elo2_diff"] = np.where(first, f["elo2_diff_bat1"], -f["elo2_diff_bat1"]).astype("float32")
    f = f.drop(columns=["bat1", "chase", "share_bat1", "share_chase", "cnt_bat1", "cnt_chase",
                        "toss_winner", "toss_decision", "elo2_diff_bat1"])
    f["match_id"] = f["match_id"].astype("category")
    f.to_parquet(PROC / "features_v4.parquet", index=False)
    new = ["striker_faced", "non_striker_faced", "partnership_runs", "runs_last18", "wkts_last18",
           "fi_wkts_lost", "fi_dot_share", "fi_last5_runs", "bat_won_toss",
           "venue_familiarity_diff", "elo2_diff"]
    print(f"\nRows {len(f):,}. Missing share by innings:")
    print(f.groupby("innings")[new].apply(lambda x: x.isna().mean()).round(3).T.to_string())
    print(f[f.split == "train"][new].describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()

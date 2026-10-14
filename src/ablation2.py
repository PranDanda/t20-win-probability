"""Batch 2 ablation: set batters/momentum, first-innings shape, toss/home, Elo v2.
Same LightGBM settings as week 4; each group added on its own, then all together.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import log_loss
from boosting import CONTEXT, MONOTONE, PARAMS, STATE

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

GROUPS = {
    "A set batters & momentum": {1: ["striker_faced", "striker_runs_inn", "non_striker_faced",
                                      "partnership_runs", "partnership_balls", "runs_last18", "wkts_last18"]},
    "B first-innings shape": {2: ["fi_wkts_lost", "fi_dot_share", "fi_last5_runs"]},
    "C toss & home": {1: ["bat_won_toss", "toss_winner_chose_bat", "bat_venue_share",
                          "bowl_venue_share", "venue_familiarity_diff"]},
    "D Elo v2 (replaces Elo)": {1: ["elo2_diff"]},
}
GROUPS["A set batters & momentum"][2] = GROUPS["A set batters & momentum"][1]
GROUPS["C toss & home"][2] = GROUPS["C toss & home"][1]
GROUPS["D Elo v2 (replaces Elo)"][2] = ["elo2_diff"]
MONO = dict(MONOTONE, elo2_diff=1)


def cols_for(inn, groups):
    base = STATE[inn] + CONTEXT[inn]
    extra = []
    for g in groups:
        extra += GROUPS[g].get(inn, [])
    if "D Elo v2 (replaces Elo)" in groups:
        base = [c for c in base if c != "elo_diff"]
    return base + extra


def run(train, valid, groups, seed=7):
    preds = pd.Series(np.nan, index=valid.index)
    for inn in (1, 2):
        cols = cols_for(inn, groups)
        tr, va = train[train.innings == inn], valid[valid.innings == inn]
        params = dict(PARAMS, seed=seed, monotone_constraints=[MONO.get(c, 0) for c in cols],
                      monotone_constraints_method="advanced")
        dtr = lgb.Dataset(tr[cols], tr["label"])
        dva = lgb.Dataset(va[cols], va["label"], reference=dtr)
        b = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.loc[va.index] = b.predict(va[cols], num_iteration=b.best_iteration)
    return preds.to_numpy()


def main():
    f = pd.read_parquet(PROC / "features_v4.parquet")
    train, valid = f[f.split == "train"], f[f.split == "valid"]
    y = valid["label"].to_numpy()
    scopes = {"all": np.ones(len(valid), bool),
              "innings 1": (valid.innings == 1).to_numpy(),
              "innings 2": (valid.innings == 2).to_numpy(),
              "pre-match": ((valid.innings == 1) & (valid.balls_bowled == 0)).to_numpy(),
              "powerplay": (valid.phase == "powerplay").to_numpy(),
              "middle": (valid.phase == "middle").to_numpy(),
              "death": (valid.phase == "death").to_numpy()}
    variants = {"week4 baseline": []}
    variants.update({g: [g] for g in GROUPS})
    variants["all four"] = list(GROUPS)
    rows = []
    for name, groups in variants.items():
        p = run(train, valid, groups)
        r = {"variant": name}
        r.update({s: log_loss(y[m], p[m]) for s, m in scopes.items()})
        rows.append(r)
        print(f"{name:28s} all {r['all']:.4f} inn1 {r['innings 1']:.4f} inn2 {r['innings 2']:.4f} "
              f"pre-match {r['pre-match']:.4f}", flush=True)
    out = pd.DataFrame(rows).set_index("variant")
    out.to_csv(RES / "ablation_batch2.csv")
    print("\nChange vs week 4 baseline (negative = better):")
    print((out - out.loc["week4 baseline"]).drop("week4 baseline").round(4).to_string())


if __name__ == "__main__":
    main()

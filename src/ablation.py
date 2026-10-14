"""Ablation: does adding bowling / batting resources improve the week 4 LightGBM model?

Each variant retrains LightGBM per innings with identical settings; only the feature list
changes. The baseline is run with 3 seeds to measure run-to-run noise.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import log_loss
from boosting import CONTEXT, MONOTONE, PARAMS, STATE

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"

BOWLING = ["bowl_now_econ", "bowl_now_wkt", "rem_bowl_econ", "rem_bowl_wkt"]
BATTING = ["bat_res_avg", "bat_res_sr"]
MONO = dict(MONOTONE, bowl_now_econ=1, bowl_now_wkt=-1, rem_bowl_econ=1, rem_bowl_wkt=-1,
            bat_res_avg=1, bat_res_sr=1)
VARIANTS = {"week4 (context)": [], "+ bowling": BOWLING, "+ batting": BATTING,
            "+ both": BOWLING + BATTING}


def run(train, valid, extra, seed):
    preds = pd.Series(np.nan, index=valid.index)
    trees = {}
    for inn in (1, 2):
        cols = STATE[inn] + CONTEXT[inn] + extra
        tr, va = train[train.innings == inn], valid[valid.innings == inn]
        params = dict(PARAMS, seed=seed, monotone_constraints=[MONO.get(c, 0) for c in cols],
                      monotone_constraints_method="advanced")
        dtr = lgb.Dataset(tr[cols], tr["label"])
        dva = lgb.Dataset(va[cols], va["label"], reference=dtr)
        b = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.loc[va.index] = b.predict(va[cols], num_iteration=b.best_iteration)
        trees[inn] = b.best_iteration
    return preds.to_numpy(), trees


def main():
    f = pd.read_parquet(PROC / "features_v3.parquet")
    train, valid = f[f.split == "train"], f[f.split == "valid"]
    y = valid["label"].to_numpy()
    scopes = {"all": np.ones(len(valid), bool),
              "innings 1": (valid.innings == 1).to_numpy(),
              "innings 2": (valid.innings == 2).to_numpy(),
              "powerplay": (valid.phase == "powerplay").to_numpy(),
              "middle": (valid.phase == "middle").to_numpy(),
              "death": (valid.phase == "death").to_numpy()}
    rows = []
    for name, extra in VARIANTS.items():
        seeds = [7, 8, 9] if name.startswith("week4") else [7]
        for seed in seeds:
            p, trees = run(train, valid, extra, seed)
            r = {"variant": name, "seed": seed, "trees_inn1": trees[1], "trees_inn2": trees[2]}
            r.update({s: log_loss(y[m], p[m]) for s, m in scopes.items()})
            rows.append(r)
            print(f"{name:16s} seed {seed}: all {r['all']:.4f} | inn1 {r['innings 1']:.4f} "
                  f"inn2 {r['innings 2']:.4f} | death {r['death']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "ablation_player_features.csv", index=False)
    base = out[out.variant.str.startswith("week4")]
    print("\nSeed noise in the baseline (max - min):",
          {s: round(base[s].max() - base[s].min(), 4) for s in scopes})
    b7 = base[base.seed == 7].iloc[0]
    print("\nChange vs baseline (seed 7); negative = better:")
    for _, r in out[(out.seed == 7) & ~out.variant.str.startswith("week4")].iterrows():
        print(f"  {r['variant']:12s}", {s: round(r[s] - b7[s], 4) for s in scopes})


if __name__ == "__main__":
    main()

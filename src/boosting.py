"""Week 4: gradient-boosted trees (LightGBM) vs logistic regression, with and without
context features. A 2x2 comparison per innings, scored on 2024 validation.

  model      x  features
  logistic      state (week 3 engineered)      | state + context
  lightgbm      state (plain columns)          | state + context

Saves models/, results/week4_metrics.csv, results/week4_importance.csv,
results/valid_predictions_v2.parquet
"""
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import accuracy, brier, log_loss
from logistic import engineer, fit

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"

STATE = {1: ["balls_left", "wickets_in_hand", "runs", "current_rr"],
         2: ["balls_left", "wickets_in_hand", "runs_needed", "required_rr", "current_rr"]}
CONTEXT = {1: ["elo2_diff", "venue_par", "venue_chase_rate", "comp_level", "pace_vs_par",
               "striker_rpb", "non_striker_rpb"],
           2: ["elo2_diff", "venue_par", "venue_chase_rate", "comp_level", "target_vs_par",
               "striker_rpb", "non_striker_rpb"]}
# Direction each feature is allowed to push the batting side's win chance:
# +1 only up, -1 only down, 0 free
MONOTONE = {"balls_left": 1, "wickets_in_hand": 1, "runs": 1, "runs_needed": -1,
            "required_rr": -1, "elo_diff": 1, "elo2_diff": 1, "pace_vs_par": 1, "target_vs_par": 0,
            "striker_rpb": 1, "non_striker_rpb": 1}

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=500,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=4, seed=7)


def train_lgb(Xtr, ytr, Xva, yva):
    params = dict(PARAMS, monotone_constraints=[MONOTONE.get(c, 0) for c in Xtr.columns],
                  monotone_constraints_method="advanced")
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return booster


def main():
    f = pd.read_parquet(PROC / "features_v4.parquet")   # Elo v2 adopted after batch 2
    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"].copy()
    imp_rows = []
    for inn in (1, 2):
        tr, vm = train[train["innings"] == inn], (valid["innings"] == inn)
        va = valid[vm]
        ytr, yva = tr["label"].to_numpy(), va["label"].to_numpy()

        # Logistic regression: week 3 engineered features, then + context
        for name, add in [("lr_state", []), ("lr_context", CONTEXT[inn])]:
            t = time.time()
            med = tr[add].astype(float).median()   # fill gaps with TRAINING medians only
            Xtr = pd.concat([engineer(tr, inn), tr[add].astype(float).fillna(med)], axis=1)
            Xva = pd.concat([engineer(va, inn), va[add].astype(float).fillna(med)], axis=1)
            mdl = fit(Xtr, ytr)
            valid.loc[vm, f"p_{name}"] = mdl.predict_proba(Xva)[:, 1]
            joblib.dump(mdl, MODELS / f"{name}_innings{inn}.joblib")
            print(f"innings {inn} {name:12s} {time.time()-t:5.1f}s", flush=True)

        # LightGBM: plain state columns (no hand engineering), then + context
        for name, cols in [("lgb_state", STATE[inn]), ("lgb_context", STATE[inn] + CONTEXT[inn])]:
            t = time.time()
            b = train_lgb(tr[cols], ytr, va[cols], yva)
            valid.loc[vm, f"p_{name}"] = b.predict(va[cols], num_iteration=b.best_iteration)
            b.save_model(str(MODELS / f"{name}_innings{inn}.txt"), num_iteration=b.best_iteration)
            gains = b.feature_importance("gain", iteration=b.best_iteration)
            for c, g in zip(cols, gains):
                imp_rows.append({"innings": inn, "model": name, "feature": c,
                                 "gain_share": g / gains.sum()})
            print(f"innings {inn} {name:12s} {time.time()-t:5.1f}s, {b.best_iteration} trees",
                  flush=True)

    y = valid["label"].to_numpy()
    names = ["lr_state", "lr_context", "lgb_state", "lgb_context"]
    scopes = {"all": np.ones(len(valid), bool),
              "innings 1": (valid["innings"] == 1).to_numpy(),
              "innings 2": (valid["innings"] == 2).to_numpy(),
              "powerplay": (valid["phase"] == "powerplay").to_numpy(),
              "middle": (valid["phase"] == "middle").to_numpy(),
              "death": (valid["phase"] == "death").to_numpy(),
              "innings 1, first ball": ((valid["innings"] == 1) & (valid["balls_bowled"] == 0)).to_numpy()}
    rows = []
    for n in names:
        p = valid[f"p_{n}"].to_numpy()
        for s, msk in scopes.items():
            rows.append({"model": n, "scope": s, "log_loss": log_loss(y[msk], p[msk]),
                         "brier": brier(y[msk], p[msk]), "accuracy": accuracy(y[msk], p[msk])})
    met = pd.DataFrame(rows)
    met.to_csv(RES / "week4_metrics.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(RES / "week4_importance.csv", index=False)
    keep = ["match_id", "date", "competition", "innings", "balls_bowled", "balls_left",
            "wickets_in_hand", "runs", "runs_needed", "phase", "label"] + [f"p_{n}" for n in names]
    valid[keep].to_parquet(RES / "valid_predictions_v2.parquet", index=False)

    print("\nValidation (2024) log loss:")
    print(met.pivot(index="scope", columns="model", values="log_loss")
          .loc[list(scopes)][names].round(4).to_string())
    print("\nFeature importance (share of total gain), LightGBM + context:")
    imp = pd.DataFrame(imp_rows)
    print(imp[imp.model == "lgb_context"].pivot(index="feature", columns="innings",
                                                values="gain_share").round(3).to_string())


if __name__ == "__main__":
    main()

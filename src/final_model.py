"""Week 5: refit the frozen model on all matches 2005-2024 and save it to models/final/.

Everything here is fixed by the pre-registered plan in LEARNING_NOTES.md (week 5, part 2):
features, settings, tree counts, 50/50 average, no calibration layer.
Also exposes predict(df) for scoring new balls (test set now, the website later).
"""
import json
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from boosting import CONTEXT, MONOTONE, PARAMS, STATE
from logistic import engineer, fit

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FINAL = ROOT / "models" / "final"
TREES = {1: 74, 2: 125}        # from validation early stopping (week 4 rerun with Elo v2)


def lr_matrix(df, inn, medians):
    ctx = df[CONTEXT[inn]].astype(float).fillna(medians)
    return pd.concat([engineer(df, inn), ctx], axis=1)


def train_final():
    FINAL.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(PROC / "features_v4.parquet")
    fit_set = f[f["split"].isin(["train", "valid"])]
    manifest = {"trained_on": "all usable matches 2005-01-01 to 2024-12-31",
                "created": date.today().isoformat(), "combine": "mean of logistic and lightgbm",
                "calibration": "none", "innings": {}}
    for inn in (1, 2):
        d = fit_set[fit_set["innings"] == inn]
        y = d["label"].to_numpy()
        med = d[CONTEXT[inn]].astype(float).median()
        lr = fit(lr_matrix(d, inn, med), y)
        joblib.dump({"model": lr, "medians": med}, FINAL / f"logistic_innings{inn}.joblib")
        cols = STATE[inn] + CONTEXT[inn]
        params = dict(PARAMS, monotone_constraints=[MONOTONE.get(c, 0) for c in cols],
                      monotone_constraints_method="advanced")
        b = lgb.train(params, lgb.Dataset(d[cols], y), num_boost_round=TREES[inn])
        b.save_model(str(FINAL / f"lightgbm_innings{inn}.txt"))
        manifest["innings"][inn] = {"rows": int(len(d)), "matches": int(d["match_id"].nunique()),
                                    "lightgbm_features": cols, "lightgbm_trees": TREES[inn],
                                    "logistic_extra_features": CONTEXT[inn]}
        print(f"innings {inn}: refit on {len(d):,} balls from {d['match_id'].nunique():,} matches")
    (FINAL / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Saved to", FINAL)


_cache = {}


def _load(inn):
    if inn not in _cache:
        lr = joblib.load(FINAL / f"logistic_innings{inn}.joblib")
        gb = lgb.Booster(model_file=str(FINAL / f"lightgbm_innings{inn}.txt"))
        _cache[inn] = (lr, gb)
    return _cache[inn]


def predict(df):
    """Win probability for the batting side, for rows of features_v4 format."""
    out = np.full(len(df), np.nan)
    for inn in (1, 2):
        m = (df["innings"] == inn).to_numpy()
        if not m.any():
            continue
        (lr, gb), d = _load(inn), df[m]
        p_lr = lr["model"].predict_proba(lr_matrix(d, inn, lr["medians"]))[:, 1]
        p_gb = gb.predict(d[STATE[inn] + CONTEXT[inn]])
        out[m] = (p_lr + p_gb) / 2
    return out


if __name__ == "__main__":
    train_final()

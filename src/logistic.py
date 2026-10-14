"""Week 3: logistic regression, one model per innings.

Two variants per innings, to show what feature engineering buys:
  raw         : the plain match-state columns, straight in
  engineered  : extra hand-built features that let a straight-line model follow curves

Scored on the 2024 validation set against the week 2 lookup table.
Saves models/, results/logistic_metrics.csv, results/valid_predictions.parquet
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import add_bins, brier, fit_lookup, log_loss, predict_lookup, accuracy

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"
MODELS.mkdir(exist_ok=True)

RAW = {
    1: ["balls_left", "wickets_in_hand", "runs", "current_rr"],
    2: ["balls_left", "wickets_in_hand", "runs_needed", "required_rr", "current_rr"],
}


def engineer(df: pd.DataFrame, innings: int) -> pd.DataFrame:
    """Hand-built features. Each one encodes a curve or interaction a linear model
    cannot discover on its own."""
    x = pd.DataFrame(index=df.index)
    bl = df["balls_left"].astype(float)
    wk = df["wickets_in_hand"].astype(float)
    frac_left = bl / 120                       # share of the innings remaining
    x["frac_left"] = frac_left
    x["wickets_in_hand"] = wk
    x["wk_x_left"] = wk * frac_left            # wickets matter more with lots of balls left
    x["wk_sq"] = wk ** 2                       # last few wickets are worth less than the first
    x["log_balls_left"] = np.log(bl)
    if innings == 1:
        runs = df["runs"].astype(float)
        crr = df["current_rr"].astype(float)
        # Projected total = runs so far + current rate over remaining overs,
        # but early in the innings the current rate is noise, so blend it with a
        # typical T20 rate (8 an over) in proportion to how much has been bowled.
        bowled = 1 - frac_left
        blended_rate = bowled * crr + (1 - bowled) * 8.0
        x["projected"] = runs + blended_rate * bl / 6
        x["runs"] = runs
        x["blended_rate"] = blended_rate
        x["proj_x_wk"] = x["projected"] * wk / 10
    else:
        need = df["runs_needed"].clip(lower=0).astype(float)
        rrr = df["required_rr"].clip(lower=0, upper=36).astype(float)
        x["runs_needed"] = need
        x["log_need"] = np.log1p(need)
        x["required_rr"] = rrr
        x["rrr_sq"] = rrr ** 2                 # each extra run an over gets harder, not linearly
        x["rrr_x_wk"] = rrr * wk               # a tough rate is gettable with wickets in hand
        x["rrr_x_left"] = rrr * frac_left      # a high rate early is less scary than at the death
        x["need_per_wicket"] = need / (wk + 1)
        x["rr_gap"] = (rrr - df["current_rr"].astype(float)).clip(-36, 36)
    return x


def fit(X, y):
    # Scaling puts every feature on the same footing (mean 0, sd 1) so the optimiser
    # converges and the size of each weight is comparable. C is the inverse of the
    # regularisation strength; 1.0 is a mild penalty on very large weights.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(X, y)
    return model


def main():
    f = add_bins(pd.read_parquet(PROC / "features.parquet"))
    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"].copy()

    base, p1, p2 = fit_lookup(train)
    valid["p_lookup"] = predict_lookup(valid, base, p1, p2)

    coef_rows = []
    for inn in (1, 2):
        tr, va = train[train["innings"] == inn], valid["innings"] == inn
        for variant in ("raw", "engineered"):
            Xtr = tr[RAW[inn]].astype(float) if variant == "raw" else engineer(tr, inn)
            Xva = (valid.loc[va, RAW[inn]].astype(float) if variant == "raw"
                   else engineer(valid.loc[va], inn))
            model = fit(Xtr, tr["label"])
            valid.loc[va, f"p_lr_{variant}"] = model.predict_proba(Xva)[:, 1]
            joblib.dump(model, MODELS / f"logistic_{variant}_innings{inn}.joblib")
            lr = model[-1]
            for name, w in zip(Xtr.columns, lr.coef_[0]):
                coef_rows.append({"innings": inn, "variant": variant, "feature": name,
                                  "weight_per_sd": round(float(w), 3)})
            coef_rows.append({"innings": inn, "variant": variant, "feature": "(intercept)",
                              "weight_per_sd": round(float(lr.intercept_[0]), 3)})
            print(f"innings {inn} {variant:10s}: trained on {len(tr):,} balls, "
                  f"{lr.n_iter_[0]} optimiser steps")

    y = valid["label"].to_numpy()
    models = {"lookup_table": "p_lookup", "logistic_raw": "p_lr_raw",
              "logistic_engineered": "p_lr_engineered"}
    scopes = {"all": slice(None), "innings 1": valid["innings"] == 1,
              "innings 2": valid["innings"] == 2, "powerplay": valid["phase"] == "powerplay",
              "middle": valid["phase"] == "middle", "death": valid["phase"] == "death"}
    rows = []
    for mname, col in models.items():
        p = valid[col].to_numpy()
        for sname, mask in scopes.items():
            m = np.ones(len(valid), bool) if isinstance(mask, slice) else mask.to_numpy()
            rows.append({"model": mname, "scope": sname, "log_loss": log_loss(y[m], p[m]),
                         "brier": brier(y[m], p[m]), "accuracy": accuracy(y[m], p[m])})
    met = pd.DataFrame(rows)
    met.to_csv(RES / "logistic_metrics.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(RES / "logistic_coefficients.csv", index=False)
    keep_cols = ["match_id", "date", "competition", "innings", "balls_left", "wickets_in_hand",
                 "runs", "runs_needed", "phase", "label", "p_lookup", "p_lr_raw", "p_lr_engineered"]
    valid[keep_cols].to_parquet(RES / "valid_predictions.parquet", index=False)

    print("\nValidation (2024) log loss, lower is better:")
    print(met.pivot(index="scope", columns="model", values="log_loss")
          .loc[list(scopes)][list(models)].round(4).to_string())
    print("\nAll balls:")
    print(met[met.scope == "all"].set_index("model")[["log_loss", "brier", "accuracy"]]
          .round(4).to_string())
    print("\nEngineered-model weights (per standard deviation of each feature):")
    c = pd.DataFrame(coef_rows)
    print(c[c.variant == "engineered"].pivot(index="feature", columns="innings",
                                              values="weight_per_sd").to_string())


if __name__ == "__main__":
    main()

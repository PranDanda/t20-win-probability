"""Week 5: does a calibration layer help the ensemble?

The calibrator must learn from predictions the models did NOT train on. Our models trained
on 2005-23, so 2024 validation predictions are fair game. To judge the calibrator honestly
we cross-fit WITHIN 2024: split matches into 5 groups, fit on 4, score on the 5th, rotate.
Candidates:
  none      : the raw ensemble
  platt     : logistic regression on the logit of the prediction (a smooth S-curve)
  platt_inn : Platt fitted separately per innings
  isotonic  : a free-form increasing step function
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from baseline import brier, log_loss

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models"
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    """Expected calibration error: average |predicted - actual| across 10% bands, weighted by size."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y); return self
    def predict(self, p):
        return self.m.predict_proba(logit(p).reshape(-1, 1))[:, 1]


class Iso:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y); return self
    def predict(self, p):
        return self.m.predict(p)


def main():
    v = pd.read_parquet(RES / "valid_predictions_v2.parquet")
    v["match_id"] = v["match_id"].astype(str)
    p = ((v["p_lr_context"] + v["p_lgb_context"]) / 2).to_numpy()
    y = v["label"].to_numpy()
    inn = v["innings"].to_numpy()
    groups = v["match_id"].to_numpy()

    cands = {"none": None, "platt": Platt, "platt_inn": "per_innings", "isotonic": Iso}
    out = {k: np.zeros_like(p) for k in cands}
    for tr, te in GroupKFold(n_splits=5).split(p, y, groups):
        out["none"][te] = p[te]
        out["platt"][te] = Platt().fit(p[tr], y[tr]).predict(p[te])
        out["isotonic"][te] = Iso().fit(p[tr], y[tr]).predict(p[te])
        for i in (1, 2):
            a, b = tr[inn[tr] == i], te[inn[te] == i]
            out["platt_inn"][b] = Platt().fit(p[a], y[a]).predict(p[b])
    rows = []
    for k, q in out.items():
        rows.append({"calibrator": k, "log_loss": log_loss(y, q), "brier": brier(y, q),
                     "ECE": ece(y, q),
                     "log_loss_inn1": log_loss(y[inn == 1], q[inn == 1]),
                     "log_loss_inn2": log_loss(y[inn == 2], q[inn == 2])})
    r = pd.DataFrame(rows).set_index("calibrator")
    r.to_csv(RES / "calibration_choice.csv")
    print("Cross-fitted within 2024 validation (5 match groups):")
    print(r.round(4).to_string())

    # Platt parameters on all of 2024, to interpret: slope < 1 = model too confident
    pl = Platt().fit(p, y)
    a, b = pl.m.coef_[0][0], pl.m.intercept_[0]
    print(f"\nPlatt fit on all 2024: calibrated logit = {a:.3f} x logit(p) + {b:.3f}")
    print("  slope < 1 means the model is overconfident; > 1 underconfident")
    joblib.dump(pl, MODELS / "calibrator_platt.joblib")


if __name__ == "__main__":
    main()

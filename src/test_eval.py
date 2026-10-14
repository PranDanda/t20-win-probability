"""Week 5: score the 2025-26 TEST set once, exactly as pre-registered.

Comparisons (all fitted on 2005-2024): coin flip, lookup table, week 3 logistic (scoreboard
only), and the frozen final model. Writes results/test_*.csv, test_predictions.parquet and
results/figures/test_calibration.png
"""
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import accuracy, add_bins, brier, fit_lookup, log_loss, predict_lookup
from calibration import ece
from final_model import predict
from logistic import engineer, fit

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"


def main():
    f = add_bins(pd.read_parquet(PROC / "features_v4.parquet"))
    hist, test = f[f["split"].isin(["train", "valid"])], f[f["split"] == "test"].copy()
    print(f"TEST: {test['match_id'].nunique():,} matches, {len(test):,} balls, "
          f"{test['date'].min().date()} to {test['date'].max().date()}\n")

    test["p_coin"] = 0.5
    base, p1, p2 = fit_lookup(hist)
    test["p_lookup"] = predict_lookup(test, base, p1, p2)
    for inn in (1, 2):
        h, m = hist[hist["innings"] == inn], test["innings"] == inn
        mdl = fit(engineer(h, inn), h["label"])
        test.loc[m, "p_lr_state"] = mdl.predict_proba(engineer(test[m], inn))[:, 1]
    test["p_final"] = predict(test)

    y = test["label"].to_numpy()
    models = {"coin flip": "p_coin", "lookup table": "p_lookup",
              "logistic, scoreboard only": "p_lr_state", "FINAL model": "p_final"}
    overall = pd.DataFrame([{"model": k, "log_loss": log_loss(y, test[c].to_numpy()),
                             "brier": brier(y, test[c].to_numpy()),
                             "accuracy": accuracy(y, test[c].to_numpy()),
                             "ECE": ece(y, test[c].to_numpy())} for k, c in models.items()]).set_index("model")
    overall.to_csv(RES / "test_overall.csv")
    print("Overall (test 2025-26):")
    print(overall.round(4).to_string())

    pre = (test["innings"] == 1) & (test["balls_bowled"] == 0)
    slices = {("innings", "1"): test["innings"] == 1, ("innings", "2"): test["innings"] == 2,
              ("pre-match", "first ball"): pre}
    for ph in ["powerplay", "middle", "death"]:
        slices[("phase", ph)] = test["phase"] == ph
    for c in sorted(test["competition"].astype(str).unique()):
        slices[("league", c)] = test["competition"].astype(str) == c
    rows = []
    for (kind, name), msk in slices.items():
        msk = msk.to_numpy()
        r = {"slice": kind, "value": name, "balls": int(msk.sum())}
        for k, c in models.items():
            r[k] = log_loss(y[msk], test[c].to_numpy()[msk])
        rows.append(r)
    sl = pd.DataFrame(rows)
    sl["final_minus_scoreboard"] = sl["FINAL model"] - sl["logistic, scoreboard only"]
    sl.to_csv(RES / "test_slices.csv", index=False)
    print("\nLog loss by slice:")
    print(sl.round(4).to_string(index=False))

    fin = overall.loc["FINAL model"]
    c1 = fin["log_loss"] < min(overall.loc["lookup table", "log_loss"],
                               overall.loc["logistic, scoreboard only", "log_loss"])
    c2 = fin["ECE"] < 0.02
    worst = sl["final_minus_scoreboard"].max()
    c3 = worst <= 0.01
    print("\nPre-registered success criteria:")
    print(f"  1. beats lookup and scoreboard-only logistic: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. calibration error < 2% (got {fin['ECE']:.2%}): {'PASS' if c2 else 'FAIL'}")
    print(f"  3. no slice worse than scoreboard-only by > 0.01 (worst {worst:+.4f}, "
          f"{sl.loc[sl['final_minus_scoreboard'].idxmax(), 'value']}): {'PASS' if c3 else 'FAIL'}")

    keep = ["match_id", "date", "competition", "innings", "balls_bowled", "balls_left",
            "wickets_in_hand", "runs", "runs_needed", "phase", "label", "p_lookup",
            "p_lr_state", "p_final"]
    test[keep].to_parquet(RES / "test_predictions.parquet", index=False)

    # Calibration table for the chart
    b = pd.cut(test["p_final"], np.linspace(0, 1, 11), include_lowest=True)
    cal = test.groupby(b, observed=True).agg(n=("label", "size"), pred=("p_final", "mean"),
                                             actual=("label", "mean"))
    cal.to_csv(RES / "test_calibration.csv")
    print("\nCalibration of the final model on test:")
    print(cal.round(3).to_string())


if __name__ == "__main__":
    main()

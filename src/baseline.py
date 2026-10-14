"""Week 2 baselines, scored on the 2024 validation set (test set stays locked).

1. Coin flip      : 0.5 for every ball
2. Base rate      : training-set win rate of the batting team, per innings
3. Lookup table   : training-set win rate for similar match states, with smoothing

Writes results/baseline_metrics.csv and results/baseline_calibration.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

K = 30  # smoothing strength: a cell needs ~30 balls before its own rate dominates
EPS = 1e-6


def add_bins(f):
    f = f.copy()
    f["b_overs_left"] = ((f["balls_left"] - 1) // 6).astype("int16")      # 0..19
    f["b_wkts"] = f["wickets_in_hand"].astype("int16")
    # 2nd innings: runs needed in 5-run buckets (capped); 1st innings: runs scored in 5s
    need = f["runs_needed"].fillna(0).clip(lower=0, upper=250)
    f["b_runs"] = np.where(f["innings"] == 2, need // 5, f["runs"].clip(upper=300) // 5).astype("int16")
    return f


def fit_lookup(train):
    """Hierarchical smoothed rates: innings -> (overs, wkts) -> (overs, wkts, runs)."""
    base = train.groupby("innings")["label"].mean().rename("p0")
    lvl1 = train.groupby(["innings", "b_overs_left", "b_wkts"])["label"].agg(["sum", "count"])
    lvl1 = lvl1.join(base, on="innings")
    lvl1["p1"] = (lvl1["sum"] + K * lvl1["p0"]) / (lvl1["count"] + K)
    lvl2 = train.groupby(["innings", "b_overs_left", "b_wkts", "b_runs"])["label"].agg(["sum", "count"])
    lvl2 = lvl2.join(lvl1["p1"], on=["innings", "b_overs_left", "b_wkts"])
    lvl2["p2"] = (lvl2["sum"] + K * lvl2["p1"]) / (lvl2["count"] + K)
    return base, lvl1["p1"], lvl2["p2"]


def predict_lookup(df, base, p1, p2):
    out = df.join(p2, on=["innings", "b_overs_left", "b_wkts", "b_runs"])
    out = out.join(p1, on=["innings", "b_overs_left", "b_wkts"])
    out = out.join(base, on="innings")
    # Unseen state -> fall back to the coarser level
    return out["p2"].fillna(out["p1"]).fillna(out["p0"]).to_numpy()


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    return float(np.mean((p >= 0.5) == (y == 1)))


def main():
    f = pd.read_parquet(PROC / "features.parquet")
    f = add_bins(f)
    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"].copy()
    y = valid["label"].to_numpy()

    base, p1, p2 = fit_lookup(train)
    preds = {
        "coin_flip": np.full(len(valid), 0.5),
        "base_rate": valid["innings"].map(base).to_numpy(),
        "lookup_table": predict_lookup(valid, base, p1, p2),
    }

    rows = []
    for name, p in preds.items():
        for scope, mask in [("all", np.ones(len(valid), bool)),
                            ("innings 1", (valid["innings"] == 1).to_numpy()),
                            ("innings 2", (valid["innings"] == 2).to_numpy()),
                            ("powerplay", (valid["phase"] == "powerplay").to_numpy()),
                            ("middle", (valid["phase"] == "middle").to_numpy()),
                            ("death", (valid["phase"] == "death").to_numpy())]:
            rows.append({"model": name, "scope": scope, "n": int(mask.sum()),
                         "log_loss": log_loss(y[mask], p[mask]),
                         "brier": brier(y[mask], p[mask]),
                         "accuracy": accuracy(y[mask], p[mask])})
    met = pd.DataFrame(rows)
    met.to_csv(RES / "baseline_metrics.csv", index=False)
    print("Validation (2024) scores, lower log loss / Brier is better:\n")
    print(met.pivot(index="scope", columns="model", values="log_loss")
          .loc[["all", "innings 1", "innings 2", "powerplay", "middle", "death"]]
          [["coin_flip", "base_rate", "lookup_table"]].round(4).to_string())
    print()
    print(met[met.scope == "all"].set_index("model")[["log_loss", "brier", "accuracy"]].round(4).to_string())

    # Calibration of the lookup table: predicted vs actual in 10% bands
    valid["p"] = preds["lookup_table"]
    valid["band"] = pd.cut(valid["p"], np.linspace(0, 1, 11), include_lowest=True)
    cal = valid.groupby("band", observed=True).agg(
        n=("label", "size"), mean_predicted=("p", "mean"), actual_win_rate=("label", "mean"))
    cal.to_csv(RES / "baseline_calibration.csv")
    print("\nLookup-table calibration (validation):")
    print(cal.round(3).to_string())
    print(f"\nLookup cells: {len(p2):,} detailed, {len(p1):,} coarse")


if __name__ == "__main__":
    main()

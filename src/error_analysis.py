"""Week 5: where does the ensemble go wrong? (2024 validation only)"""
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import log_loss

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EPS = 1e-6


def main():
    v = pd.read_parquet(RES / "valid_predictions_v2.parquet")
    v["match_id"] = v["match_id"].astype(str)
    v["p"] = (v["p_lr_context"] + v["p_lgb_context"]) / 2
    p = np.clip(v["p"], EPS, 1 - EPS)
    v["ll"] = -(v["label"] * np.log(p) + (1 - v["label"]) * np.log(1 - p))
    out = []

    def table(title, by):
        t = v.groupby(by, observed=True).agg(balls=("ll", "size"), log_loss=("ll", "mean"),
                                             mean_pred=("p", "mean"), actual=("label", "mean"))
        t["share_of_total_loss"] = v.groupby(by, observed=True)["ll"].sum() / v["ll"].sum()
        print(f"\n{title}\n{t.round(3).to_string()}")
        out.append((title, t))

    table("By competition", "competition")
    table("By innings and phase", ["innings", "phase"])
    v["conf"] = pd.cut((v["p"] - 0.5).abs() * 2, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True,
                       labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    table("By how confident the model was (favourite's probability)", "conf")

    # Matches the model got most wrong: average log loss per ball across the match
    mm = v.groupby("match_id").agg(date=("date", "first"), comp=("competition", "first"),
                                   mean_ll=("ll", "mean"), balls=("ll", "size"))
    # favourite's lowest point for the eventual winner
    v["p_winner"] = np.where(v["label"] == 1, v["p"], 1 - v["p"])
    mm["winner_low"] = v.groupby("match_id")["p_winner"].min()
    worst = mm.sort_values("mean_ll", ascending=False).head(8)
    print("\nMatches the model got most wrong (lowest point = eventual winner's lowest probability):")
    print(worst.round(3).to_string())
    # Pre-match favourite record
    pre = v[(v["innings"] == 1) & (v["balls_bowled"] == 0)]
    fav_right = ((pre["p"] >= 0.5) == (pre["label"] == 1)).mean()
    print(f"\nPre-match favourite won {fav_right:.1%} of {len(pre)} matches; "
          f"pre-match log loss {pre['ll'].mean():.4f}")
    big_comebacks = (mm["winner_low"] < 0.05).mean()
    print(f"Matches won by a side the model had below 5% at some point: {big_comebacks:.1%}")
    with pd.ExcelWriter(RES / "error_analysis.xlsx") if False else open(RES / "error_analysis.txt", "w") as fh:
        for title, t in out:
            fh.write(f"{title}\n{t.round(4).to_string()}\n\n")
        fh.write(worst.round(4).to_string())


if __name__ == "__main__":
    main()

"""Sanity checks: does the chase model move the right way when one thing changes?"""
import joblib
import numpy as np
import pandas as pd

from logistic import engineer

m2 = joblib.load("../models/logistic_engineered_innings2.joblib")


def chase_states(need, balls, wk, target=160):
    need, balls, wk = (np.asarray(a, float) for a in np.broadcast_arrays(need, balls, wk))
    bowled = 120 - balls
    crr = np.where(bowled > 0, (target - need) / np.maximum(bowled, 1) * 6, 0.0)
    return pd.DataFrame({"balls_left": balls, "wickets_in_hand": wk, "runs_needed": need,
                         "required_rr": need / (balls / 6), "current_rr": crr})


def p(need, balls, wk):
    return m2.predict_proba(engineer(chase_states(need, balls, wk), 2))[:, 1] * 100


def show(title, key, values, probs):
    print(title)
    print("   " + "  ".join(f"{k}={v:>3}: {pr:5.1f}%" for k, v, pr in
                            zip([key] * len(values), values, probs)))


def main():
    w = np.array([10, 8, 6, 4, 2, 1]); show("Need 30 off 30, vary wickets in hand", "wkts", w, p(30, 30, w))
    n = np.array([10, 20, 30, 40, 50, 60, 80]); show("6 wkts, 30 balls left, vary runs needed", "need", n, p(n, 30, 6))
    b = np.array([1, 2, 3, 6, 12, 24]); show("Need 9, 6 wkts, vary balls left", "balls", b, p(9, b, 6))
    n = np.array([1, 2, 4, 6, 7, 10]); show("Last ball, 5 wkts, vary runs needed", "need", n, p(n, 1, 5))

    # Whole grid: more wickets must never lower the chaser's chances,
    # and more runs needed must never raise them.
    B, N, W = np.meshgrid(np.arange(1, 121), np.arange(1, 200), np.arange(1, 11), indexing="ij")
    probs = p(N.ravel(), B.ravel(), W.ravel()).reshape(B.shape)
    viol_w = (np.diff(probs, axis=2) < -1e-6)
    viol_n = (np.diff(probs, axis=1) > 1e-6)
    print(f"\nGrid of {probs.size:,} chase states")
    print(f"  extra wicket LOWERS win chance: {viol_w.sum():,} of {viol_w.size:,} pairs")
    print(f"  extra run needed RAISES win chance: {viol_n.sum():,} of {viol_n.size:,} pairs")
    if viol_w.any():
        i = np.argwhere(viol_w)[:5]
        for bi, ni, wi in i:
            print(f"    e.g. {ni+1} needed off {bi+1}: {wi+1} wkts {probs[bi,ni,wi]:.1f}% -> "
                  f"{wi+2} wkts {probs[bi,ni,wi+1]:.1f}%")
    if viol_n.any():
        i = np.argwhere(viol_n)[:5]
        for bi, ni, wi in i:
            print(f"    e.g. off {bi+1} balls, {wi+1} wkts: need {ni+1} {probs[bi,ni,wi]:.1f}% -> "
                  f"need {ni+2} {probs[bi,ni+1,wi]:.1f}%")


if __name__ == "__main__":
    main()

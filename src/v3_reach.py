"""Version 3: a 'chance of reaching the target' feature for the chase.

Why v2 was not enough: exp_runs_left is an AVERAGE. With 2 wickets left and 34 balls, tail-enders
average ~23 more runs, so needing 17 looks comfortable. But the average hides a skewed spread:
often the last two wickets fall for 5, sometimes they add 50. What matters is the probability of
scoring at least 17, not the mean. Diagnosis (src/diag_v2b.py): in all 20 years only 11 chases
reached '2 wickets left, need 12-22 off 25-45 balls'; chasers won 27%, v2 said 71%.

Feature (innings 2 only):
  reach_logit = logit P(remaining runs >= runs_needed | balls_left, wickets_in_hand)
The probability comes from the empirical distribution of runs still to come in FIRST innings of the
fitting years, pooled over balls_left +/-3 (window widened until >= 200 rows), then forced to be
monotone (more balls or wickets never lowers it). Kept: v2's exp_runs_left and need_ratio.
Acceptance rule (written before running, same as v2): improves the validation tail slice versus v2
AND overall validation log loss no worse than v1 by more than 0.001.
"""
import json
import numpy as np
import pandas as pd
import joblib
import v2_resources as v2
from baseline import log_loss

MAXN = 300


def reach_table(fit_rows, first_totals):
    i1 = fit_rows[fit_rows["innings"] == 1]
    rem = (i1["match_id"].map(first_totals) - i1["runs"]).to_numpy()
    bl, wk = i1["balls_left"].to_numpy(), i1["wickets_in_hand"].to_numpy()
    P = np.zeros((120, 10, MAXN + 1), dtype="float32")
    n = np.arange(MAXN + 1)
    for w in range(1, 11):
        rw, bw = rem[wk == w], bl[wk == w]
        for b in range(1, 121):
            h = 3
            while True:
                vals = rw[np.abs(bw - b) <= h]
                if len(vals) >= 200 or h >= 30:
                    break
                h += 3
            if len(vals) == 0:
                continue
            s = np.sort(vals)
            P[b - 1, w - 1] = 1 - np.searchsorted(s, n, side="left") / len(s)
    P = np.maximum.accumulate(np.maximum.accumulate(P, axis=0), axis=1)
    return P


def add_reach(df, P):
    df = df.copy()
    bl = df["balls_left"].clip(1, 120).to_numpy() - 1
    wk = df["wickets_in_hand"].clip(1, 10).to_numpy() - 1
    need = df["runs_needed"].fillna(0).clip(0, MAXN).astype(int).to_numpy()
    p = np.clip(P[bl, wk, need], 0.005, 0.995)
    df["reach_logit"] = np.where(df["innings"] == 2, np.log(p / (1 - p)), np.nan).astype("float32")
    return df


def main():
    v2.EXTRA = {1: ["exp_runs_left", "res_projected"], 2: ["exp_runs_left", "need_ratio", "reach_logit"]}
    v2.MONO = dict(v2.MONO, reach_logit=1)
    f = pd.read_parquet(v2.PROC / "features_v4.parquet")
    d = pd.read_parquet(v2.PROC / "deliveries.parquet", columns=["match_id", "innings", "runs_cum"])
    ft = d[d["innings"] == 1].groupby("match_id", observed=True)["runs_cum"].last()
    ft.index = ft.index.astype(str)
    f["match_id"] = f["match_id"].astype(str)

    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"]
    tab, P = v2.resource_table(train, ft), reach_table(train, ft)
    print("P(score >= need) from 2005-23 first innings, 34 balls left:")
    print(pd.DataFrame({w: P[33, w - 1, [10, 17, 25, 40]] for w in (1, 2, 3, 6)}, index=[10, 17, 25, 40]).round(2))
    tr, va = add_reach(v2.add_features(train, tab), P), add_reach(v2.add_features(valid, tab), P)
    p3, trees = np.full(len(va), np.nan), {}
    for inn in (1, 2):
        m = v2.train_pair(tr[tr.innings == inn], inn, valid=va[va.innings == inn])
        trees[inn] = m["gb"].best_iteration
        p3[(va.innings == inn).to_numpy()] = v2.predict_pair(m, va[va.innings == inn], inn)
    y = va["label"].to_numpy()
    v2p = pd.read_parquet(v2.RES / "valid_predictions_v2model.parquet")
    v2p["match_id"] = v2p["match_id"].astype(str)
    p2 = va[["match_id", "innings", "balls_bowled"]].merge(v2p, how="left")["p_v2"].to_numpy()
    v1 = pd.read_parquet(v2.RES / "valid_predictions_v2.parquet")
    v1["match_id"] = v1["match_id"].astype(str)
    v1 = va[["match_id", "innings", "balls_bowled"]].merge(v1, how="left")
    p1 = ((v1["p_lr_context"] + v1["p_lgb_context"]) / 2).to_numpy()
    ll1, ll2, ll3 = log_loss(y, p1), log_loss(y, p2), log_loss(y, p3)
    print(f"\nValidation 2024 log loss: v1 {ll1:.4f}, v2 {ll2:.4f}, v3 {ll3:.4f}; trees {trees}")
    t2, t3 = v2.tail_report(va, p2, "v2"), v2.tail_report(va, p3, "v3")
    ok = t3 < t2 and ll3 - ll1 <= 0.001
    print(f"\nAcceptance: tail {t2:.4f} -> {t3:.4f}, overall vs v1 {ll3-ll1:+.4f}: {'ACCEPT' if ok else 'REJECT'}")
    if not ok:
        return

    fitset = f[f["split"].isin(["train", "valid"])]
    tab_f, P_f = v2.resource_table(fitset, ft), reach_table(fitset, ft)
    fs = add_reach(v2.add_features(fitset, tab_f), P_f)
    test = add_reach(v2.add_features(f[f["split"] == "test"], tab_f), P_f)
    out = v2.MODELS / "final_v3"
    out.mkdir(parents=True, exist_ok=True)
    tab_f.to_csv(out / "resource_table.csv")
    np.save(out / "reach_table.npy", P_f)
    pt = np.full(len(test), np.nan)
    for inn in (1, 2):
        m = v2.train_pair(fs[fs.innings == inn], inn, trees=trees[inn])
        joblib.dump({k: m[k] for k in ("lr", "med", "lr_cols", "cols")}, out / f"logistic_innings{inn}.joblib")
        m["gb"].save_model(str(out / f"lightgbm_innings{inn}.txt"))
        pt[(test.innings == inn).to_numpy()] = v2.predict_pair(m, test[test.innings == inn], inn)
    (out / "manifest.json").write_text(json.dumps({"version": 3, "trained_on": "2005-2024", "adds": v2.EXTRA,
        "trees": trees, "note": "post-test fix; 2025-26 scores are not out-of-sample for model design"}, indent=2))
    t = pd.read_parquet(v2.RES / "test_predictions_v2.parquet")
    t["match_id"] = t["match_id"].astype(str)
    t = test[["match_id", "innings", "balls_bowled"]].assign(p_v3=pt).merge(t, on=["match_id", "innings", "balls_bowled"])
    yt = t["label"].to_numpy()
    print(f"\nTest 2025-26 (POST-TEST): v1 {log_loss(yt, t.p_final.to_numpy()):.4f}, "
          f"v2 {log_loss(yt, t.p_v2.to_numpy()):.4f}, v3 {log_loss(yt, t.p_v3.to_numpy()):.4f}")
    v2.tail_report(test, pt, "v3 test tail")
    t.to_parquet(v2.RES / "test_predictions_v3.parquet", index=False)
    va[["match_id", "innings", "balls_bowled"]].assign(p_v3=p3).to_parquet(v2.RES / "valid_predictions_v3model.parquet", index=False)
    k = t[(t.match_id == "1473468") & (t.innings == 2) & t.balls_bowled.isin([60, 80, 86, 90])]
    print("\nPBKS v KKR (KKR chasing):\n", k[["balls_bowled", "p_final", "p_v2", "p_v3"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

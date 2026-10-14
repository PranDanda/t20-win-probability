"""Version 2: DLS-style 'resources remaining' features, to fix late-chase overconfidence.

Found while building the site: with 1-2 wickets left the frozen model was far too sure of the
chasing side (e.g. 97.6% at 95/8 needing 17 off 33; such sides historically win ~26%).

Fix: an expected-runs-remaining table, E[runs still to come | balls left, wickets in hand],
estimated from FIRST innings of the fitting years only (a first innings always runs to the
end, so remaining runs are fully observed), smoothed toward the overs-left level.
Features:
  exp_runs_left        expected runs the batting side can still score
  need_ratio (chase)   runs needed / exp_runs_left   (DLS-style pressure)
  res_projected (1st)  runs so far + exp_runs_left - venue par
Acceptance rule (set before running): fixes the tail-end slice on 2024 validation AND overall
validation log loss no worse than v1 by more than 0.001.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import log_loss
from boosting import CONTEXT, MONOTONE, PARAMS, STATE
from logistic import engineer, fit

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MODELS = ROOT / "data" / "processed", ROOT / "results", ROOT / "models"
K = 30
EXTRA = {1: ["exp_runs_left", "res_projected"], 2: ["exp_runs_left", "need_ratio"]}
MONO = dict(MONOTONE, exp_runs_left=1, need_ratio=-1, res_projected=1)


def resource_table(fit_rows, first_totals):
    i1 = fit_rows[fit_rows["innings"] == 1].copy()
    i1["remaining"] = i1["match_id"].map(first_totals) - i1["runs"]
    i1["ol"] = (i1["balls_left"] - 1) // 6
    coarse = i1.groupby(["ol", "wickets_in_hand"])["remaining"].mean().rename("c")
    fine = i1.groupby(["balls_left", "wickets_in_hand"])["remaining"].agg(["sum", "count"]).reset_index()
    fine["ol"] = (fine["balls_left"] - 1) // 6
    fine = fine.join(coarse, on=["ol", "wickets_in_hand"])
    fine["exp"] = (fine["sum"] + K * fine["c"]) / (fine["count"] + K)
    grid = pd.MultiIndex.from_product([range(1, 121), range(1, 11)], names=["balls_left", "wickets_in_hand"])
    t = fine.set_index(["balls_left", "wickets_in_hand"])["exp"].reindex(grid)
    # fill unseen cells from the coarse level, then enforce: more balls or wickets never fewer runs
    t = t.reset_index()
    t["ol"] = (t["balls_left"] - 1) // 6
    t["exp"] = t["exp"].fillna(t.join(coarse, on=["ol", "wickets_in_hand"])["c"])
    m = t.pivot(index="balls_left", columns="wickets_in_hand", values="exp").sort_index()
    m = m.interpolate(limit_direction="both").cummax(axis=0).cummax(axis=1)
    return m


def add_features(df, table):
    df = df.copy()
    bl = df["balls_left"].clip(1, 120).to_numpy()
    wk = df["wickets_in_hand"].clip(1, 10).to_numpy()
    df["exp_runs_left"] = table.to_numpy()[bl - 1, wk - 1].astype("float32")
    df["need_ratio"] = np.where(df["innings"] == 2,
                                (df["runs_needed"].clip(lower=0) / df["exp_runs_left"].clip(lower=1)).clip(upper=5),
                                np.nan).astype("float32")
    df["res_projected"] = np.where(df["innings"] == 1, df["runs"] + df["exp_runs_left"] - df["venue_par"],
                                   np.nan).astype("float32")
    return df


def train_pair(d, inn, trees=None, valid=None):
    lr_cols = CONTEXT[inn] + EXTRA[inn]
    med = d[lr_cols].astype(float).median()
    X = pd.concat([engineer(d, inn), d[lr_cols].astype(float).fillna(med)], axis=1)
    lr = fit(X, d["label"])
    cols = STATE[inn] + CONTEXT[inn] + EXTRA[inn]
    params = dict(PARAMS, monotone_constraints=[MONO.get(c, 0) for c in cols],
                  monotone_constraints_method="advanced")
    dtr = lgb.Dataset(d[cols], d["label"])
    if valid is not None:
        dva = lgb.Dataset(valid[cols], valid["label"], reference=dtr)
        gb = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        gb = lgb.train(params, dtr, trees)
    return {"lr": lr, "med": med, "lr_cols": lr_cols, "gb": gb, "cols": cols}


def predict_pair(mdl, d, inn):
    X = pd.concat([engineer(d, inn), d[mdl["lr_cols"]].astype(float).fillna(mdl["med"])], axis=1)
    p_lr = mdl["lr"].predict_proba(X)[:, 1]
    p_gb = mdl["gb"].predict(d[mdl["cols"]], num_iteration=mdl["gb"].best_iteration or None)
    return (p_lr + p_gb) / 2


def tail_report(d, p, title):
    c = d[(d["innings"] == 2) & (d["wickets_in_hand"] <= 3) & (d["runs_needed"] <= 30) & (d["balls_left"] >= 6)].copy()
    c["p"] = p[(d["innings"] == 2).to_numpy() & (d["wickets_in_hand"] <= 3).to_numpy()
               & (d["runs_needed"] <= 30).to_numpy() & (d["balls_left"] >= 6).to_numpy()]
    g = c.groupby("wickets_in_hand").agg(balls=("label", "size"), matches=("match_id", "nunique"),
                                         actual=("label", "mean"), model=("p", "mean"))
    y = c["label"].to_numpy()
    print(f"  {title}: tail slice log loss {log_loss(y, c['p'].to_numpy()):.4f}")
    print(g.round(3).to_string())
    return log_loss(y, c["p"].to_numpy())


def main():
    f = pd.read_parquet(PROC / "features_v4.parquet")
    d = pd.read_parquet(PROC / "deliveries.parquet", columns=["match_id", "innings", "runs_cum"])
    first_totals = d[d["innings"] == 1].groupby("match_id", observed=True)["runs_cum"].last()
    first_totals.index = first_totals.index.astype(str)
    f["match_id"] = f["match_id"].astype(str)

    # ---- Validation run: table and models from 2005-2023 only
    train, valid = f[f["split"] == "train"], f[f["split"] == "valid"]
    tab = resource_table(train, first_totals)
    print("Expected runs remaining (from 2005-23 first innings), selected cells:")
    print(tab.loc[[120, 60, 33, 18, 6], [10, 6, 4, 2, 1]].round(1).to_string())
    tr, va = add_features(train, tab), add_features(valid, tab)
    p_v2 = np.full(len(va), np.nan)
    trees = {}
    for inn in (1, 2):
        mdl = train_pair(tr[tr.innings == inn], inn, valid=va[va.innings == inn])
        trees[inn] = mdl["gb"].best_iteration
        p_v2[(va.innings == inn).to_numpy()] = predict_pair(mdl, va[va.innings == inn], inn)
    v1 = pd.read_parquet(RES / "valid_predictions_v2.parquet")
    v1["match_id"] = v1["match_id"].astype(str)
    v1 = va[["match_id", "innings", "balls_bowled"]].merge(
        v1[["match_id", "innings", "balls_bowled", "p_lr_context", "p_lgb_context"]],
        on=["match_id", "innings", "balls_bowled"], how="left")
    p_v1 = ((v1["p_lr_context"] + v1["p_lgb_context"]) / 2).to_numpy()
    y = va["label"].to_numpy()
    ll1, ll2 = log_loss(y, p_v1), log_loss(y, p_v2)
    print(f"\nValidation 2024 overall log loss: v1 {ll1:.4f}, v2 {ll2:.4f} (change {ll2-ll1:+.4f}); trees {trees}")
    t1 = tail_report(va, p_v1, "v1")
    t2 = tail_report(va, p_v2, "v2")
    accept = (t2 < t1) and (ll2 - ll1 <= 0.001)
    print(f"\nAcceptance rule: tail slice improved ({t1:.4f} -> {t2:.4f}) and overall within +0.001: "
          f"{'ACCEPT' if accept else 'REJECT'}")
    if not accept:
        return

    # ---- Final v2: table and models from 2005-2024, fixed tree counts
    fitset = f[f["split"].isin(["train", "valid"])]
    tab_f = resource_table(fitset, first_totals)
    fs = add_features(fitset, tab_f)
    out = ROOT / "models" / "final_v2"
    out.mkdir(parents=True, exist_ok=True)
    tab_f.to_csv(out / "resource_table.csv")
    test = add_features(f[f["split"] == "test"], tab_f)
    p_test = np.full(len(test), np.nan)
    for inn in (1, 2):
        mdl = train_pair(fs[fs.innings == inn], inn, trees=trees[inn])
        joblib.dump({"lr": mdl["lr"], "med": mdl["med"], "lr_cols": mdl["lr_cols"], "cols": mdl["cols"]},
                    out / f"logistic_innings{inn}.joblib")
        mdl["gb"].save_model(str(out / f"lightgbm_innings{inn}.txt"))
        p_test[(test.innings == inn).to_numpy()] = predict_pair(mdl, test[test.innings == inn], inn)
    (out / "manifest.json").write_text(json.dumps({"version": 2, "trained_on": "2005-2024",
        "adds": EXTRA, "trees": trees, "note": "post-test fix; 2025-26 scores are not out-of-sample for model design"}, indent=2))
    t = pd.read_parquet(RES / "test_predictions.parquet")
    t["match_id"] = t["match_id"].astype(str)
    t = test[["match_id", "innings", "balls_bowled"]].assign(p_v2=p_test).merge(
        t, on=["match_id", "innings", "balls_bowled"])
    yt = t["label"].to_numpy()
    print(f"\nTest 2025-26 (POST-TEST, already seen): v1 {log_loss(yt, t['p_final'].to_numpy()):.4f}, "
          f"v2 {log_loss(yt, t['p_v2'].to_numpy()):.4f}")
    t.to_parquet(RES / "test_predictions_v2.parquet", index=False)
    va2 = va[["match_id", "innings", "balls_bowled"]].assign(p_v2=p_v2)
    va2.to_parquet(RES / "valid_predictions_v2model.parquet", index=False)
    x = test[(test.match_id == "1473468") & (test.innings == 2) & (test.balls_bowled.isin([60, 86]))]
    print("\nPBKS v KKR check (KKR chasing):")
    print(x[["balls_bowled", "wickets_in_hand", "runs_needed", "exp_runs_left", "need_ratio"]].round(2).to_string(index=False))
    print("  v2 KKR win chance:", np.round(p_test[x.index.map(lambda i: test.index.get_loc(i))], 3))


if __name__ == "__main__":
    main()

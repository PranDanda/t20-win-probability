"""Week 6: frozen model vs Betfair, pre-match, on linked 2025-26 test matches (pre-registered)."""
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import brier, log_loss
from calibration import ece

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
SIX_HOURS_MS = 6 * 3600 * 1000
EDGE, COMMISSION = 0.05, 0.05
rng = np.random.default_rng(42)


def boot(values_fn, n, reps=2000):
    out = []
    for _ in range(reps):
        out.append(values_fn(rng.integers(0, n, n)))
    return np.percentile(out, [2.5, 97.5])


def main():
    L = pd.read_parquet(PROC / "bf_links.parquet")
    L = L[L["matched"]].copy()
    pr = pd.read_parquet(PROC / "bf_prices.parquet")
    pr = pr[pr["market_id"].isin(L["market_id"])]
    mk = pd.read_parquet(PROC / "bf_markets.parquet").set_index("market_id")
    L["start_pt"] = L["inplay_pt"].fillna(
        L["market_id"].map(mk["market_time"]).astype("int64") // 10**6)

    # Last traded price for each team in the 6 hours before the first ball
    pre = pr.merge(L[["market_id", "start_pt"]], on="market_id")
    pre = pre[(pre["pt"] < pre["start_pt"]) & (pre["pt"] >= pre["start_pt"] - SIX_HOURS_MS)]
    last = pre.sort_values("pt").groupby(["market_id", "runner_id"]).last()["ltp"]
    L["odds_team1"] = [last.get((m, r), np.nan) for m, r in zip(L["market_id"], L["runner_team1"])]
    L["odds_team2"] = [last.get((m, r), np.nan) for m, r in zip(L["market_id"], L["runner_team2"])]
    n_linked = len(L)
    L = L.dropna(subset=["odds_team1", "odds_team2"])
    inv1, inv2 = 1 / L["odds_team1"], 1 / L["odds_team2"]
    L["mkt_team1"] = inv1 / (inv1 + inv2)
    L["overround"] = inv1 + inv2

    m = pd.read_parquet(PROC / "matches.parquet")[["match_id", "team_batting_first"]]
    m["match_id"] = m["match_id"].astype(str)
    L = L.merge(m, on="match_id")
    t = pd.read_parquet(RES / "test_predictions.parquet")
    t["match_id"] = t["match_id"].astype(str)
    first = t[(t["innings"] == 1) & (t["balls_bowled"] == 0)][["match_id", "p_final", "label"]]
    L = L.merge(first, on="match_id")
    bat1_is_team1 = L["team_batting_first"] == L["team1"]
    L["p_market"] = np.where(bat1_is_team1, L["mkt_team1"], 1 - L["mkt_team1"])
    L["odds_bat1"] = np.where(bat1_is_team1, L["odds_team1"], L["odds_team2"])
    L["odds_bat2"] = np.where(bat1_is_team1, L["odds_team2"], L["odds_team1"])
    L["p_model"] = L["p_final"]
    L["p_blend"] = (L["p_model"] + L["p_market"]) / 2
    L["franchise"] = L["competition"] != "t20s"

    print(f"Linked {n_linked:,}; with pre-match prices for both teams: {len(L):,}")
    print(f"Median market overround (sum of 1/odds): {L['overround'].median():.3f}\n")
    y = L["label"].to_numpy()
    rows = []
    for name, sub in [("all", L), ("franchise leagues", L[L.franchise]), ("T20 internationals", L[~L.franchise])]:
        ys = sub["label"].to_numpy()
        r = {"slice": name, "matches": len(sub)}
        for k in ["p_model", "p_market", "p_blend"]:
            r[f"logloss_{k[2:]}"] = log_loss(ys, sub[k].to_numpy())
        r["brier_model"] = brier(ys, sub["p_model"].to_numpy())
        r["brier_market"] = brier(ys, sub["p_market"].to_numpy())
        r["fav_won_model"] = ((sub["p_model"] >= .5) == (ys == 1)).mean()
        r["fav_won_market"] = ((sub["p_market"] >= .5) == (ys == 1)).mean()
        rows.append(r)
    res = pd.DataFrame(rows).set_index("slice")
    res.to_csv(RES / "market_prematch.csv")
    print(res.round(4).to_string())

    pm, pk = L["p_model"].to_numpy(), L["p_market"].to_numpy()
    eps = 1e-6
    def ll_vec(p): p = np.clip(p, eps, 1 - eps); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll_vec(pm) - ll_vec(pk)
    lo, hi = boot(lambda i: d[i].mean(), len(d))
    print(f"\nModel minus market log loss: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (negative = model better)")
    db = ll_vec(L["p_blend"].to_numpy()) - ll_vec(pk)
    lo, hi = boot(lambda i: db[i].mean(), len(db))
    print(f"Blend minus market log loss: {db.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"Calibration error: model {ece(y, pm):.3f}, market {ece(y, pk):.3f}")
    print(f"Correlation of model and market probabilities: {np.corrcoef(pm, pk)[0,1]:.3f}")

    # Paper betting: back the side the model rates 5+ points above the market
    bets = []
    for r in L.itertuples():
        for side, p_mod, p_mkt, odds, won in [("bat1", r.p_model, r.p_market, r.odds_bat1, r.label == 1),
                                              ("bat2", 1 - r.p_model, 1 - r.p_market, r.odds_bat2, r.label == 0)]:
            if p_mod - p_mkt >= EDGE:
                profit = (odds - 1) * (1 - COMMISSION) if won else -1.0
                bets.append({"match_id": r.match_id, "edge": p_mod - p_mkt, "odds": odds,
                             "won": won, "profit": profit, "franchise": r.franchise})
    B = pd.DataFrame(bets)
    B.to_csv(RES / "market_paper_bets.csv", index=False)
    pr_ = B["profit"].to_numpy()
    lo, hi = boot(lambda i: pr_[i].mean(), len(pr_))
    print(f"\nPaper bets (edge >= {EDGE:.0%}): {len(B)} bets, won {B['won'].mean():.1%}, "
          f"avg odds {B['odds'].mean():.2f}")
    print(f"  Profit per unit staked: {pr_.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    for fr, g in B.groupby("franchise"):
        print(f"  {'franchise' if fr else 'T20I':9s}: {len(g)} bets, profit/unit {g['profit'].mean():+.3f}")
    L.to_parquet(RES / "market_prematch_rows.parquet", index=False)


if __name__ == "__main__":
    main()

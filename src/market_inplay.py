"""Week 6, exploratory: model vs market DURING matches, with approximate ball timings.

Cricsheet has ball order but no timestamps. Approximation per match: spread the legal balls
evenly over the in-play window (first in-play price to the last price before the market is
decided), leaving a 15-minute gap at the innings break. Alignment check BEFORE comparing:
for each match, find the time shift (-12..+12 min) that best correlates model and market; if
alignment were good the best shift would sit near 0 with high correlation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import log_loss

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data" / "processed", ROOT / "results"
MIN = 60_000
BREAK_MS = 15 * MIN


def market_series(prices, r):
    p = prices[prices["market_id"] == r.market_id].sort_values("pt")
    p = p[p["pt"] >= r.start_pt]
    w = p.pivot_table(index="pt", columns="runner_id", values="ltp", aggfunc="last").ffill()
    if r.runner_team1 not in w or r.runner_team2 not in w:
        return None
    w = w[[r.runner_team1, r.runner_team2]].dropna()
    inv = 1 / w
    s = inv.iloc[:, 0] / inv.sum(axis=1)               # P(team1)
    decided = s[(s < 0.02) | (s > 0.98)]
    end = decided.index[0] if len(decided) else s.index[-1]
    return s[s.index <= end]


def main():
    L = pd.read_parquet(RES / "market_prematch_rows.parquet")
    prices = pd.read_parquet(PROC / "bf_prices.parquet")
    prices = prices[prices["market_id"].isin(L["market_id"])]
    t = pd.read_parquet(RES / "test_predictions.parquet")
    t["match_id"] = t["match_id"].astype(str)
    t = t[t["match_id"].isin(L["match_id"])].sort_values(["match_id", "innings", "balls_bowled"])

    rows, shifts = [], []
    for r in L.itertuples():
        s = market_series(prices, r)
        if s is None or len(s) < 30:
            continue
        balls = t[t["match_id"] == r.match_id]
        n1, n2 = (balls["innings"] == 1).sum(), (balls["innings"] == 2).sum()
        if n1 == 0 or n2 == 0:
            continue
        span = s.index[-1] - s.index[0] - BREAK_MS
        if span <= 0:
            continue
        per_ball = span / (n1 + n2)
        k = np.arange(len(balls))
        times = s.index[0] + k * per_ball + np.where(balls["innings"].to_numpy() == 2, BREAK_MS, 0)
        # model P(team1): label/p are for the batting side
        bat_is_team1 = np.where(balls["innings"].to_numpy() == 1,
                                r.team_batting_first == r.team1, r.team_batting_first != r.team1)
        p_mod_t1 = np.where(bat_is_team1, balls["p_final"], 1 - balls["p_final"])
        best = (-1, None)
        for sh in range(-12, 13):
            idx = np.searchsorted(s.index.to_numpy(), times + sh * MIN, side="right") - 1
            ok = idx >= 0
            if ok.sum() < 30:
                continue
            c = np.corrcoef(p_mod_t1[ok], s.to_numpy()[idx[ok]])[0, 1]
            if c > best[0]:
                best = (c, sh)
        shifts.append({"match_id": r.match_id, "best_corr": best[0], "best_shift_min": best[1]})
        idx = np.searchsorted(s.index.to_numpy(), times, side="right") - 1
        ok = idx >= 0
        p_mkt_t1 = s.to_numpy()[idx[ok]]
        lab_t1 = int(r.cs_winner == r.team1)
        rows.append(pd.DataFrame({"match_id": r.match_id, "innings": balls["innings"].to_numpy()[ok],
                                  "balls_bowled": balls["balls_bowled"].to_numpy()[ok],
                                  "p_model": p_mod_t1[ok], "p_market": p_mkt_t1, "label": lab_t1}))
    S = pd.DataFrame(shifts)
    print(f"Matches with usable in-play prices: {len(S):,}")
    print(f"Alignment check: median best shift {S['best_shift_min'].median():+.0f} min; "
          f"share with |shift| <= 3 min: {(S['best_shift_min'].abs() <= 3).mean():.1%}; "
          f"median best correlation {S['best_corr'].median():.3f}")
    A = pd.concat(rows, ignore_index=True)
    A["phase"] = pd.cut(A["balls_bowled"], [-1, 35, 89, 200], labels=["powerplay", "middle", "death"])
    out = []
    for (inn, ph), g in A.groupby(["innings", "phase"], observed=True):
        y = g["label"].to_numpy()
        out.append({"innings": inn, "phase": ph, "balls": len(g),
                    "model": log_loss(y, g["p_model"].to_numpy()),
                    "market": log_loss(y, g["p_market"].to_numpy())})
    O = pd.DataFrame(out)
    y = A["label"].to_numpy()
    print(f"\nAll aligned balls ({len(A):,}): model {log_loss(y, A['p_model'].to_numpy()):.4f}, "
          f"market {log_loss(y, A['p_market'].to_numpy()):.4f}")
    print(O.round(4).to_string(index=False))
    S.to_csv(RES / "market_inplay_alignment.csv", index=False)
    O.to_csv(RES / "market_inplay_by_phase.csv", index=False)


if __name__ == "__main__":
    main()

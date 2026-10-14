"""Charts for the write-up, saved as PNGs in results/figures/.

1. calibration.png  : predicted vs actual win rate, lookup table vs logistic regression
2. wc_final_2024.png: India's win probability through the 2024 T20 World Cup final
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Palette (reference data-viz palette, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def calibration_chart(v):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls=(0, (4, 4)), zorder=1,
            label="Perfect calibration")
    bands = np.linspace(0, 1, 11)
    for col, colour, name in [("p_lookup", ORANGE, "Lookup table"),
                              ("p_lr_engineered", BLUE, "Logistic regression")]:
        b = pd.cut(v[col], bands, include_lowest=True)
        g = v.groupby(b, observed=True).agg(pred=(col, "mean"), act=("label", "mean"))
        ax.plot(g["pred"] * 100, g["act"] * 100, color=colour, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Predicted win probability (%)")
    ax.set_ylabel("Actual win rate (%)")
    ax.set_title("Calibration on 2024 validation matches", loc="left", fontsize=12,
                 color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)


def wc_final_chart(v, deliveries):
    mid = "1415755"  # India v South Africa, 29 June 2024, Bridgetown
    m = v[v["match_id"] == mid].copy()
    m["ball"] = np.where(m["innings"] == 1, 121 - m["balls_left"], 241 - m["balls_left"])
    # Predictions are for the batting side: India bat first, South Africa chase
    for col in ["p_lr_engineered", "p_lookup"]:
        m[col + "_ind"] = np.where(m["innings"] == 1, m[col], 1 - m[col]) * 100
    m = m.sort_values("ball")
    end = pd.DataFrame({"ball": [241], "p_lr_engineered_ind": [100], "p_lookup_ind": [100]})
    m = pd.concat([m, end], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.axhline(50, color=INK2, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.axvline(120.5, color=GRID, lw=1.5, zorder=1)
    ax.plot(m["ball"], m["p_lookup_ind"], color=ORANGE, lw=1.5, alpha=0.9,
            label="Lookup table", zorder=2)
    ax.plot(m["ball"], m["p_lr_engineered_ind"], color=BLUE, lw=2,
            label="Logistic regression", zorder=3)

    def note(ball, text, dy, dx=0):
        y = float(np.interp(ball, m["ball"], m["p_lr_engineered_ind"]))
        ax.annotate(text, (ball, y), xytext=(ball + dx, y + dy), ha="center", fontsize=8.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    note(211, "SA need 30 off 30,\n6 wickets left", 8, dx=-34)
    note(218, "Klaasen out", 28)
    note(236, "Miller caught\non the boundary", 26)
    note(13, "Rohit and Pant\nout in over 2", -24)

    ax.text(60, 103, "India bat: 176/7", ha="center", color=INK2, fontsize=9)
    ax.text(180, 103, "South Africa chase 177: 169/8", ha="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 242); ax.set_ylim(0, 108)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240])
    ax.set_xticklabels(["0", "5", "10", "15", "20 | 0", "5", "10", "15", "20"])
    ax.set_xlabel("Overs (first innings | second innings)")
    ax.set_ylabel("India win probability (%)")
    ax.set_title("2024 T20 World Cup final: India v South Africa", loc="left",
                 fontsize=12, color=INK, pad=18)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "wc_final_2024.png", dpi=150)
    plt.close(fig)


def wc_final_week4(v2):
    """Week 3 (scoreboard only) vs week 4 (scoreboard + context ensemble)."""
    mid = "1415755"
    m = v2[v2["match_id"] == mid].copy()
    m["ball"] = np.where(m["innings"] == 1, 121 - m["balls_left"], 241 - m["balls_left"])
    m["p_ensemble"] = (m["p_lr_context"] + m["p_lgb_context"]) / 2
    for col in ["p_lr_state", "p_ensemble"]:
        m[col + "_ind"] = np.where(m["innings"] == 1, m[col], 1 - m[col]) * 100
    m = m.sort_values("ball")
    end = pd.DataFrame({"ball": [241], "p_lr_state_ind": [100], "p_ensemble_ind": [100]})
    m = pd.concat([m, end], ignore_index=True)

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.axhline(50, color=INK2, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.axvline(120.5, color=GRID, lw=1.5, zorder=1)
    ax.plot(m["ball"], m["p_lr_state_ind"], color=ORANGE, lw=1.5, label="Week 3: scoreboard only")
    ax.plot(m["ball"], m["p_ensemble_ind"], color=BLUE, lw=2, label="Week 4: + team, venue, era")
    first = m.iloc[0]
    ax.annotate(f"Before a ball: {first['p_lr_state_ind']:.0f}% vs {first['p_ensemble_ind']:.0f}%\n"
                "(India the stronger side on Elo)", (1, first["p_ensemble_ind"]),
                xytext=(28, 86), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(60, 103, "India bat: 176/7", ha="center", color=INK2, fontsize=9)
    ax.text(180, 103, "South Africa chase 177: 169/8", ha="center", color=INK2, fontsize=9)
    ax.set_xlim(0, 242); ax.set_ylim(0, 108)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180, 210, 240])
    ax.set_xticklabels(["0", "5", "10", "15", "20 | 0", "5", "10", "15", "20"])
    ax.set_xlabel("Overs (first innings | second innings)")
    ax.set_ylabel("India win probability (%)")
    ax.set_title("2024 T20 World Cup final: adding context", loc="left", fontsize=12,
                 color=INK, pad=18)
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "wc_final_2024_week4.png", dpi=150)
    plt.close(fig)


def test_calibration_chart():
    t = pd.read_parquet(RES / "test_predictions.parquet")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls=(0, (4, 4)), zorder=1, label="Perfect calibration")
    bands = np.linspace(0, 1, 11)
    for col, colour, name in [("p_lr_state", ORANGE, "Scoreboard-only logistic"),
                              ("p_final", BLUE, "Final model")]:
        b = pd.cut(t[col], bands, include_lowest=True)
        g = t.groupby(b, observed=True).agg(pred=(col, "mean"), act=("label", "mean"))
        ax.plot(g["pred"] * 100, g["act"] * 100, color=colour, lw=2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Predicted win probability (%)"); ax.set_ylabel("Actual win rate (%)")
    ax.set_title("Calibration on the 2025-26 test set (1,539 matches)", loc="left", fontsize=12,
                 color=INK, pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "test_calibration.png", dpi=150); plt.close(fig)


def main():
    v = pd.read_parquet(RES / "valid_predictions.parquet")
    v["match_id"] = v["match_id"].astype(str)
    calibration_chart(v)
    wc_final_chart(v, None)
    v2p = RES / "valid_predictions_v2.parquet"
    if v2p.exists():
        v2 = pd.read_parquet(v2p)
        v2["match_id"] = v2["match_id"].astype(str)
        wc_final_week4(v2)
    if (RES / "test_predictions.parquet").exists():
        test_calibration_chart()
    print("Saved:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()

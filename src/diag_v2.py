"""Why does v2 still give KKR ~97% at 95/8 needing 17 off 34? Diagnosis only."""
import joblib, lightgbm as lgb, numpy as np, pandas as pd
from pathlib import Path
from logistic import engineer
from v2_resources import add_features
ROOT = Path(__file__).resolve().parents[1]
F = pd.read_parquet(ROOT/"data/processed/features_v4.parquet")
tab = pd.read_csv(ROOT/"models/final_v2/resource_table.csv", index_col=0)
F = add_features(F, tab)
lr = joblib.load(ROOT/"models/final_v2/logistic_innings2.joblib")
gb = lgb.Booster(model_file=str(ROOT/"models/final_v2/lightgbm_innings2.txt"))
def parts(d):
    X = pd.concat([engineer(d, 2), d[lr["lr_cols"]].astype(float).fillna(lr["med"])], axis=1)
    return lr["lr"].predict_proba(X)[:, 1], gb.predict(d[lr["cols"]])
row = F[(F.match_id == "1473468") & (F.innings == 2) & (F.balls_bowled == 86)]
a, b = parts(row)
print("logistic", a.round(3), "lightgbm", b.round(3))
print(row[lr["cols"]].T.to_string())
# Counterfactual: set context to neutral (median) values
neutral = row.copy()
for c in ["elo2_diff", "venue_par", "venue_chase_rate", "comp_level", "target_vs_par", "pace_vs_par", "striker_rpb", "non_striker_rpb"]:
    if c in neutral: neutral[c] = F.loc[F.innings == 2, c].median()
a, b = parts(neutral)
print("\nneutral context (state only): logistic", a.round(3), "lightgbm", b.round(3))
for c in ["target_vs_par", "striker_rpb", "non_striker_rpb", "elo2_diff"]:
    t = row.copy(); t[c] = F.loc[F.innings == 2, c].median(); a, b = parts(t)
    print(f"  neutralise {c:16s}: lr {a[0]:.3f} gb {b[0]:.3f}")
# Empirical: chases at <=2 wkts in hand, 12-22 needed, 25-45 balls left, all years
e = F[(F.innings == 2) & (F.wickets_in_hand == 2) & F.runs_needed.between(12, 22) & F.balls_left.between(25, 45)]
m = e.groupby("match_id").agg(label=("label", "first"), tvp=("target_vs_par", "first"))
print(f"\nEmpirical, 2 wkts, need 12-22 off 25-45: {len(m)} matches, chaser won {m.label.mean():.2f}")
print("  of which low targets (target_vs_par < -40):", (m.tvp < -40).sum(), "won", m[m.tvp < -40].label.mean().round(2))
e = F[(F.innings == 2) & (F.wickets_in_hand <= 3) & F.runs_needed.between(10, 25) & F.balls_left.between(20, 50)]
m = e.groupby("match_id").agg(label=("label", "first"), w=("wickets_in_hand", "min"))
print(f"Wider (<=3 wkts, need 10-25 off 20-50): {len(m)} matches, chaser won {m.label.mean():.2f}")

import numpy as np, pandas as pd, joblib, lightgbm as lgb
from pathlib import Path
from logistic import engineer
from v2_resources import add_features
ROOT = Path(__file__).resolve().parents[1]
F = pd.read_parquet(ROOT/"data/processed/features_v4.parquet")
F["match_id"] = F["match_id"].astype(str)
tab = pd.read_csv(ROOT/"models/final_v2/resource_table.csv", index_col=0)
F = add_features(F, tab)
lr = joblib.load(ROOT/"models/final_v2/logistic_innings2.joblib")
gb = lgb.Booster(model_file=str(ROOT/"models/final_v2/lightgbm_innings2.txt"))
X = lambda d: pd.concat([engineer(d, 2), d[lr["lr_cols"]].astype(float).fillna(lr["med"])], axis=1)
for w, lo, hi in [(2, 12, 22), (1, 8, 16), (3, 15, 30)]:
    e = F[(F.innings == 2) & (F.wickets_in_hand == w) & F.runs_needed.between(lo, hi) & F.balls_left.between(25, 45)].copy()
    e["p"] = (lr["lr"].predict_proba(X(e))[:, 1] + gb.predict(e[lr["cols"]])) / 2
    # first qualifying ball per match so each match counts once
    m = e.sort_values("balls_bowled").groupby("match_id").first()
    print(f"{w} wkts, need {lo}-{hi} off 25-45 balls: {len(m)} matches, chaser won {m.label.mean():.2f}, "
          f"v2 says {m.p.mean():.2f}; split counts {m.split.value_counts().to_dict()}")

# T20 win probability model

Ball-by-ball win probability for men's T20 cricket. **Live site: https://prandanda.github.io/t20-win-probability/**

- 8,694 matches since 2005 from Cricsheet; trained on 2005-2024, tuned on 2024.
- Scored once on 1,539 unseen matches from 2025-26: log loss 0.445 (scoreboard-only 0.486,
  coin flip 0.693), called the winner 77.5% of the time, calibration error 1.5%.
- Benchmarked against the Betfair pre-match market, which is much better (0.565 against 0.650).
  This is a tool for explaining matches, not for betting.
- The site shows model v2, a labelled post-test fix for late-chase overconfidence.
  The headline numbers above are the frozen v1.

The reasoning behind every step, including the experiments that failed, is in
[LEARNING_NOTES.md](LEARNING_NOTES.md). The plan is in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

## Data

Ball-by-ball data from [Cricsheet](https://cricsheet.org), used under the
[Open Data Commons Attribution Licence](https://opendatacommons.org/licenses/by/1-0/).
Cricsheet must be credited in any public use.

**`data/` is not in this repository.** Download the Cricsheet T20 zips into `data/raw/` to rebuild.
Betfair data and anything derived from its prices are kept out of the repo (personal-use licence);
only summary scores are published.

Locally, `data/betfair/data.tar` holds Betfair historic data (BASIC tier, downloaded by hand from historicdata.betfair.com; personal use under Betfair's terms). `data/raw/` holds the Cricsheet zips (downloaded by hand: cricsheet.org is not reachable
from the build environment). `data/json/` is the unzipped copy.

Rebuild the processed tables:

```
pip install -r requirements.txt
python3 src/build_deliveries.py
```

## Processed tables (`data/processed/`)

| File | One row per | Notes |
| --- | --- | --- |
| `matches.parquet` | match | teams, venue, date, toss, winner, method, `tied`, `usable` flag |
| `deliveries.parquet` | delivery (regulation innings, no super overs) | runs, extras, wickets, cumulative score/wickets/legal balls *after* the ball, chase target, label `batting_team_won` |
| `summary.csv` | competition | match and delivery counts |

`usable` = has a winner, two full innings, 20 x 6-ball overs, no D/L or other method, no
reduced-overs target. 8,033 of 8,694 matches; 1.89M deliveries.

Known quirks:
- Older files omit the chase target; it is filled as first-innings total + 1 (`target_filled`).
- 64 usable matches are ties decided by a super over (`tied` = True); decide how to label them in modelling.
- 20 innings record more than 120 legal balls (scoring errors or 7-ball overs); clip balls remaining at 0.
- `batting_team_won` = -1 where the match had no winner.

## Pipeline

```
python3 src/build_deliveries.py   # zips -> deliveries.parquet, matches.parquet
python3 src/build_features.py     # -> features.parquet (one row per legal ball, with split)
python3 src/baseline.py           # -> results/baseline_*.csv
cd src && python3 logistic.py     # -> models/*.joblib, results/logistic_*.csv, valid_predictions
python3 plots.py                  # -> results/figures/*.png
python3 checks.py                 # sanity checks on the chase model
python3 context_features.py       # -> features_v2.parquet (Elo, venue, era, batter quality)
python3 batch_features.py         # -> features_v4.parquet (adds Elo v2 + batch-2 experiment features)
python3 boosting.py               # -> models with Elo v2, metrics, importance
python3 player_features.py        # -> features_v3.parquet (bowling/batting resources; experiment)
python3 ablation.py               # -> results/ablation_player_features.csv
python3 ablation2.py              # -> results/ablation_batch2.csv
python3 error_analysis.py         # week 5: where the model goes wrong (validation)
python3 calibration.py            # week 5: calibration layer test (none adopted)
python3 final_model.py            # week 5: refit frozen model on 2005-2024 -> models/final/
python3 test_eval.py              # week 5: score the 2025-26 test set (run once)
python3 parse_betfair.py          # week 6: Betfair BASIC files -> bf_markets, bf_prices
python3 match_betfair.py          # week 6: link test matches to Betfair markets
python3 market_eval.py            # week 6: pre-match model vs market, blend, paper betting
python3 market_inplay.py          # week 6: exploratory in-play (alignment check fails)
python3 v2_resources.py           # week 7: v2, expected-runs-remaining features -> models/final_v2/
python3 v3_reach.py               # week 7: v3, chance-of-reaching-target feature (rejected)
python3 export_site.py            # week 7: site/data/*.json for the website
```

The site is plain HTML, CSS and JavaScript in `site/`, deployed by `.github/workflows/pages.yml`.
To view it locally: `cd site && python3 -m http.server`.
See `LEARNING_NOTES.md` for the reasoning behind each step and `docs/BUILD_PLAN.md` for the plan.

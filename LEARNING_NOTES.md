# Learning notes: building an ML model end to end

A running record of each step in this project: what we did, why, the alternatives we
rejected, and the general lesson that carries over to any machine learning project.

---

## Week 1: Getting and cleaning the data

### What we did
1. Downloaded Cricsheet ball-by-ball JSON for 8 men's T20 competitions (8,694 matches).
2. Parsed every delivery into one table (`deliveries.parquet`, 1.99M rows) and every match
   into another (`matches.parquet`).
3. Stored the score, wickets and legal balls **after** each delivery, plus the label:
   did the batting team go on to win?
4. Flagged which matches are fit for modelling (`usable`): 8,033 of 8,694.
5. Checked the data against facts that must be true, found problems, fixed them.

### Why: key decisions
- **Two tables, not one.** Match facts (venue, toss, winner) live once in `matches`;
  ball facts live in `deliveries`. This is *normalisation*: no repeated data, so nothing
  can get out of sync. We join them when needed.
- **Parquet, not CSV.** Columnar, compressed (14.8MB vs 700MB of JSON), keeps data types
  (a match ID stays text, a date stays a date), and lets you load only the columns you need.
- **Exclude rather than patch messy matches.** Rain-affected (D/L) games, shortened chases,
  no-results and 20-over-but-not-really games follow different rules. Including them would
  teach the model wrong lessons ("needing 60 off 30 is easy" when the target was revised).
  We have plenty of data, so dropping 8% costs little.
- **Keep the raw data untouched.** The zips stay in `data/raw/`; everything else is
  rebuilt by a script. Any mistake can be fixed by editing the script and rerunning,
  never by hand-editing data. This is called *reproducibility*.

### Problems found and how
We tested the data against cricket facts that must hold:
- "Target = first-innings score + 1" failed for 9% of matches. Cause: older files simply
  don't record the target. Fix: fill it from the first-innings total (flagged `target_filled`).
- "A chasing team that won must have reached the target" failed for 31 matches. Cause:
  ties decided by super overs. Fix: flagged `tied`; we will exclude them from modelling.
- 20 innings show more than 120 legal balls: scorer errors or umpires miscounting an over.
  Too rare to matter; we will cap "balls remaining" at 0.
- Engineering problem: the first run crashed (out of memory) building 2M rows at once.
  Fix: process one competition at a time, use compact data types (int8, categories).

### Alternatives considered
- *A database (DuckDB/Postgres) instead of Parquet files*: overkill for one person and
  a few tables; Parquet files can be queried by DuckDB directly later if needed.
- *Fixing D/L matches by rescaling targets*: possible, but complex and error-prone for
  a small gain in data.

### General lessons
- **Most of an ML project is data work.** Expect 60–80% of the effort before any model.
- **Validate data against known truths ("invariants").** It is the cheapest way to catch
  errors that would otherwise surface as a model that looks good but is quietly wrong.
- **Never trust a field is always present.** Data formats change over the years; older
  records often lack fields that newer ones have.
- **Write the pipeline as a script, not manual steps**, so it can be rerun when new
  matches arrive.

---

## Week 2: Features, the split, and a baseline

### What we did
1. `src/build_features.py`: one row per **legal ball** (1.8M rows, 7,969 matches) describing
   the match state *before* that ball: balls left, wickets in hand, runs, current run rate,
   phase, and for chases runs needed, required rate and the gap between the two rates.
2. Split by date: train = up to 2023 (5,549 matches), validation = 2024 (881),
   test = 2025 onwards (1,539, locked away until week 5).
3. `src/baseline.py`: three benchmarks scored on validation.

### Results (validation, 2024)
| Model | Log loss | Brier | Accuracy |
| --- | --- | --- | --- |
| Coin flip (always 50%) | 0.693 | 0.250 | 48% |
| Base rate (average win rate) | 0.694 | 0.250 | 52% |
| Lookup table (similar past situations) | **0.536** | **0.182** | **72%** |

By innings: lookup log loss 0.628 in the first innings, 0.432 in the chase.
By phase: 0.601 powerplay, 0.516 middle, 0.481 death.

### Why: key decisions
- **Features = what an expert looks at.** Start with the obvious drivers; add cleverness only
  once a simple model shows where it falls short.
- **State before the ball, not after.** Using the state after a ball would leak that ball's
  outcome into its own prediction. *Leakage* = the model seeing information it would not
  have when making a real prediction. It is the most common way ML projects fool themselves.
- **Only legal balls are rows.** Wides and no-balls add runs but don't use a ball, so they
  fold into the next row's state. Otherwise one situation would appear twice.
- **Split by time and by match.** Real predictions are always of future matches; a random
  split lets the model learn from the future and makes scores look better than they are.
  All balls of a match share one result, so a match must sit entirely in one split.
- **Three sets, not two.** Every time you look at a score and tweak something, you tune to
  that data. Do all tweaking on validation; touch test once, at the end, for an honest number.
- **Benchmarks first.** A score means nothing without a reference. The coin flip and base
  rate show the "knows nothing" level (log loss 0.693 = ln 2); the lookup table is the
  "simple but sensible" bar every real model must clear.
- **Smoothing in the lookup table.** Rare situations (e.g. 2 overs left, 3 wickets, 40 needed)
  have few past examples, so their raw win rate is noisy. Each detailed cell is blended with
  a coarser cell (same overs and wickets, any score) in proportion to how much data it has
  (`K = 30`). This is *shrinkage*: trust a small sample less, pull it toward a broader average.
  Same idea as credibility weighting in insurance or Bayesian priors.

### Reading the metrics
- **Log loss** punishes confident wrong answers heavily (saying 95% and losing costs far more
  than saying 60% and losing). It is the main metric because we care about the *probability*.
- **Brier score** = average squared gap between probability and outcome (0 or 1). Gentler
  on confident mistakes; easier to explain.
- **Accuracy** only checks which side of 50% you were on, so it ignores how confident you
  were. A useful side note, never the target.
- **Calibration table**: group predictions into 10% bands and compare with how often those
  teams actually won. The lookup table is well calibrated (e.g. predicted 25% → won 26%),
  slightly underrating the batting team around 50–60%.

### Something odd we checked
The lookup table predicted the team batting first wins 46% of the time in 2024; they actually
won 50%. Checking year by year, the batting-first win rate swings between 42% and 52% with no
trend, and average first-innings totals stay around 154–167. So this is mostly year-to-year
noise, not a change in the game. Lesson: before "fixing" a gap, check whether it is a real
pattern or random variation.

### What the baseline tells us about the next model
- The chase (log loss 0.43) is far easier than the first innings (0.63): with a target, the
  state says a lot; early in the first innings, the outcome depends on things the scoreboard
  can't see yet (team strength, pitch). That is where week 4's context features should help.
- The lookup table chops states into boxes, so its predictions jump at box edges and can't
  share information between neighbouring boxes. Logistic regression (week 3) gives smooth
  predictions, and should beat it especially in sparse situations.

### General lessons
- **Always build a dumb baseline first.** It sets the bar and catches bugs (a "clever" model
  that can't beat a lookup table is broken).
- **Pick the metric before the model**, and pick one that matches how the output is used.
- **Report scores by slice** (innings, phase), not just overall; averages hide where a model is weak.

### Appendix: how `baseline.py` works, line by line

**"Training" the lookup table is counting, not learning.** No weights are adjusted; it
records how often the batting team won from each kind of situation.

1. **Put each ball in a box** (`add_bins`). `//` is whole-number division:
   - balls left -> overs left: `(balls_left - 1) // 6` (24 balls -> box 3)
   - runs needed (chase) or runs scored (1st innings) -> 5-run buckets: `need // 5`
   - `clip` caps extreme values so freak scores don't create tiny boxes.
   Why bucket: an exact situation may have occurred only a few times; grouping similar
   situations gives each box enough examples.
2. **Count wins per box** (`fit_lookup`). `groupby(...)["label"].agg(["sum","count"])`
   is a pivot table: `sum` = balls in the box whose batting team won, `count` = balls in
   the box. Three levels: innings only -> overs+wickets -> overs+wickets+runs (9,039 boxes).
3. **Smooth small boxes** (shrinkage): `p = (wins + K * broader_rate) / (balls + K)`,
   K = 30 "imaginary balls" at the broader box's rate.
   - 3,000 balls, 1,800 wins -> 59.9% (own data dominates)
   - 4 balls, 4 wins -> 56%, not 100% (4 balls prove almost nothing)
4. **Predict by lookup** (`predict_lookup`). `join` is a VLOOKUP into the tables built on
   2005-23; `fillna` falls back to the broader box when a detailed box never appeared.
5. **Score** each 2024 ball: log loss = average of -log(probability given to what actually
   happened); `clip` avoids log(0). Brier = mean((p - y)^2). Accuracy = right side of 50%?
   Calibration = `pd.cut` into 10% bands, then predicted vs actual per band.

Note: the week 2 calibration chart was drawn by a chat chart tool from the numbers in
`results/baseline_calibration.csv`, not by project code. From week 3, charts come from
`src/plots.py` so they are reproducible.

---

## Week 3: Logistic regression

### What we did
1. `src/logistic.py`: trained logistic regression separately for each innings, in two
   variants: **raw** (the plain columns) and **engineered** (hand-built extra features).
2. `src/plots.py`: calibration chart and a ball-by-ball chart of the 2024 T20 World Cup
   final (`results/figures/`).
3. `src/checks.py`: sanity checks that the model moves the right way when one thing changes.

### Results (validation, 2024, log loss; lower is better)
| Scope | Lookup table | Logistic (raw) | Logistic (engineered) |
| --- | --- | --- | --- |
| All balls | 0.536 | 0.510 | **0.506** |
| 1st innings | 0.628 | 0.615 | **0.613** |
| 2nd innings | 0.432 | 0.390 | **0.384** |
| Powerplay | 0.601 | 0.580 | **0.578** |
| Middle overs | 0.516 | 0.484 | **0.481** |
| Death overs | 0.481 | 0.457 | **0.449** |

Brier 0.182 -> 0.173; accuracy 71.7% -> 72.9%.

### How logistic regression learns
- **Model:** score = w0 + w1·feature1 + w2·feature2 + ...; probability = 1 / (1 + e^-score).
  The sigmoid squashes any score into 0-100%: score 0 = 50%, +2 = 88%, -2 = 12%.
- **Training:** start all weights at 0, compute log loss on 1.26M training balls, compute the
  *gradient* (which way each weight should move to lower the loss), step, repeat until the
  loss stops improving. scikit-learn's L-BFGS is a smarter stepping rule; it converged in
  13-73 steps. Same principle as training a neural network, with far fewer weights.
- **Scaling** (`StandardScaler`): each feature is converted to mean 0, standard deviation 1
  first, so the optimiser treats them evenly and weights are comparable.
- **Regularisation** (`C=1.0`): a mild penalty on large weights, which discourages the
  model from leaning too hard on any one feature.

### Why: key decisions
- **One model per innings.** Chasing a known target and setting one are different problems
  (different features even exist), so separate models are simpler than one model that has to
  learn the difference.
- **Feature engineering.** A linear model can only learn straight-line effects on the score;
  cricket is full of curves and interactions. So we built features that encode them:
  required rate squared (each extra run an over gets harder), required rate × wickets in hand
  (a steep rate is gettable with wickets left), wickets × share of innings left (wickets
  matter more early), log of runs needed, a projected first-innings total that blends the
  current rate with a typical 8 an over early on.
- **Kept the raw variant as a control.** Comparing raw vs engineered shows what the
  engineering is worth (about 0.004 log loss, most of it in the death overs).

### What we learned
- **Smooth beats boxes.** Even the raw logistic model beats the lookup table everywhere.
  The lookup table treats each box separately and wastes data; logistic regression shares
  information across all situations. Example from the final: needing 9 off the last ball,
  the lookup table said 32% (its tiny box was smoothed toward a broader average), the
  logistic model said 0.4%.
- **Calibration vs sharpness.** The lookup table is actually slightly *better calibrated*
  (its points sit closer to the diagonal), yet logistic regression has the better log loss.
  Log loss rewards both: being calibrated *and* being decisive (pushing probabilities away
  from 50% when the situation warrants). Logistic regression is sharper. Its calibration
  wobble (e.g. predicted 15% -> actual 20%) is what week 5's calibration layer will fix.
- **Don't over-read individual weights.** Several features overlap heavily (required rate,
  its square, its interactions), so weights trade off against each other and some look odd
  in isolation (`proj_x_wk` is negative in innings 1). This is *multicollinearity*. The
  combined prediction is fine; the individual coefficients aren't a reliable story.

### Sanity checks (`src/checks.py`)
| Situation (chasing) | Model says |
| --- | --- |
| Need 30 off 30: 10 / 6 / 2 / 1 wickets in hand | 98% / 93% / 36% / 12% |
| 6 wickets, 30 balls: need 20 / 40 / 60 / 80 | 99% / 76% / 23% / 4% |
| Need 9 with 6 wickets: 1 / 3 / 6 / 12 balls left | 0.3% / 24% / 89% / 99% |
| Last ball: need 1 / 2 / 4 / 6 | 94% / 64% / 8% / 1% |

- Over a grid of 238,800 chase states, needing more runs **never** raises the chaser's
  chances.
- Having one more wicket in hand lowers the chances in 7% of pairs, but 96% of those are
  where the result is already below 2% or above 98%, and the typical size is under 0.01
  points (max 3 points, on the last ball). Cause: the wickets-squared feature makes the
  curve flatten and dip slightly at 9-10 wickets. Week 4's *monotonic constraints* rule
  this out by construction.
- Suspicious: "need 2 off the last ball" at 64% looks high (a tie or a dot ball are both
  losses here). Very few training examples sit at the last ball, so the model extrapolates.
  Worth revisiting with the stronger model.

### The 2024 T20 World Cup final (`results/figures/wc_final_2024.png`)
The model tells the story correctly: India below 50% after losing Rohit and Pant early;
favourites at the innings break with 176; down to 5% when South Africa needed 30 off 30
with 6 wickets; back to 100% after Klaasen, Jansen and Miller fell. The lookup table's
line is jumpier: predictions leap at box edges.

### General lessons
- **Simple, well-understood models first.** Logistic regression is fast (15 seconds),
  explainable, and often close to more complex models. Anything fancier must earn its place.
- **Model choice and feature engineering trade off.** Simple models need clever features;
  flexible models (trees, neural nets) find curves themselves but need more data and care.
- **Probe the model with "what if" checks**, not just scores. A good average can hide
  absurd behaviour in specific situations.
- **Vectorise.** Scoring 54,000 states one call at a time timed out; one batched call took
  seconds.

---

## Week 4: Context features and gradient-boosted trees

### What we did
1. `src/context_features.py`: added information the scoreboard can't show, each computed
   **only from matches on earlier dates**:
   - **Elo team ratings** (start 1500; winner takes points from loser, more for an upset; K=24).
   - **Venue par**: average first-innings total at the ground, shrunk toward the competition level.
   - **Venue chase rate**: share of matches won chasing there, shrunk toward 50%.
   - **Scoring era** (`comp_level`): average first-innings total of the competition's last 60 matches.
   - **Batter quality**: the two current batters' career runs per ball, shrunk toward 1.2.
   - Derived: `pace_vs_par` (1st innings), `target_vs_par` (chase).
2. `src/boosting.py`: a 2x2 comparison per innings, {logistic, LightGBM} x {scoreboard only,
   scoreboard + context}, plus an average of the two context models (an ensemble).

### Results (validation, 2024, log loss)
| Scope | Logistic, scoreboard | Logistic + context | LightGBM, scoreboard | LightGBM + context | Average of both + context |
| --- | --- | --- | --- | --- | --- |
| All balls | 0.506 | 0.477 | 0.501 | 0.477 | **0.474** |
| 1st innings | 0.613 | 0.579 | 0.613 | 0.578 | **0.576** |
| 2nd innings | 0.384 | 0.361 | 0.375 | 0.362 | **0.359** |
| Before the first ball | 0.693 | 0.666 | 0.696 | 0.657 | - |

Journey so far (all balls): coin flip 0.693 -> lookup 0.536 -> logistic 0.506 -> + context
0.477 -> ensemble 0.474.

### The headline lesson: features beat models
- Adding context improved both models by ~0.03 log loss.
- Switching logistic -> LightGBM with the *same* features changed almost nothing (0.4769 vs 0.4768).
- So the gain came from **giving the model new information**, not from a cleverer algorithm.
  This is the most common pattern in applied ML, and the most commonly misattributed one.
- LightGBM's real advantage here: it matched hand-engineered logistic regression *without* any
  hand-built features (plain columns only), and it respects monotonic constraints.

### How the context features avoid leakage
- Matches are processed in date order. For each date: first compute features for every match
  that day from what was known *before* that date, then update ratings/averages with the day's
  results. Same-day double-headers can't see each other.
- Batter career stats are cumulative sums *minus the current day*.
- Spot check: recomputed the 2024 final's venue par by hand from the 58 earlier matches at
  Bridgetown: 156.63 both ways, no same-day or later match included.
- Bug found and fixed: a **cold start**. The second-ever T20I (2005) got a "recent scoring level"
  of 214 because only one earlier match existed. Now a default of 160 applies until 10 matches exist.
- Missing values (a few non-strikers with no career data): LightGBM handles them natively;
  for logistic regression they are filled with the **training-set** median (never validation's,
  which would leak).

### How gradient-boosted trees learn
- A **decision tree** is a flowchart of yes/no splits ("balls left < 30? required rate > 12?").
  Training picks the split that best separates wins from losses.
- **Boosting** builds trees one after another; each new tree predicts the *errors* the
  ensemble still makes and nudges predictions to fix them, scaled by a learning rate (0.05).
- **Early stopping**: keep adding trees only while the validation score improves. The models
  stopped at 62-239 trees; beyond that they would start memorising training quirks (overfitting).
- **Monotonic constraints**: we declared that more wickets in hand, more balls left, higher Elo
  and better batters can only help the batting side, and more runs needed can only hurt the
  chaser. Result: 0 cases (of 214,920) where an extra wicket lowers the chaser's chance, vs
  15,777 for week 3's logistic model. "Need 2 off the last ball" fell from 64% to a more
  plausible 51%.
- Settings worth knowing: `num_leaves` (tree size), `min_child_samples=500` (each leaf needs
  500 balls, which stops trees fitting single matches), `bagging_fraction` (each tree sees a
  random 80% of rows), `lambda_l2` (a penalty on extreme leaf values).

### What the features are worth (share of LightGBM's total split gain)
| Feature | 1st innings | 2nd innings |
| --- | --- | --- |
| Current run rate | 37% | 4% |
| Required run rate | - | 66% |
| Pace vs venue par | 16% | - |
| Elo difference | 14% | 2% |
| Wickets in hand | 8% | 6% |
| Venue chase rate / venue par / era | 8% / 7% / 7% | 3% / 3% / 2% |
| Batter quality (both) | 1% | 2% |

- Context matters most **before and early in the first innings**; once a target exists, the
  required rate dominates.
- Batter quality barely helps. Career runs per ball is a noisy signal and two batters are a
  small part of the story; better player features (batting order, bowlers' overs left) would
  need the full team sheets.
- Importance shows what the model *used*, not what *causes* winning; correlated features share
  credit unpredictably (balls left shows 0% because runs and run rate already encode it).

### Ensembling
Averaging logistic + LightGBM (both with context) beat either alone (0.4743). The two models
make different mistakes, so averaging cancels some. Cheap and almost always worth trying.

### What the 2024 final shows (`results/figures/wc_final_2024_week4.png`)
Before a ball, week 3 said 50/50 (it only knows the scoreboard); week 4 says India 66% (higher
Elo). Through India's early wickets week 4 stays above 60% where week 3 fell to 36%. In the
chase the two lines nearly coincide: once there is a target, the scoreboard dominates.

### General lessons
- **New information beats new algorithms.** Ask "what does an expert know that my model
  doesn't?" before reaching for a fancier model.
- **Compare fairly**: change one thing at a time (the 2x2) so you know where gains come from.
- **Time-aware features are where leakage hides.** Compute them in date order and spot-check by hand.
- **Encode domain knowledge as constraints** where you're sure of the direction; it prevents
  nonsense in rare situations the data barely covers.

---

## Interlude: what is a model physically, and how does one ball flow through the pipeline?

### Three layers
1. **Algorithm code** lives in libraries (scikit-learn, LightGBM), mostly C/C++. We never write it.
2. **Our recipe** is `src/`: which data, which features, the split, settings, evaluation.
3. **Trained models** in `models/` are files of learned numbers:
   - logistic `.joblib`: 61 numbers (feature means, standard deviations, weights, intercept).
   - LightGBM `.txt`: readable text of 120 trees; each split has a feature and threshold, each
     leaf a small nudge. Tree 0 of the chase model first splits on required rate at 8.49.
   Models are disposable outputs: delete them, rerun the recipe, get them back.

### Vocabulary
Features = variables = predictors = inputs = "X" (the columns the model reads).
Label = target = outcome = "y" (what it learns to predict: did the batting side win?).
One row = one example = one observation (here: one legal ball).

### One ball, end to end: South Africa need 30 off 30, 2024 final (15.1 overs)
| Stage | Script | What it produces for this ball |
| --- | --- | --- |
| 0. Raw data | Cricsheet JSON | `{"batter": "DA Miller", "bowler": "JJ Bumrah", "runs": {"total": 1}}` |
| 1. Parse | `build_deliveries.py` | a deliveries row: running totals after the ball (148/4, 91 legal balls), target 177, label 0 (SA lost) |
| 2. State features | `build_features.py` | state *before* the ball: 147/4, 30 balls left, 6 wickets, need 30, required rate 6.0, current rate 9.8, death overs, split = valid |
| 3. Context features | `context_features.py` | Elo difference -123 (India stronger), venue par 156.6, venue chase rate 39%, T20I scoring level 143.6, target 19 above par, batters' career runs/ball 1.39 and 1.59 |
| 4a. Logistic | `logistic.py` / `boosting.py` | 13 engineered columns + 7 context, scaled, x weights, summed: score 3.10 -> 95.7% SA |
| 4b. LightGBM | `boosting.py` | the same ball sent down 120 trees, 120 nudges summed: score 2.70 -> 93.7% SA |
| 5. Evaluate | metrics | the label is 0: SA lost from 94-96%. That ball adds a large log loss (about 2.9); across thousands of balls, rare upsets like this are expected if the model is calibrated |

Reading the logistic "pushes" (scaled value x weight): they are relative to an *average* ball.
6 wickets in hand pushes *down* (-0.94) because the average training ball has 7.4 wickets;
needing only 30 (log_need +2.44) pushes strongly *up*. The sum of all pushes plus the
intercept is the score.

---

## Experiment: player quality (bowling and batting resources)

### Hypothesis
Knowing who is left to bowl (each bowler is capped at 4 overs) and how good the batters still
to come are should help most in the middle and death overs of a chase, and late in the first
innings.

### What we built (`src/player_features.py` -> `features_v3.parquet`)
- **Team sheets** for every match from the JSON (192,030 player-match rows), including IPL
  "impact player" substitutes recorded in the ball-by-ball data.
- **Career stats per player from strictly earlier dates**, shrunk toward typical values:
  batting average (runs per dismissal, prior 18 worth 3 dismissals), strike rate, bowling
  economy (prior 7.9 an over worth 120 balls), wickets per ball.
- **Current bowler** economy and wicket rate.
- **Remaining bowling**: at the start of each over, assume the fielding side uses its best
  remaining bowlers first, each capped at 4 overs, with a part-timer filling any gap; feature =
  expected runs and wickets per ball for the rest of the innings.
- **Batting resources**: sum of career averages of the batting side's players not yet out,
  and their mean strike rate.

### Checks
- Bumrah before the 2024 final, by hand: 4,698 balls, 6.97 an over raw, 7.00 shrunk. Pipeline: 6.997.
- In the final, batting resources fell 128.6 -> 95.6 when Klaasen was out; India's remaining
  bowling got weaker (7.31 -> 7.52 an over) as Bumrah's overs were used.
- **Bug caught before training**: the first version's "as of date" lookup skipped each player's
  most recent earlier match. Fixed by storing totals *including* each date and looking them up
  for *strictly later* dates only.

### Results (validation 2024, LightGBM log loss; change vs week 4, negative = better)
| Variant | All | 1st inns | 2nd inns | Powerplay | Middle | Death |
| --- | --- | --- | --- | --- | --- | --- |
| Week 4 baseline (seed 7) | 0.4768 | 0.5779 | 0.3621 | | | 0.4256 |
| + bowling | -0.0012 | -0.0029 | +0.0007 | -0.0033 | -0.0001 | -0.0006 |
| + batting | 0.0000 | -0.0017 | +0.0019 | -0.0025 | +0.0009 | +0.0017 |
| + both | -0.0015 | -0.0029 | +0.0001 | -0.0060 | +0.0005 | +0.0010 |
| *Seed noise (max-min of 3 runs)* | *0.0004* | *0.0012* | *0.0011* | *0.0004* | *0.0003* | *0.0007* |

### Decision: not added to the final model
Our pre-set rule was "keep only if validation log loss improves by more than 0.002".
Bowling (-0.0012) and both (-0.0015) are real (about 3x the seed noise) but below the bar;
batting is zero. So the final model stays at the week 4 feature set. The code stays in the
project so the idea can be revisited.

### Why the hypothesis failed
- **The gain appeared in the wrong place.** We expected the death overs of a chase; instead the
  only improvement is in the **first-innings powerplay**, i.e. *before the game has revealed
  much*. There, the XI's quality acts as a sharper version of team strength (who actually
  plays, not just the team name in Elo).
- **Later on, the scoreboard already knows.** By the death overs, required rate and wickets in
  hand carry most of the information; whether the next bowler is slightly better barely
  changes the outcome, and career averages are noisy (T20 batting is high-variance).
- **The remaining-bowling allocation is a guess.** Captains don't always use their best bowlers
  in order, and we can't see who is injured or has a bad day.

### General lessons
- **Decide the keep/drop rule before seeing the result.** Otherwise it's tempting to lower the
  bar for a feature you worked hard on. A small real gain still costs complexity, more data to
  collect live (the XI), and more places for leakage to hide.
- **Measure noise before judging differences.** Without the 3-seed baseline, a 0.001 change
  could be read as an improvement; it's about the size of run-to-run randomness.
- **A failed hypothesis is a result.** Recording it stops you (or a colleague) re-running the
  same idea, and explains in an interview *why* the model looks the way it does.
- **Where a feature helps is as informative as whether it helps.**

---

## Experiment batch 2: four feature groups tested together

### Groups (`src/batch_features.py` -> `features_v4.parquet`)
- **A. Set batters and momentum:** balls faced and runs by each current batter this innings,
  partnership runs/balls, runs and wickets in the previous 18 balls.
- **B. First-innings shape (chase only):** first-innings wickets lost, dot-ball share, runs in
  the last 5 overs, as proxies for how the pitch played.
- **C. Toss and home familiarity:** did the batting side win the toss, did the toss winner bat,
  share of each team's earlier matches played at this ground.
- **D. Elo v2:** home advantage (team with clearly more earlier matches at the ground), margin of
  victory (bigger wins move ratings more), ratings pulled toward 1500 at each new club season,
  and K re-chosen.

### Tuning Elo without touching validation
Elo makes its own pre-match prediction for every match, so its settings were chosen by Elo's
own log loss on **training-period matches (2010-2023)**: 72 combinations tried. Week 4's Elo
scored 0.6771; the best (K=24, home advantage 30 points, margin on, 25% season regression)
scored 0.6717. Validation then judged the tuned Elo once, like any other feature.
General trick: tune a component on its own objective, with training data, before it enters
the main model.

### Results (validation 2024, LightGBM log loss; change vs week 4, negative = better)
| Variant | All | 1st inns | 2nd inns | Pre-match | Powerplay | Middle | Death |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A set batters & momentum | +0.0003 | -0.0015 | +0.0023 | -0.0010 | -0.0003 | +0.0009 | -0.0002 |
| B first-innings shape | +0.0003 | 0.0000 | +0.0006 | 0.0000 | +0.0008 | +0.0001 | -0.0003 |
| C toss & home | +0.0019 | +0.0009 | +0.0030 | -0.0008 | +0.0012 | +0.0024 | +0.0017 |
| **D Elo v2** | **-0.0041** | **-0.0093** | +0.0018 | **-0.0130** | -0.0062 | -0.0030 | -0.0033 |
| All four | -0.0044 | -0.0090 | +0.0007 | -0.0147 | -0.0081 | -0.0026 | -0.0029 |

### Decision: adopt Elo v2 only
- D clears the 0.002 bar by 2x and is ~10x the seed noise. "All four" is only 0.0003 better than
  D alone (noise), so A, B and C are dropped.
- Leakage test for D, the **truncation test**: recompute Elo using only matches up to
  29 June 2024; every match up to that date gets identical ratings (171.2548 for the final both
  ways). If any later match leaked in, they would differ.
- Rebuilt the week 4 models with Elo v2 (`boosting.py` now reads `features_v4`):

| Model (validation log loss) | Week 4 | With Elo v2 |
| --- | --- | --- |
| Logistic + context | 0.4769 | 0.4701 |
| LightGBM + context | 0.4768 | 0.4720 |
| Average of both | 0.4743 | **0.4685** |

### Why the other groups didn't help
- **Momentum (A):** the "hot hand" is mostly noise. A batter on 40 off 25 does score a bit
  faster, but the model already has the run rate and wickets; the extra columns mostly gave
  LightGBM more ways to fit quirks of the training data (chases got worse).
- **First-innings shape (B):** `target_vs_par` already says whether the first innings was big
  for the ground; wickets lost and dot balls add little on top.
- **Toss and home (C):** the toss is close to a coin flip in effect, and home advantage was
  better captured *inside* Elo v2 (where it shifts the rating) than as separate columns.
- Pattern across both experiments: **improving one strong feature (Elo) beat adding many weak
  ones.** Pre-match team strength is where this model's uncertainty lives.

### General lessons
- **Refine what works before adding more.** The biggest gain in batch 2 came from improving an
  existing feature, not from new ideas.
- **More features can make a model worse** (A and C did). Each feature is a new way to overfit.
- **Truncation test** for anything computed sequentially: cut the data at a date, recompute,
  compare. Identical values = no look-ahead.

---

## Week 5, part 1: error analysis and calibration (validation 2024 only)

### Error analysis (`src/error_analysis.py`, `results/error_analysis.txt`)
- **The first innings carries 64% of all log loss** (powerplay 21%, middle 29%, death 15%).
  The chase is sharp late on (death overs 0.235).
- **By league:** T20 internationals are easiest (0.437; many mismatches between full members and
  associates). T20 Blast (0.540) and PSL (0.543) are hardest: evenly matched teams.
- **Bias in SA20:** the model averaged 45% for the batting side; actual 49%. A young league with
  little history for Elo.
- **The biggest misses were big totals that got chased:** CSK 210, KKR 223 and KKR 261 (chased by
  Punjab Kings, a record) in the IPL, Somerset 197 in the Blast, Japan 215 v Singapore. The
  model learned from history that 210+ is nearly safe; in the IPL's impact-player era (an extra
  batter since 2023) it isn't. Known limitation: nothing in our features tells the model the
  rule changed. Likely to show up again in the 2025-26 test.
- Pre-match favourite won 63.7% of 881 matches.

### Calibration (`src/calibration.py`)
Fitted on 2024 predictions from models trained only on 2005-23, judged by 5-fold cross-fitting
within 2024 (grouped by match, so no match is split).

| Calibrator | Log loss | Brier | Calibration error (ECE) |
| --- | --- | --- | --- |
| None | **0.4685** | 0.1573 | 1.24% |
| Platt | 0.4687 | 0.1575 | 0.98% |
| Platt per innings | 0.4706 | 0.1581 | 1.07% |
| Isotonic | 0.4693 | 0.1578 | 0.33% |

**Decision: no calibration layer.** Platt on all of 2024 gives slope 1.045, intercept 0.04,
which is almost "change nothing". Averaging logistic regression and LightGBM already produced
calibrated probabilities. Isotonic wins on ECE but loses on log loss: it fits a step per band,
partly overfitting, and ECE is measured on the very bands it adjusts.
Lesson: calibration layers fix systematically over- or under-confident models (common with
single boosted-tree or neural models); check before adding one.

---

## Week 5, part 2: the frozen model and the pre-registered test plan

Written on 2026-10-12 BEFORE any test-set (2025-26) data is scored.

### Frozen model
- **Two models per innings, averaged (50/50):**
  - Logistic regression: week 3 engineered state features + context (Elo v2, venue par,
    venue chase rate, scoring era, pace or target vs par, both batters' career runs/ball);
    StandardScaler; C = 1.0; missing values filled with training medians.
  - LightGBM: plain state columns + the same context; week 4 parameters; monotonic constraints;
    **fixed tree counts from validation early stopping: 74 (1st innings), 125 (2nd innings)**.
- **No calibration layer.**
- **Refit on all matches 2005-2024** (train + validation) with everything above fixed.
- Excluded matches as before: no result, D/L or other method, reduced overs, ties.

### Test plan
- **Data:** every usable match dated 1 Jan 2025 or later (1,539 matches, 343,227 balls).
- **Primary metric:** log loss over all test balls.
- **Comparisons**, all refit on 2005-2024: coin flip; lookup table; week 3 logistic (scoreboard
  only); the frozen model.
- **Success criteria (decided now):**
  1. The frozen model beats the lookup table and the scoreboard-only logistic model on log loss.
  2. Calibration error (ECE) below 2%.
  3. No slice (innings, phase, league) is worse than the scoreboard-only logistic model by
     more than 0.01.
- **Always reported, whatever they show:** innings, phase, league, pre-match, calibration chart.
- **Expectation:** test log loss somewhat worse than validation (0.4685), especially in the IPL,
  given the scoring change found in error analysis.
- **Rule:** no changes after seeing test results. Fixes go into a new version, judged on data
  that arrives later (e.g. the 2027 seasons).

---

## Week 5, part 3: the test set, scored once

`src/final_model.py` refit the frozen model on 2005-2024 (models/final/, with manifest.json).
`src/test_eval.py` scored 1,539 matches (343,227 balls), 1 Jan 2025 to 6 Oct 2026.

### Overall
| Model (all fitted on 2005-2024) | Log loss | Brier | Accuracy | Calibration error |
| --- | --- | --- | --- | --- |
| Coin flip | 0.6931 | 0.2500 | 48.3% | 1.7% |
| Lookup table | 0.5156 | 0.1738 | 73.2% | 1.5% |
| Logistic, scoreboard only (week 3) | 0.4858 | 0.1647 | 74.3% | 0.6% |
| **Final model** | **0.4448** | **0.1482** | **77.5%** | **1.5%** |

The final model cuts the scoreboard-only model's log loss by 8.4%, and is better in every
innings and phase: 1st innings 0.551 vs 0.602, chase 0.324 vs 0.354, pre-match 0.633 vs 0.694.

### Pre-registered criteria
1. Beats lookup and scoreboard-only logistic: **PASS**.
2. Calibration error below 2% (1.46%): **PASS**.
3. No slice worse than scoreboard-only by more than 0.01: **FAIL**. SA20 +0.016 (56 matches),
   CPL +0.010 (65 matches).

The verdict stands as written. Nothing was changed after seeing these numbers.

### Understanding the failure (analysis only, no changes)
- **Bootstrap confidence intervals** (resampling whole matches 2,000 times): SA20 +0.016,
  95% CI [-0.015, +0.047]; CPL +0.010, CI [-0.029, +0.054]. Both include zero, so with ~60
  matches we can't tell whether context genuinely hurts there. By contrast IPL -0.047
  [-0.079, -0.015] and T20Is -0.059 [-0.072, -0.045] are clear wins.
- **Franchise-league team strength is hard to know pre-match.** The pre-match favourite won
  69% of T20Is but only 45% (SA20), 47% (IPL), 48% (PSL), 51% (CPL), 56% (BBL), 61% (Blast) of
  test matches. Auctions and drafts reshuffle squads every season, so last season's Elo says
  little; in T20Is, stable teams and big mismatches (full members v associates) make Elo strong.
- **Chasing dominated in BBL and CPL in 2025-26:** batting first won only 35% of matches, while the
  model expected ~48%.
- Lesson for next time: **criteria on small slices need uncertainty built in** (e.g. "the 95%
  interval must not exceed +0.01"). A point estimate on 56 matches is mostly noise.

### Validation vs test
Test log loss (0.4448) came in *better* than validation (0.4685), against our expectation.
By league it's mixed: T20Is and the Blast improved, while the IPL (0.512 -> 0.549), BBL and CPL
got worse. The IPL result matches the warning from error analysis (impact-player era scoring).
Year-to-year variation of this size is normal; one year's number shouldn't be over-read.

### Calibration on test (`results/figures/test_calibration.png`)
Close to the diagonal throughout; slightly *under*-confident at the extremes (predicted 3.6% ->
actual 2.0%; predicted 85% -> actual 87%).

### Backlog for a version 2 (to be judged on data that arrives after this test)
- Franchise-league team strength: build it from the XI's player ratings, not the franchise name.
- An era flag for the IPL impact-player rule (2023 onwards).
- League-specific chase/defend tendencies, refreshed each season.
- Re-test bowling resources (pre-match benefit in the first innings).

### General lessons
- **Pre-register, then report whatever happens.** A failed criterion reported honestly is worth
  more than a passed one nobody trusts.
- **A test score is one draw.** Use bootstrap intervals and slice comparisons before concluding.
- **Small slices are noisy**; judge them with intervals, not point estimates.
- **The same feature can help in one setting and not another** (Elo: strong for national teams,
  weak for auction-built franchises). Understanding *why* is the real output of error analysis.

---

## Week 6: model vs the betting market (pre-registered plan)

Written on 2026-10-14 BEFORE any model-vs-market numbers were computed.

### Data
Betfair Exchange historic data, BASIC tier (free): MATCH_ODDS markets, Jan 2025 - Oct 2026,
8,276 markets (all cricket). Last traded price (LTP) per team, sampled about once a minute,
with timestamps. Matched to our 2025-26 test matches by date (+/- 1 day) and team names.

### How a price becomes a probability
Decimal odds 2.5 imply 1/2.5 = 40%. The two teams' implied probabilities sum to slightly more
or less than 100% (the market's margin and stale last-trade prices), so we normalise:
p_A = (1/odds_A) / (1/odds_A + 1/odds_B).

### Primary test: pre-match
- Market: last traded price for each team before the market turns in-play (first ball);
  both teams must have traded in the final 6 hours.
- Model: the frozen final model's prediction at the first ball (pre-match).
- Metric: log loss of each over the matched matches; difference with a 95% bootstrap interval
  (resampling matches). Also Brier and calibration.
- Expectation (stated in advance): **the market wins pre-match.** It knows team news, injuries,
  toss and pitch reports; our model only knows history.

### Secondary tests
1. **Blend:** does a 50/50 average of model and market beat the market alone? If yes, the model
   contains information the market lacks.
2. **Paper betting (no real money):** back a team when the model's probability exceeds the
   market's by 5+ percentage points, flat 1-unit stakes at the pre-match price, 5% commission on
   winnings. Report profit per unit staked with a 95% bootstrap interval. Expectation: negative.
3. **In-play (exploratory, not judged):** Cricsheet has no ball timestamps, so any ball-to-price
   alignment is approximate. Reported only if an alignment check passes.

### Rules
No changes to the model. Matching rules are fixed before results are computed; if matching
fails for some matches they are reported as unmatched, not hand-fixed afterward.

---

## Week 6: results

### Pipeline
- `src/parse_betfair.py`: 8,276 markets, 2.9M last-traded-price updates.
- `src/match_betfair.py`: linked test matches to markets by date (+/- 1 day) and fuzzy team names.
  First run: 1,314 linked, Betfair and Cricsheet winners agreed 97.4%. The disagreements were
  series with several games on consecutive days and associate double-headers, so an
  **ambiguity rule** was added (exclude a match if more than one market fits), before any
  model-vs-market numbers existed: 1,130 linked, winners agree 99.7%. Coverage: 96-100% of IPL,
  PSL, BBL, SA20; 91% Blast; 82% CPL; 59% T20Is (associates play back-to-back).
- `src/market_eval.py`: pre-match comparison, blend, paper betting.
- `src/market_inplay.py`: exploratory in-play comparison with an alignment check.

### Primary result: pre-match, 1,110 matches with prices for both teams
| Slice | Matches | Model log loss | Market log loss | Blend | Favourite won (model / market) |
| --- | --- | --- | --- | --- | --- |
| All | 1,110 | 0.650 | **0.565** | 0.597 | 60% / 68% |
| Franchise leagues | 590 | 0.698 | **0.676** | 0.684 | 53% / 58% |
| T20 internationals | 520 | 0.595 | **0.439** | 0.497 | 68% / 80% |

- Model minus market: **+0.085 log loss, 95% CI [+0.069, +0.102]**. The market is clearly better,
  as predicted in advance.
- **Blend** (50/50) is worse than the market alone (+0.032, CI [+0.023, +0.040]): the model holds
  no pre-match information the market lacks.
- Model and market probabilities correlate 0.64. The market's median overround is 1.001
  (last-traded prices on an exchange carry almost no margin).
- **Franchise leagues are close to unpredictable pre-match for everyone**: even the market
  scores 0.676, barely better than a coin flip (0.693). Our model is worse than a coin flip
  there (0.698), the same weakness found in week 5.
- In T20Is the market is far sharper (0.439): it knows the strength of associate sides and
  each team's actual XI much better than our Elo does.

### Paper betting (no real money)
Backing the side the model rated 5+ points above the market: 691 bets, 28% won at average odds
7.07, **profit per unit staked -0.25, 95% CI [-0.36, -0.13]**. T20Is -0.32, franchise -0.14.
The "edges" were mostly the model overrating outsiders: where the market and model disagree,
the market is usually right.

### In-play (exploratory): alignment check FAILED, not reported as a result
- Ball times were estimated by spreading balls evenly over the in-play window with a 15-minute
  innings break. Checking each match for the time shift that best lines up model and market:
  median shift **+6 minutes**, only 29% within 3 minutes (median correlation 0.89).
- With that misalignment, the raw numbers show the model "beating" the market in the chase
  (death overs 0.216 vs 0.287). **This is an artefact, not skill**: the model is effectively
  seeing balls a few minutes before the market price we pair it with, so it gets to react to
  wickets and boundaries the stale price hasn't priced yet. It's the same mistake as leakage,
  in timing form.
- A fair in-play test needs ball timestamps (paid data feeds, or the Advanced/PRO Betfair tier
  plus a ball-by-ball feed with times).

### What this means for the product
- **Don't sell betting edges.** The model doesn't beat the market pre-match, and the paper
  betting loses money with a clearly negative interval.
- **Its value is explanation, not prediction:** a transparent, calibrated win-probability line
  that shows *why* a match swung (wickets, required rate), match replays, biggest-swing moments,
  player impact. The market gives a number; it doesn't explain it.
- The honest market comparison is itself a strong portfolio piece: it shows the discipline to
  test against the toughest benchmark and report a negative result.

### General lessons
- **Markets are very hard to beat.** They aggregate everyone's information, including things no
  public dataset has (team news, injuries, the toss).
- **Check alignment before comparing time series.** A few minutes' misalignment manufactured a
  fake "win" for the model. Any comparison where one side can see information the other can't
  is invalid, however good the numbers look.
- **Linking datasets is a measurable step.** The winner-agreement check turned "I think the
  matching works" into "99.7% verified".
- **A blend test asks a sharper question than a head-to-head:** not "who is better" but "does my
  model know anything the market doesn't?"

---

## Week 7, part 1: version 2, a post-test fix (labelled as such)

### Why
Building the website exposed a flaw that the averages hid. In PBKS v KKR (IPL, 15 Apr 2025),
KKR were 95/8 needing 17 off 34 balls and the frozen model gave them 97.6%. They were bowled out
and lost. The error analysis had flagged the same slice: chases down to 1-2 wickets, where the
model was too sure of the chasing side.

**The rule for touching the model after the test:** the 2025-26 test score has been seen, so it
can't judge a new version fairly. The **v1 score (0.4448) stays the headline**. v2 is decided on
the 2024 validation set alone, with an acceptance rule written down before running. Any 2025-26
number for v2 is reported as "post-test".

### v2: expected runs remaining (`src/v2_resources.py`)
- **The idea comes from DLS (Duckworth-Lewis-Stern).** It builds a table of how many runs a side
  can still expect, given balls left and wickets in hand: E[runs to come | balls_left, wickets].
- **The table is built from FIRST innings only.** A first innings always runs to the end, so
  "runs still to come" is fully observed; a chase stops when the target is reached. It uses the
  fitting years only and is shrunk (K=30) towards the overs-left average. It is also forced to be
  monotone: more balls or more wickets never means fewer runs.
- **New features:** exp_runs_left; need_ratio (runs needed / expected runs, for a chase); and
  res_projected (first innings: runs + expected runs − venue par).
- **Acceptance rule:** the validation tail slice must improve, AND overall validation log loss
  must be no worse than v1 by more than 0.001.

| | v1 | v2 |
|---|---|---|
| Validation 2024, overall | 0.4685 | 0.4689 |
| Validation tail slice (≤3 wkts, need ≤30, ≥6 balls left) | 0.3622 | 0.3365 |
| Tail, 2 wickets left: model avg (actual 0.271) | 0.351 | 0.316 |
| Tail, 1 wicket left: model avg (actual 0.000) | 0.111 | 0.050 |
| Test 2025-26 (POST-TEST, not a clean test) | 0.4448 | 0.4445 |

The verdict was **ACCEPT**. But the motivating match barely moved: KKR went from 98.8% to 97.3%.

### Diagnosis (`src/diag_v2.py`, `src/diag_v2b.py`)
- **It isn't one feature.** Setting the context features (low target, batters, Elo) to neutral
  values still leaves the state alone at about 92-97%.
- **The state is very rare.** In 20 years only **11 chases** were ever at "2 wickets left, need
  12-22 off 25-45 balls". The chasers won 27%; v2 says 71%. Tree models and logistic regression
  both extrapolate from the thousands of "low required rate means win" examples with more wickets.
- **The average hides the spread.** Two tail-enders average about 23 more runs from 34 balls, so
  17 looks easy. But the spread is skewed: often they add 5, sometimes 50. What matters is the
  probability of scoring at least 17, not the mean.

### v3: chance of reaching the target (`src/v3_reach.py`), REJECTED
- **Feature:** reach_logit = logit P(runs to come ≥ runs needed | balls, wickets), taken from the
  empirical first-innings distribution. It pools ±3 balls (widened until there are at least 200
  rows) and is forced to be monotone. For 2 wickets and 34 balls, P(≥17) = 0.59.
- **Result:** overall validation improved to 0.4683, the best so far. But the tail slice got
  worse than v2 (0.3365 → 0.3407), so it fails the pre-stated rule.
- **The decision is to stop at v2.** The site uses v2.

### Lessons
- **Don't chase an anecdote.** Adding features until one famous match "looks right" is
  overfitting with extra steps. 11 matches can't tell a real fix from noise. The rule decides,
  not the example.
- **Averages can't describe risk.** A mean feature (expected runs) misses what drives the outcome,
  which is the chance of a collapse. That is why v3 was worth trying, even though it didn't pass.
- **Keep the frozen score as the headline.** Once a test set has influenced a design choice it is
  no longer a test set. New versions need new data (matches from after Oct 2026).
- **Moving work to more memory:** v3 training was killed on the laptop's 4GB VM, so it was run in
  the 8GB cloud container instead. Same code, same data, staged across.

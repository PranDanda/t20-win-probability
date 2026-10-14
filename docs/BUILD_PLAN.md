# T20 Win Probability Build Plan

Exported 12 Oct 2026 from the live doc (https://claude.ai/code/artifact/c1b73313-9ebe-4fbd-a027-c50bd1f7bc99). The live doc is the master copy.

Build a ball-by-ball T20 win-probability model on free Cricsheet data, prove it beats a simple baseline and holds up against betting-exchange prices, then ship it as a live site within 8 weeks.

## What we're building

A model that, after every legal ball of a men's T20 match, outputs the probability that each team wins. It is the cricket equivalent of a live option price: the match state goes in, a calibrated probability comes out.

- **Version 1 (state-only):** uses only the scoreboard — innings, balls left, wickets left, runs, target. Same idea as the Hundred version of WinViz, which deliberately ignores team strength.
- **Version 2 (context-aware):** adds venue scoring rates, a pre-match team-strength prior and who is still to bat. This is where an edge over simple models comes from.
- **Scope out for now:** women's cricket, ODI, Test, rain-affected (DLS) results, The Hundred's 100-ball format. Each is a later extension.

**Done means:** a public GitHub repo, a written backtest showing calibration and Brier score versus baselines, a comparison against Betfair exchange prices, and a live page plotting win probability for matches in progress.

## Data

Cricsheet gives free ball-by-ball JSON for every major men's T20 competition — enough for roughly 1.5–2 million deliveries once leagues are combined. It is licensed under the Open Data Commons Attribution Licence, which allows commercial use as long as you credit Cricsheet.

| Competition (men's) | Matches on Cricsheet |
| --- | --- |
| T20 internationals | 3,592 |
| T20 Blast | 1,570 |
| Indian Premier League | 1,243 |
| Syed Mushtaq Ali Trophy | 695 |
| Big Bash League | 662 |
| Bangladesh Premier League | 469 |
| Caribbean Premier League | 445 |
| Pakistan Super League | 357 |
| CSA T20 Challenge | 314 |
| SA20 | 130 |

Counts as listed on the [Cricsheet downloads page](https://cricsheet.org/downloads/) on 12 Oct 2026. About 379 matches involving Afghanistan are withheld site-wide.

**What each delivery gives you:** batting team, over and ball, batter, bowler, non-striker, runs (batter, extras, total), extras type, wicket and dismissal kind, plus match-level info: venue, date, toss, result and method (e.g. D/L).

**Odds data for the market backtest:**

- **Betfair Historic Data, Basic tier** — free; gives last traded price over time for each selection, including in-play. Enough to compare model probability with market probability ball by ball.
- **Betfair delayed app key** — free, delayed live prices for development.
- **OddsPapi free tier** — historical odds across many bookmakers including Betfair Exchange, 250 requests a month; a fallback if the Betfair files are awkward.

**Live feed (for the product, not the model):** Cricsheet updates after matches, not live. A live page needs a ball-by-ball feed — a paid sports-data API or scraping a scorecard site. Decide in week 6; check terms before scraping.

## Model

One row per delivery, labelled 1 if the batting team went on to win. Train separate models for the first and second innings: chasing a known target is a different problem, and published work finds the second-innings model much sharper.

**Features, built in this order:**

| Group | Features | Version |
| --- | --- | --- |
| Match state | balls remaining, wickets in hand, runs scored, current run rate | 1 |
| Chase state (2nd innings) | runs required, required run rate, required minus current rate | 1 |
| Phase | powerplay / middle / death overs flag | 1 |
| Venue | venue's average first-innings score and 2nd-innings win rate (from past matches only) | 2 |
| Team strength | pre-match Elo rating per team, updated after each match | 2 |
| Resources left | batting quality of players yet to bat; overs left for the best bowlers | 2 |
| Competition | league indicator (IPL, Blast, T20I…) to capture scoring-era differences | 2 |

Every feature must use only information available at that ball — venue averages and Elo are computed from earlier matches, never the match being predicted.

**Model ladder (each step must beat the last on held-out data):**

1. **Lookup baseline** — historical win rate for similar (balls left, wickets, runs needed) states. Hard to beat late in a chase; a good sanity check.
2. **Logistic regression** on match-state features. Interpretable and easy to explain in an interview.
3. **Gradient-boosted trees** (LightGBM) with all Version 2 features, monotonic constraints so more wickets in hand can never lower win probability.
4. **Calibration layer** — isotonic or Platt scaling on a held-out season, so a stated 70% wins about 70% of the time.
5. **Optional stretch:** a ball-outcome simulator (predict each ball's runs/wicket distribution, then Monte Carlo the rest of the match). Closer to how WinViz works and better for pricing side markets like total runs.

**Tools:** Python, pandas, LightGBM, scikit-learn, DuckDB or Parquet for storage.

## Evaluation

Judge the model on how well its probabilities match reality, not on accuracy. A model that says 55% for every match is "accurate" half the time and useless.

- **Split by time, never randomly.** Train on matches up to 2023, tune on 2024, test on 2025–26. Balls from the same match must never sit on both sides of a split.
- **Metrics:** log loss and Brier score (lower is better), reported overall and by phase (powerplay, middle, death). Accuracy only as a side note.
- **Calibration plot:** bucket predictions into 10% bands and plot predicted versus actual win rate. This one chart is the headline of the write-up.
- **Beat the baselines:** the lookup table and logistic regression from the model ladder. If LightGBM doesn't beat logistic regression on log loss, keep the simpler model.
- **Market backtest:** for test-set matches with Betfair data, compare model probability with the exchange's implied probability (1 ÷ decimal odds) at the same ball. Report the log loss of each, and how often a simple rule — back when the model is 5+ points above the market — would have made or lost money after 5% commission. Treat a positive result with suspicion until it holds across seasons and leagues.
- **Sanity checks:** probabilities move the right way after a wicket or a six; the chasing team sits at 0% or 100% at the final ball; no leakage from the result into features.

## Product and money

The model itself is not new — CricViz's WinViz and ESPNcricinfo's Forecaster already show live win probability — so the value is in what you build on top and in a public, verifiable track record.

| Competitor | What it offers | Gap you can fill |
| --- | --- | --- |
| CricViz WinViz | Live win probability in its app and on Sky/BBC broadcasts | Black box; no published calibration or market comparison |
| ESPNcricinfo Forecaster | Projected score and win probability during IPL coverage | Tied to their site; no odds comparison |
| Hobby models (blogs, GitHub) | One-off R/Python notebooks | No live product, no backtest against the market |

**What ships first:** a site with a live win-probability chart per match, a "biggest swing" moment for each match, and a public scorecard of the model's calibration.

**Revenue routes, in order of effort:**

1. **Newsletter + social clips** — free, built around the biggest swings each week ("this six moved the match 31 points"). Builds an audience; sponsorship later.
2. **Model-vs-market alerts** — paid tier flagging when model and exchange disagree. Only once the backtest holds up; it carries gambling-advertising and responsible-gambling obligations, so take advice first.
3. **Data API** — win probability and player impact (win probability added) for fantasy sites, creators and smaller media.
4. **Career value** — the backtest write-up is a strong quant-style portfolio piece regardless of revenue.

## Build plan

Eight weeks at roughly 6–8 hours a week. Each week ends with something you can show.

| Week | Work | Shows at the end |
| --- | --- | --- |
| 1 | Download men's T20 JSON from Cricsheet; parse into one deliveries table (Parquet/DuckDB); add match results; drop no-results and DLS matches | Clean dataset + row counts per league |
| 2 | Build match-state features for both innings; time-based train/validation/test split; lookup-table baseline | Baseline log loss and Brier score |
| 3 | Logistic regression per innings; calibration plot; sanity checks on known matches | First win-probability chart for a famous match |
| 4 | Version 2 features: venue averages, Elo ratings, batters still to come; LightGBM with monotonic constraints | Comparison table: baseline vs logistic vs LightGBM |
| 5 | Calibration layer; error analysis by phase and league; freeze the model | Final test-set results |
| 6 | Get Betfair Basic historic data for test matches; align prices to balls; run the market backtest; pick a live data feed | Model vs market write-up |
| 7 | Simple site: past-match replays with win-probability charts, biggest-swing moments, calibration page | Live site on free hosting |
| 8 | Live mode for one league in season; README, blog post, short video walkthrough | Public launch + portfolio piece |

The next league windows to aim the live launch at are the Big Bash (Dec–Jan) and SA20 (Jan–Feb); check this season's dates.

## Risks and open questions

- **Leakage** is the most likely way to fool yourself: venue and team stats must come from earlier matches only.
- **The market is hard to beat.** Exchange prices already absorb team news and conditions; a model that matches the market is still a good outcome for a portfolio.
- **Scoring inflation:** T20 totals have risen sharply in recent seasons (especially the IPL), so old matches may mislead the model. Test down-weighting older seasons.
- **Live data cost** is the main unknown for the product; Cricsheet is post-match only.
- **Gambling rules:** any betting-related feature in the UK brings advertising and responsible-gambling obligations.
- **Employer conflict:** check my employer's policy on outside business activities before charging money.

- [ ] Which league to launch live with first?
- [ ] Keep the Hundred out, or adapt the model to 100 balls later?

## Sources

- [Cricsheet — match data downloads](https://cricsheet.org/downloads/)
- [Cricsheet — licensing](https://cricsheet.org/licensing/)
- [CricViz — WinViz on the Hundred broadcasts](https://ellipsedata.com/?p=642)
- [CricViz — enhanced WinViz model](https://ellipsedata.com/?p=39772)
- [IIT Madras — ESPNcricinfo Superstats Forecaster](https://acr.iitm.ac.in/?p=65023)
- [R-bloggers — T20 win probability with logistic regression](https://www.r-bloggers.com/2020/06/twenty20-win-probability-added/)
- [Betfair — historic data workbook (Basic tier fields)](https://github.com/betfair/historic-data-workbook)
- [Betfair — API plans and pricing](https://apis.io/plans/betfair/betfair-plans-pricing/)
- [OddsPapi — free tier and historical odds](https://oddspapi.io/blog/?p=3145)

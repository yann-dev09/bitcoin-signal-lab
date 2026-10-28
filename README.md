# Crypto-predictor1

Historical backtesting of a **combined Bitcoin signal** that fuses technical
analysis (RSI + hammer candles), on-chain fundamentals (active addresses &
transaction count), and market sentiment.

## Pipeline overview

| Step | Script                                   | Output / artifact          | Description |
|------|------------------------------------------|----------------------------|-------------|
| 0    | `fetch_btc_data.py`                      | `btc_data.csv`             | Daily BTC OHLC (Binance, up to 730 days) |
| 0b   | `fetch_btc_data.py --full`               | `btc_data_full.csv`        | Full daily BTC OHLC (2018-02-01 -> today, paginated) |
| 1    | `fetch_fundamentals_history.py`          | `fundamentals_history.csv` | On-chain history (Blockchain.com `timespan=all`, cut to 2018-02-01) |
| 2    | `fetch_sentiment_history.py`             | `fng_history.csv`          | Full Fear & Greed Index history (proxy) |
| ~    | `verify_data.py`                         | (printed report)           | Verify ranges / row counts / missing days for all 3 CSVs |
| 3    | `signal_combiner.py` + `test_signal_combiner.py` | —                    | Combined signal logic + unit tests |
| 4    | `backtest_full.py`                       | (printed report)           | Full combined-signal backtest (2-year window) |
| 5    | `data_utils.py`                          | —                          | Shared merged dataset + feature engineering (2018-2026) |
| 6    | `backtest_variants.py`                   | (printed report)           | 5 rule-based strategies, analysis + hold-out 80/20 |
| 7    | `model_walkforward.py`                   | `model_walkforward_predictions.csv`, `walkforward_equity.png` | Expanding-window logistic regression (analysis + hold-out) |

### Running the pipeline

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Fetch BTC OHLC
python fetch_btc_data.py --days 730

# 3. Fetch 2-year on-chain history
python fetch_fundamentals_history.py        # -> fundamentals_history.csv

# 4. Fetch sentiment proxy history
python fetch_sentiment_history.py           # -> fng_history.csv

# 5. Run the signal-combiner unit tests
python test_signal_combiner.py              # -> 10 passed

# 6. Run the full combined-signal backtest
python backtest_full.py
```

### Multi-year analysis (2018-2026)

The extended pipeline evaluates on the full **2018-02-01 -> today** window using a
date-based 80/20 split: the first 80% of days is the *analysis* window
(threshold / model selection + walk-forward reporting) and the last 20% is the
*hold-out* window, which is touched **only once**, in a single final forward run.

```bash
# 1. Fetch the full daily BTC OHLC history (2018 -> today, paginated)
python fetch_btc_data.py --full               # -> btc_data_full.csv

# 2. Refresh on-chain + sentiment histories (cut to 2018-02-01)
python fetch_fundamentals_history.py          # -> fundamentals_history.csv
python fetch_sentiment_history.py             # -> fng_history.csv

# 3. Verify data integrity (ranges, counts, missing days)
python verify_data.py

# 4. Five rule-based strategies - analysis window (first 80%)
python backtest_variants.py

# 5. Expanding-window logistic model - analysis window (first 80%)
python model_walkforward.py

# 6. FINAL STEP - single forward report on the hold-out window (last 20%)
python backtest_variants.py --holdout
python model_walkforward.py --holdout
```

> The original live/snapshot tools are unchanged: `sentiment.py` +
> `fetch_news.py` (CoinDesk RSS + VADER), `fetch_fundamentals.py`
> (30-day Blockchain.com snapshot), `detect_hammer.py`, `analyze.py`,
> and `backtest.py` (technical-only backtest).

## How each script works

### `fetch_btc_data.py`
Fetches daily BTC/USD OHLC. Uses CoinGecko for the last 90 days and Binance
klines for longer ranges. Saves `btc_data.csv`
(columns: `date, open, high, low, close`).

`--full` paginates Binance klines in 1000-candle chunks (with `startTime`)
starting 2018-02-01 through the latest available day and writes
`btc_data_full.csv` — the master daily calendar for the multi-year pipeline
(3192 consecutive daily rows, no gaps).

### `fetch_fundamentals_history.py`  *(Step 1)*
Downloads two Blockchain.com **charts** series with `timespan=all&format=json`
(the longest range Blockchain.com offers, 2009 -> today) and cuts the merged
result to **2018-02-01 onward** so every data file shares the same analysis
window:
* `n-unique-addresses` -> `unique_addresses`
* `n-transactions` -> `transactions`

Note on granularity: `timespan=all` is auto-coarsened by Blockchain.com to
~4-day sampling across the full span (the two metrics land on slightly offset
4-day grids). The script therefore merges by the **union** of dates, with each
metric forward-filled onto the union grid (a plain inner join would discard
most samples). Written to `fundamentals_history.csv` (columns:
`date, unique_addresses, transactions`); `data_utils.py` later forward-fills
this onto the daily BTC calendar. Missing calendar days are reported at the end
of the run.

### `fetch_sentiment_history.py`  *(Step 2)*
Downloads the **full** Fear & Greed Index history from
`https://api.alternative.me/fng/?limit=0&format=json` and writes
`fng_history.csv` (columns: `date, value, sentiment_score`).

`value` is the raw 0-100 FNG reading. `sentiment_score = (value - 50) / 50`
rescales it to a -1..+1 range so it is roughly comparable in magnitude to a
VADER compound score, letting it plug into the combiner's `>= 0.05` /
`<= -0.2` thresholds.

> **IMPORTANT - this is a PROXY, not the same data as `sentiment.py`.**
> `sentiment.py` scores real CoinDesk RSS headlines with VADER. The Fear &
> Greed Index is an aggregate market-sentiment indicator (volatility,
> momentum, social-media, Bitcoin dominance) and is NOT VADER headline
> sentiment. It is used only because CoinDesk RSS cannot supply multi-year
> daily history.

### `signal_combiner.py`  *(Step 3)*
Single function:

```python
determine_signal(rsi, has_hammer, sentiment, addr_change_7d, tx_change_7d) -> dict
```

It reproduces exactly the decision logic of `determineSignal()` in
`index.html`, with the same thresholds:

* Bullish reason #1: `RSI < 30` AND `sentiment >= 0.05`
* Bullish reason #2: `has_hammer` AND `sentiment > -0.2` (i.e. `!isStronglyNegative`)
* If any bullish reason exists:
  - `transactions` 7-day drop > 10% -> "Cautious" (on-chain veto)
  - else `addresses > 0` AND `transactions > 0` -> "Bullish" (stronger)
  - else -> "Bullish"
* Otherwise -> "Neutral"

It also adds the requested **Bearish** branch (not present in the JS):

* `RSI > 70` AND `sentiment <= 0.05` -> "Bearish"

The Bearish branch is evaluated **first**. `RSI > 70` (overbought) and
`RSI < 30` (oversold) are mutually exclusive, so it never conflicts with the
RSI-driven bullish reason. The only overlap is a hammer co-occurring with an
overbought, non-positive-sentiment day, and there Bearish correctly takes
priority. `None`/NaN inputs (no RSI yet, or no on-chain data) are handled the
same way as the JS `null` checks.

`test_signal_combiner.py` has 10 assert-based cases (run with
`python test_signal_combiner.py`).

### `backtest_full.py`  *(Step 4)*
Merges the three CSVs by date, using the **BTC daily OHLC series as the master
calendar** (it is consecutive daily). On-chain metrics and the FNG sentiment
score are reindexed onto that calendar with **forward-fill**. Forward-fill is
**not** look-ahead: a missing day inherits the most recent *past* observation,
exactly as a trader would have seen it.

For each day (using only same-day / prior data):
1. RSI(14), Wilder smoothing (same code as `analyze.py`).
2. Hammer detection via `detect_hammer.is_hammer`.
3. 7-day % change of unique addresses and transactions
   (`pct_change(periods=7)` on the consecutive daily frame -> calendar-correct).
4. Combined signal via `signal_combiner.determine_signal`.
5. **Next 1-, 3- and 7-day** forward returns (`close[t+N]/close[t]-1`) are
   computed as the *outcome* (the label, not a signal input).

For each signal class (Bullish, Bearish, Cautious, Neutral) and each horizon
it reports: number of signals, up-rate, average and median forward return, the
baseline (all-days) up-rate, the edge in percentage points, and a two-sided
**binomial test** p-value (`scipy.stats.binomtest`, H0: signal up-rate ==
baseline up-rate). A 70/30 date-based train/test split is reported separately.
The script prints an explicit verdict when no class is significant.

## Results summary (backtest_full, 7-day horizon)

* Total valid days: **723**
* Signal counts: **Bullish 3, Bearish 1, Cautious 2, Neutral 717**
  (99.2% Neutral)
* Baseline 7-day up-rate: **52.4%**
* **No signal class reached statistical significance (p >= 0.05) at any
  horizon.** The combined signal fires too rarely to evaluate statistically.

### Why is the signal so sparse?
* The RSI-bullish conjunction (`RSI < 30` AND positive sentiment) **never
  occurs** -- whenever BTC was oversold, market fear was high (FNG negative),
  so the two conditions are anti-correlated.
* Genuine hammer candles are rare (8 in 730 days), and the on-chain "Cautious"
  veto (`tx` drop > 10%) further filters them.
* The Bearish conjunction (`RSI > 70` AND `sentiment <= 0.05`) fires only once.

### Verdict
The full combined signal is **too conservative to produce a statistically
evaluable edge** over this 2-year window. Observed differences in the rare
non-Neutral signals are consistent with small-sample noise. To obtain a
meaningful edge, thresholds would need to be relaxed (e.g. decouple the
sentiment requirement, or lower the hammer rarity) -- left as a future task.
The pipeline and honest numbers are reproducible via `python backtest_full.py`.

## Results summary (multi-year 80/20, 2018-02-01 -> 2026-10-28)

### Step 1 - data (`verify_data.py`)
* BTC daily: **3192 rows, 2018-02-01 -> 2026-10-28, 0 missing days**.
* FNG daily: **3188 rows** (4 calendar gaps, forward-filled).
* On-chain: **1389 rows at ~4-day sampling** (union-forward-filled to daily
  inside `data_utils.py`). The 4-day on-chain cadence is consumed only by the
  Step 7 model; none of the Step 6 rules use on-chain data.

### Step 6 - five rule-based strategies (analysis window, 2476 days)
Buy & Hold baseline: **+943% / +41.3% annual / Sharpe 0.86**.

| Rule | Long days | Total return | Annual return | Sharpe | 7-day win rate |
|------|-----------|--------------|---------------|--------|----------------|
| R2: Mom7>0 & price>MA90 | 18.9% | +1155% | +45.2% | **1.16** | 52.7% |
| R5: price>MA30 & Mom7>0 & sent>0 | 14.9% | +962% | +41.7% | **1.16** | 54.8% |
| R1: Mom30>2.5% & VOL30<50% | 8.6% | +213% | +18.3% | 0.86 | 56.7% |
| R3: Mom30>5% & RSI14<70 | 14.5% | +172% | +15.9% | 0.61 | 56.3% |
| R4: Onchain>0 & sentiment>0 | 12.1% | +115% | +12.0% | 0.53 | 54.7% |

Two momentum rules (R2, R5) beat buy-and-hold on a risk-adjusted basis
(Sharpe 1.16 vs 0.86) with ~80-85% less market exposure. The on-chain+sentiment
rule (R4) underperforms, so on-chain metrics do not add value with this signal
definition.

### Step 7 - logistic regression walk-forward (analysis window, 2392 scored days)
* AUC: **0.455** (no-skill baseline ~0.50). In-sample AUC is 0.545 - the slight
  in-sample signal does **not** generalize in the expanding walk-forward, i.e.
  next-7-day BTC direction is not predictable from this feature set
  (consistent with weak-form market efficiency over 2018-2026).
* F1-optimal threshold (0.10) -> strategy ~= buy-and-hold (+46.9% annual,
  Sharpe 0.92): the model only adds value by reducing exposure, not by
  timing the market.
* Strategy at the natural 0.50 threshold loses money (-14.0% annual, Sharpe
  -0.01, 31.6% win rate): the model's "confident up" calls are
  anti-correlated, further confirming no edge.
* Top standardized coefficients (standardized features): RSI14 +0.25,
  momentum_30 +0.18, ma_ratio_30_90 +0.10; negatives on momentum_7 -0.11
  and returns_1d -0.08 (short-horizon reversals).

### Hold-out (final, touch-once)
Run `python backtest_variants.py --holdout` and
`python model_walkforward.py --holdout` for the single forward report on the
last 20% of days (2025-02-15 -> 2026-10-28). The threshold is the one selected
on the analysis window and is **not** re-tuned on the hold-out.

**Overall conclusion:** the pipeline is correct, leak-free, and fully
reproducible from `fetch_*_history.py` -> `verify_data.py` -> the analysis and
hold-out scripts. The two momentum rules (R2/R5) offer the only
risk-adjusted edge; the logistic model does not beat a random baseline on
next-7-day direction over this window.

### `verify_data.py`  *(Step 1)*
Loads the three CSVs and prints, for each, the date range, row count, and number
of missing calendar days within the 2018-02-01 onward window. Run it after
fetching to sanity-check that the three series are aligned and complete.

### `data_utils.py`  *(Step 5)*
The shared data layer. `load_merged_dataset()` builds one daily DataFrame by
taking `btc_data_full.csv` as the master calendar and reindexing FNG and on-chain
metrics onto it with forward-fill (no lookahead: each day inherits the most
recent past observation). It then engineers leak-free features using only
same-day/prior data:

* `returns_1d`, `ma_7`/`ma_30`/`ma_90`, MA cross-ratios (`ma_ratio_7_30` etc.),
* 14-day Wilder RSI, 30-day annualized volatility, 7-day & 30-day momentum,
* forward-filled `sentiment_score`, `unique_addresses`, `transactions`,
* 7-day on-chain changes (`addr_7d`, `tx_7d`).

The prediction target is the sign of the next-7-day return
(`future_return_7d = close[t+7]/close[t] - 1 > 0 -> target_7d = 1`).
`split_dataset()` splits by date 80/20 (analysis / hold-out).

### `backtest_variants.py`  *(Step 6)*
Five simple rule-based long-only strategies, each evaluated next-day
(signal on day t -> position on day t+1, no lookahead). The five rules:

* R1: 30-day momentum > 2.5% AND 30-day volatility < 50%
* R2: 7-day momentum > 0 AND price above 90-day MA
* R3: 30-day momentum > 5% AND 14-day RSI < 70 (not overbought)
* R4: 7-day on-chain activity rising AND sentiment > 0
* R5: price above 30-day MA AND 7-day momentum > 0 AND sentiment > 0

For each rule it reports long/flat days, 7-day forward-return stats (avg
return, win rate, precision of "long" calls), and daily-position strategy
stats (total/annualized return, Sharpe, win rate) vs buy & hold.

### `model_walkforward.py`  *(Step 7)*
An expanding-window logistic regression predicting `target_7d`. It is retrained
every 30 days on all data strictly before day t (expanding window, no lookahead)
and predicts daily in between. The probability threshold that maximizes F1
**on the analysis window** is selected there and reused unchanged on the
hold-out. Per window it reports AUC (threshold-independent), classification
metrics at the threshold, and the long-only strategy metrics (at the F1
threshold and at 0.50) vs buy & hold. Per-day predictions are written to
`model_walkforward_predictions.csv` and an equity curve to
`walkforward_equity.png`.

> Retraining every 30 days (rather than every day) keeps the expanding refit
> from becoming O(n^2); 30 days of history between fits is stable for this
> standardized feature set.

## Files

```
fetch_btc_data.py              BTC OHLC fetcher (+ --full pagination -> btc_data_full.csv)
fetch_fundamentals.py          30-day on-chain snapshot (live tool, unchanged)
fetch_fundamentals_history.py  On-chain history (Blockchain.com timespan=all, cut to 2018-02-01)
fetch_sentiment_history.py     Full Fear & Greed history
verify_data.py                 Data integrity check (NEW, Step 1)
sentiment.py / fetch_news.py   VADER headline sentiment (live tool, unchanged)
detect_hammer.py               Hammer candlestick detector
analyze.py                     RSI(14) + SMA analysis
signal_combiner.py             Combined signal logic (NEW, Step 3)
test_signal_combiner.py        Unit tests for the combiner (NEW, Step 3)
backtest.py                    Technical-only (RSI + hammer) backtest (unchanged)
backtest_full.py               Full combined-signal backtest (2-year window)
data_utils.py                  Shared merged dataset + features (NEW, Step 5)
backtest_variants.py           5 rule-based strategies (NEW, Step 6)
model_walkforward.py           Expanding-window logistic model (NEW, Step 7)
| stress_test_rules.py            Cost + robustness + significance stress test of R2/R5 (NEW)
| paper_trade.py                  Daily forward paper-trading logger (MA100 live; R2 & BH counterfactuals) (NEW)
| paper_report.py                 Performance report over paper_log.csv (NEW)
| RULES_FROZEN.md                 Frozen paper-trading rule parameters + 90-day evaluation note (NEW)
| paper_log.csv                   Paper-trading ledger (one row per closed day) (NEW)
index.html                     Live dashboard / JS signal combiner
requirements.txt               Python dependencies (+ scikit-learn, matplotlib)
README.md                      This file
```

## Data artifacts

`.json` snapshot files (`fundamentals.json`, `sentiment.json`, `hammers.json`)
are produced by the live snapshot tools; the new `_history.csv` files feed the
historical backtest.

## Stress test: do the trend rules R2/R5 survive costs and beat a baseline?

`stress_test_rules.py` re-runs R2 and R5 from `backtest_variants.py` **with
transaction costs** (0.1% fee + 0.05% slippage per side = 0.30% round-trip on
every entry/exit), adds simple moving-average baselines (long when
`close > MA`), and runs four robustness/significance checks. Rule parameters are
**fixed** by the Step-6 definitions (R2 = close > MA(90) & 7-day momentum > 0;
R5 = close > MA(30) & 7-day momentum > 0 & sentiment > 0). The MA-length /
momentum-window sweep is run on the **analysis** window only; the hold-out is
touched only for reporting (Task 1) and a single final run of the chosen rule.

Run it with `python stress_test_rules.py` (analysis-window evidence only).

### Tasks 1-2 - cost-adjusted performance vs buy & hold (0.15% per side)

ANALYSIS window (2018-02-02 -> 2025-01-27, 2552 days):

| rule       | total % | ann %  | Sharpe | max DD % | win % | trades |
|------------|---------|--------|--------|----------|-------|--------|
| Buy & Hold | 1006.65 | 41.03  | 0.853  | -76.63   | 51.3  | 0      |
| R2         | 738.66  | 35.55  | 0.991  | -50.20   | 18.0  | 139    |
| R5         | 581.37  | 31.58  | 0.957  | -44.11   | 14.3  | 126    |
| MA200-long | 1182.45 | 44.04  | 1.034  | -64.38   | 27.6  | 22     |
| MA100-long | 2448.85 | 58.91  | 1.246  | -38.80   | 28.3  | 41     |

HOLD-OUT window (2025-01-28 -> 2026-10-28, 639 days):

| rule       | total % | ann %  | Sharpe | max DD % | win % | trades |
|------------|---------|--------|--------|----------|-------|--------|
| Buy & Hold | -21.17  | -12.71 | -0.102 | -52.97   | 49.1  | 0      |
| R2         | -1.64   | -0.94  | 0.034  | -18.54   | 13.0  | 27     |
| R5         | -6.70   | -3.88  | -0.184 | -17.23   | 9.1   | 20     |
| MA200-long | -1.59   | -0.91  | 0.037  | -19.95   | 11.3  | 5      |
| MA100-long | +10.46  | +5.85  | 0.369  | -24.89   | 19.7  | 9      |

R2/R5 beat buy-and-hold on Sharpe (0.99 / 0.96 vs 0.85) with roughly half the
drawdown, but mostly by sitting in cash through the 2018 and 2022 crashes - a
volatility-screen effect, not trend-following alpha. **They do not clearly beat
the trivial `close > MA(100)` baseline**, which dominates on both windows
(analysis Sharpe 1.246; hold-out +5.85% annualized).

### 3 - R2 parameter robustness (analysis window, Sharpe with costs)

|        | mom5  | mom7  | mom10 | mom14 |
|--------|-------|-------|-------|-------|
| MA50   | 0.974 | 1.141 | 1.200 | 1.276 |
| MA90   | 0.881 | 0.991 | 1.069 | 1.283 |
| MA150  | 0.967 | 1.053 | 1.044 | 1.404 |
| MA200  | 0.878 | 0.896 | 0.971 | 1.322 |

Grid mean 1.084; the default (MA90, mom7) = 0.991 is mid-pack. Sharpe rises
**monotonically** with the momentum lookback (mom5 = 0.93 -> mom14 = 1.32), so the
rule is **parameter-sensitive** rather than a robust plateau.

### 4 - bootstrap significance (block bootstrap, B=1000, 20-day blocks, analysis)

| rule        | mean Sharpe(diff vs BH) | 95% CI        | p(diff<=0) | verdict  |
|-------------|-------------------------|---------------|------------|----------|
| R2          | +0.127                  | [-0.49, +0.77] | 0.33     | NOT-sig  |
| R5          | +0.074                  | [-0.62, +0.76] | 0.41     | NOT-sig  |
| MA200-long  | +0.165                  | [-0.49, +0.79] | 0.29     | NOT-sig  |
| MA100-long  | +0.363                  | [-0.22, +0.95] | 0.10     | NOT-sig  |

Every 95% CI crosses zero: **no rule's Sharpe edge over buy-and-hold is
statistically significant** (even MA100-long is only ~significant at the 10% level).

### 5 - per-year (2018-2026, descriptive, full history, with costs)

| year | BH ret % | BH mdd % | R2 ret % | R2 mdd % | MA100 ret % | MA100 mdd % |
|------|----------|----------|----------|----------|-------------|-------------|
| 2018 | -59.86   | -72.11   | -7.98    | -13.61   | -9.02       | -16.23      |
| 2019 | +94.31   | -49.41   | +118.47  | -29.18   | +133.12     | -28.54      |
| 2020 | +301.98  | -53.60   | +84.83   | -29.89   | +166.23     | -25.29      |
| 2021 | +59.79   | -53.14   | +54.26   | -33.33   | +76.80      | -36.16      |
| 2022 | -64.21   | -66.93   | -25.30   | -28.06   | -21.13      | -29.45      |
| 2023 | +155.61  | -20.00   | +50.29   | -24.21   | +81.48      | -27.02      |
| 2024 | +121.31  | -26.15   | +35.06   | -29.12   | +63.50      | -36.68      |
| 2025 | -6.33    | -32.02   | -1.27    | -13.08   | +3.82       | -18.28      |
| 2026 | -8.19    | -39.53   | -5.19    | -12.47   | +6.52       | -13.27      |

R2/R5 act as tail-risk screens in the 2018/2022/2025/2026 drawdown years, but
underperform big bull markets (2019/2020/2021) where they sit in cash.

### 6 - pick + final hold-out run

Picked **R2** (the stronger of the two trend rules: analysis Sharpe 0.991 vs
R5 0.957, lower max drawdown). Final hold-out run (639 days): **R2 = -0.94%
annual (Sharpe 0.034), max DD -18.5%, 27 trades, vs Buy & Hold -12.71%
(Sharpe -0.102)** - R2 preserved capital against a crashing buy-and-hold but did
not finish in the black.

### Honest verdict (R2/R5)

**No statistically significant edge.** With realistic costs the trend rules
R2/R5 do beat buy-and-hold on Sharpe and drawdown (mostly as a volatility filter
that hides in cash during crashes), but (a) they do **not** beat the trivial
`close > MA(100)` baseline on either window, (b) R2 is parameter-sensitive rather
than a robust plateau, (c) the Sharpe edge vs buy-and-hold is not
bootstrap-significant, and (d) they underperform the big bull years. Because a
simple moving-average rule already dominates, there is no justification to deploy
the momentum/combination rules for return; use MA(100) long, or treat R2 as a
defensive tail-risk screen, not an edge.

## Paper-trading (daily, forward-only)

`paper_trade.py` + `paper_report.py` run a live **forward paper-trade** of the
recommended rule (MA100 long) from `RULES_FROZEN.md`, with R2 and buy-and-hold
tracked as paper counterfactuals. No exchange API, no real money, no order
placement - it appends one row per closed daily candle to `paper_log.csv`. This
is a fresh forward test (does **not** use the stress-test hold-out).

Columns: `date, close, ma100, signal_ma100, signal_r2, position, fng, rsi_14,
paper_equity_ma100, paper_equity_r2, paper_equity_buyhold, trades_ma100,
trades_r2`. `position` = next-day target for the live MA100 rule (1=long/0=cash);
the `paper_equity_*` columns are the portfolio value at the close of `date`
(position held = signal of the prior day, no lookahead). `fng`/`rsi_14` are
informational. Each strategy starts at $1000; 0.15% per side on each
entry/exit.

```bat
python paper_trade.py            :: refresh data, log the last closed candle
python paper_trade.py --dry-run  :: preview without writing paper_log.csv
python paper_trade.py --no-refresh :: skip data refresh (offline re-runs)
python paper_report.py           :: equity / drawdown / trades vs buy & hold
```

A re-run on the same day is a no-op ("Up to date ... already logged").

### Automate it daily with Windows Task Scheduler

1. Folder = `C:\Users\USER\Desktop\Crypto-predictor1`.
2. Create `C:\tasks\run_paper_trade.bat`:
   ```bat
   @echo off
   cd /d "C:\Users\USER\Desktop\Crypto-predictor1"
   python paper_trade.py >> "C:\tasks\paper_trade.log" 2>&1
   ```
3. Open **Task Scheduler** -> *Create Task*.
4. **General**: Name = `BTC paper-trade`; tick *Run whether user is logged on or
   not* and *Run with highest privileges*.
5. **Triggers** -> *New* -> Daily, repeat every 1 day, Start = **00:10 UTC**.
   (A Binance 1d candle for date D closes at 00:00 UTC, so 00:10 UTC logs the
   previous day's closed candle with no lookahead. Convert to your local time,
   e.g. 05:10 for UTC+5.)
6. **Actions** -> *New* -> Action = *Start a program*, Program/script =
   `C:\tasks\run_paper_trade.bat`.
7. **Conditions**: un-tick *Start the task only if the computer is on AC power*
   if you want it to run on battery.
8. **OK** (you'll be asked for your Windows password).
9. Verify: right-click the task -> *Run*, then open `C:\tasks\paper_trade.log`
   and `python paper_report.py`.

### Evaluation
`RULES_FROZEN.md` fixes the rules; **evaluation date = first logged date + 90
days**. `paper_report.py` prints the start date, last logged date, and the
evaluation date each time it runs.




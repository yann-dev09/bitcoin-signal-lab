# Crypto-predictor1

Historical backtesting of a **combined Bitcoin signal** that fuses technical
analysis (RSI + hammer candles), on-chain fundamentals (active addresses &
transaction count), and market sentiment.

## Pipeline overview

| Step | Script                          | Output CSV / artifact       | Description |
|------|---------------------------------|-----------------------------|-------------|
| 0    | `fetch_btc_data.py`             | `btc_data.csv`              | Daily BTC OHLC (Binance, up to 730 days) |
| 1    | `fetch_fundamentals_history.py` | `fundamentals_history.csv`  | 2-year on-chain history (Blockchain.com) |
| 2    | `fetch_sentiment_history.py`    | `fng_history.csv`           | Full Fear & Greed Index history (proxy) |
| 3    | `signal_combiner.py` + `test_signal_combiner.py` | —   | Combined signal logic + unit tests |
| 4    | `backtest_full.py`              | (printed report)            | Full historical backtest with stats |

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

> The original live/snapshot tools are unchanged: `sentiment.py` +
> `fetch_news.py` (CoinDesk RSS + VADER), `fetch_fundamentals.py`
> (30-day Blockchain.com snapshot), `detect_hammer.py`, `analyze.py`,
> and `backtest.py` (technical-only backtest).

## How each script works

### `fetch_btc_data.py`
Fetches daily BTC/USD OHLC. Uses CoinGecko for the last 90 days and Binance
klines for longer ranges (up to 730 days). Saves `btc_data.csv`
(columns: `date, open, high, low, close`).

### `fetch_fundamentals_history.py`  *(Step 1)*
Downloads two Blockchain.com **charts** series with `timespan=2years&format=json`:
* `n-unique-addresses` -> `unique_addresses`
* `n-transactions` -> `transactions`

Both are daily series merged by date (inner join on the dates that appear in
**both** series) and written to `fundamentals_history.csv`
(columns: `date, unique_addresses, transactions`). It prints the date range and
row count.

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

## Files

```
fetch_btc_data.py              BTC OHLC fetcher (CoinGecko/Binance)
fetch_fundamentals.py          30-day on-chain snapshot (live tool, unchanged)
fetch_fundamentals_history.py  2-year on-chain history (NEW, Step 1)
fetch_sentiment_history.py     Full Fear & Greed history (NEW, Step 2)
sentiment.py / fetch_news.py   VADER headline sentiment (live tool, unchanged)
detect_hammer.py               Hammer candlestick detector
analyze.py                     RSI(14) + SMA analysis
signal_combiner.py             Combined signal logic (NEW, Step 3)
test_signal_combiner.py        Unit tests for the combiner (NEW, Step 3)
backtest.py                    Technical-only (RSI + hammer) backtest (unchanged)
backtest_full.py               FULL combined-signal backtest (NEW, Step 4)
index.html                     Live dashboard / JS signal combiner
requirements.txt               Python dependencies (NEW, Step 5)
README.md                      This file
```

## Data artifacts

`.json` snapshot files (`fundamentals.json`, `sentiment.json`, `hammers.json`)
are produced by the live snapshot tools; the new `_history.csv` files feed the
historical backtest.


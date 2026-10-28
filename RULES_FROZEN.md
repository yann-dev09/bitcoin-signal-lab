# RULES_FROZEN - paper-trading rules (FROZEN, do NOT change)

Frozen on: **2026-10-28** (UTC)

> The parameters below are frozen for the duration of the paper-trading test.
> `paper_trade.py` enforces exactly these rules. Changing any of them would
> break the hold-out discipline of the stress test, so they are recorded here
> in writing and never edited.

## Evaluation note
**Evaluation date = first logged date + 90 days.**
If the first `paper_trade.py` run is on 2026-10-28, the first logged candle is the
last *closed* daily candle, 2026-10-27, and the evaluation date is **2027-01-25**.
See `paper_report.py` for the computed evaluation date (start + 90 days).

## Assets & cash
- 1 unit = 1 virtual USD of BTC exposure. Starting virtual capital = **$1000.00**
  per strategy, held 100% in BTC when long and 100% in cash when flat.
- The daily BTC close comes from `btc_data_full.csv` (Binance 1d klines, refreshed
  each run by reusing `fetch_btc_data.main_full`). Fear & Greed comes from
  `fng_history.csv` (refreshed by reusing `fetch_sentiment_history.main`).

## LIVE rule (acted on; written to column `position`) - MA100
Long BTC **next day** when `close[T] > MA(100, SMA)` at the close of day T;
otherwise cash (0%). `MA(100)` = rolling 100-day simple moving average of close.

This is the rule recommended by the Step-6 stress test (the trivial MA(100) long
rule dominated R2/R5 after costs and was the only rule that was not clearly
beaten). It is the only rule whose target is recorded as the live `position`.

## Counterfactual rules (tracked for comparison, NOT traded)
- **R2**: long next day when `close[T] > MA(90, SMA)` AND `7-day momentum > 0`,
  where `momentum = close[T]/close[T-7] - 1`; otherwise cash.
- **Buy & Hold**: 100% long at all times (the benchmark).

## Position / equity convention (frozen)
- `position` (logged) = next-day target for the live MA100 rule = `signal_ma100[T]`
  at the close of day T (1 = long / 0 = cash). The target is applied from day T+1.
- A daily candle for date D is treated as **closed** once day D+1 (UTC) begins, so
  only dates strictly before `today(UTC)` are ever logged.
- The position HELD during day T = the signal decided on day T-1 (`signal[T-1]`),
  i.e. next-day positioning with **no same-day lookahead** (matches
  `backtest_variants.py`'s `signal.shift(1)`).
- `paper_equity_*` = portfolio value at the **close** of day T, reflecting the
  position held during day T.
- R2 and Buy & Hold are computed with the same convention, purely as paper
  portfolios (no live trading).

## Transaction costs (frozen)
- **0.15% per side** on every entry AND every exit = 0.30% round-trip per complete
  round-turn. No fee while flat or while holding. No borrowing/lending interest.
  (0.1% fee + 0.05% slippage per side, per the stress-test specification.)

## What is frozen (must not change)
- MA lookbacks: 90 (R2) and 100 (MA100 live).
- Momentum window: 7 days.
- Signal thresholds: `close > MA` (strictly greater) and `momentum > 0` (strictly).
- Cost: 0.15% per side.
- Starting capital: $1000.00 per strategy.
- Position convention: next-day, `signal[T-1]` held during day T.
- Evaluation horizon: 90 days after the first logged date.
- Candle-close cutoff: only dates strictly before `today(UTC)` are logged.

## Sources (unchanged live tools)
- BTC candles: `fetch_btc_data.py` (Binance klines, 1d).
- Sentiment: `fetch_sentiment_history.py` (alternative.me FNG) - INFORMATIONAL
  only; it does NOT enter the MA100 or R2 signals.

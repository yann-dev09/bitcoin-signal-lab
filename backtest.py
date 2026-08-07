"""
Backtest of the technical-only trading signal over historical BTC data.

For each day in btc_data.csv:
  1. Calculate RSI(14) using Wilder's smoothing (same logic as analyze.py)
  2. Check for a hammer candlestick pattern (same logic as detect_hammer.py)
  3. Determine the technical-only signal for that day:
       Bullish  -> RSI < 30 (oversold) OR a hammer pattern is present
       Bearish  -> RSI > 70 (overbought)
       Neutral  -> otherwise
  4. Look at what the price did over the NEXT 3 days after each signal.

Prints a summary:
  - Count of Bullish / Bearish / Neutral signals
  - For Bullish signals: % followed by a price increase over next 3 days vs decrease
  - Baseline: % of ALL 3-day windows in the dataset that closed up
"""

import pandas as pd

from detect_hammer import is_hammer, read_ohlc

CSV_FILE = "btc_data.csv"
FORWARD_DAYS = 3
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def calculate_rsi(close, length=14):
    """Calculate RSI using Wilder's smoothing (same as analyze.py)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (equivalent to EMA with alpha=1/length)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def determine_signal(rsi, is_hammer_candle):
    """Determine the technical-only signal for a single day."""
    if rsi is None:
        return "Neutral"
    if rsi < RSI_OVERSOLD or is_hammer_candle:
        return "Bullish"
    if rsi > RSI_OVERBOUGHT:
        return "Bearish"
    return "Neutral"


def main():
    # Read OHLC data
    candles = read_ohlc(CSV_FILE)
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    print(f"Loaded {n} daily candles from {CSV_FILE}")
    print(f"Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} to "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print()

    # Calculate RSI(14) with Wilder's smoothing
    df["rsi"] = calculate_rsi(df["close"], length=14)

    # Determine hammer pattern for each candle using detect_hammer logic
    df["is_hammer"] = df.apply(lambda row: is_hammer(row.to_dict()), axis=1)

    # Determine the signal for each day
    df["signal"] = df.apply(
        lambda row: determine_signal(row["rsi"], row["is_hammer"]), axis=1
    )

    # Compute the 3-day forward return: close[N+3] / close[N] - 1
    df["close_fwd"] = df["close"].shift(-FORWARD_DAYS)
    df["fwd_return"] = df["close_fwd"] / df["close"] - 1
    df["fwd_up"] = df["fwd_return"] > 0
    df["fwd_down"] = df["fwd_return"] < 0

    # Only days that have a 3-day forward window are valid signals
    valid = df.dropna(subset=["fwd_return"]).copy()

    # ---------------- Summary ----------------
    counts = valid["signal"].value_counts()
    total_bullish = int(counts.get("Bullish", 0))
    total_bearish = int(counts.get("Bearish", 0))

    print("=" * 60)
    print("TOTAL BULLISH SIGNALS: " + str(total_bullish))
    print("TOTAL BEARISH SIGNALS: " + str(total_bearish))
    print("=" * 60)
    print()

    print("=" * 60)
    print("SIGNAL COUNTS")
    print("=" * 60)
    for s in ["Bullish", "Bearish", "Neutral"]:
        count = int(counts.get(s, 0))
        pct = (count / len(valid)) * 100 if len(valid) else 0
        print(f"  {s:<8}: {count:>4} signals  ({pct:5.1f}% of {len(valid)} valid days)")
    print()

    # ---------------- Bullish performance ----------------
    print("=" * 60)
    print("BULLISH SIGNAL PERFORMANCE (next 3 days)")
    print("=" * 60)
    bullish = valid[valid["signal"] == "Bullish"]
    bullish_up_pct = None
    bullish_down_pct = None
    bullish_flat_count = 0
    if len(bullish) > 0:
        bullish_up_count = int(bullish["fwd_up"].sum())
        bullish_down_count = int(bullish["fwd_down"].sum())
        bullish_flat_count = len(bullish) - bullish_up_count - bullish_down_count
        bullish_up_pct = (bullish_up_count / len(bullish)) * 100
        bullish_down_pct = (bullish_down_count / len(bullish)) * 100
        bullish_avg_return = bullish["fwd_return"].mean() * 100

        print(f"  Signals:          {len(bullish)}")
        print(f"  Price increased:  {bullish_up_count} ({bullish_up_pct:.1f}%)")
        print(f"  Price decreased:  {bullish_down_count} ({bullish_down_pct:.1f}%)")
        print(f"  Price unchanged:  {bullish_flat_count}")
        print(f"  Avg 3-day return: {bullish_avg_return:+.2f}%")
        print(f"  Worst 3-day:      {bullish['fwd_return'].min() * 100:+.2f}%")
        print(f"  Best 3-day:       {bullish['fwd_return'].max() * 100:+.2f}%")
    else:
        print("  No Bullish signals with a valid 3-day forward window.")
    print()

    # ---------------- Bearish performance (for context) ----------------
    print("=" * 60)
    print("BEARISH SIGNAL PERFORMANCE (next 3 days)")
    print("=" * 60)
    bearish = valid[valid["signal"] == "Bearish"]
    bearish_up_pct = None
    bearish_down_pct = None
    if len(bearish) > 0:
        bearish_up_count = int(bearish["fwd_up"].sum())
        bearish_down_count = int(bearish["fwd_down"].sum())
        bearish_up_pct = (bearish_up_count / len(bearish)) * 100
        bearish_down_pct = (bearish_down_count / len(bearish)) * 100
        bearish_avg_return = bearish["fwd_return"].mean() * 100

        print(f"  Signals:          {len(bearish)}")
        print(f"  Price increased:  {bearish_up_count} ({bearish_up_pct:.1f}%)")
        print(f"  Price decreased:  {bearish_down_count} ({bearish_down_pct:.1f}%)")
        print(f"  Avg 3-day return: {bearish_avg_return:+.2f}%")
        print(f"  Worst 3-day:      {bearish['fwd_return'].min() * 100:+.2f}%")
        print(f"  Best 3-day:       {bearish['fwd_return'].max() * 100:+.2f}%")
    else:
        print("  No Bearish signals with a valid 3-day forward window.")
    print()

    # ---------------- Baseline ----------------
    print("=" * 60)
    print("BASELINE — ALL 3-DAY WINDOWS IN THE DATASET")
    print("=" * 60)
    if len(valid) > 0:
        baseline_up = int(valid["fwd_up"].sum())
        baseline_down = int(valid["fwd_down"].sum())
        baseline_up_pct = (baseline_up / len(valid)) * 100
        baseline_avg = valid["fwd_return"].mean() * 100

        print(f"  Total 3-day windows:       {len(valid)}")
        print(f"  Windows price increased:  {baseline_up} ({baseline_up_pct:.1f}%)")
        print(f"  Windows price decreased:  {baseline_down} ({baseline_down / len(valid) * 100:.1f}%)")
        print(f"  Avg 3-day return:         {baseline_avg:+.2f}%")
    print()

    # ---------------- Comparison ----------------
    print("=" * 60)
    print("COMPARISON: SIGNALS vs BASELINE")
    print("=" * 60)

    # Bullish comparison
    print(f"  {'BULLISH SIGNALS':<22} {'UP-RATE':>10} {'DOWN-RATE':>10} {'EDGE vs BASE':>14}")
    if len(bullish) > 0 and len(valid) > 0 and bullish_up_pct is not None:
        edge = bullish_up_pct - baseline_up_pct
        edge_str = f"{edge:+.1f} pts"
        print(f"  {'Bullish':<22} {bullish_up_pct:>9.1f}% {bullish_down_pct:>9.1f}% {edge_str:>14}")
        if edge > 0:
            print("  => Bullish signals outperformed the baseline")
        elif edge < 0:
            print("  => Bullish signals UNDERPERFORMED the baseline")
        else:
            print("  => Bullish signals matched the baseline")
    else:
        print("  No Bullish signals with a valid 3-day forward window.")
    print()

    # Bearish comparison
    print(f"  {'BEARISH SIGNALS':<22} {'UP-RATE':>10} {'DOWN-RATE':>10} {'EDGE vs BASE':>14}")
    if len(bearish) > 0 and len(valid) > 0:
        bearish_down_count = int(bearish["fwd_down"].sum())
        bearish_down_pct = (bearish_down_count / len(bearish)) * 100
        bearish_up_count = int(bearish["fwd_up"].sum())
        bearish_up_pct = (bearish_up_count / len(bearish)) * 100

        # For bearish, the "edge" is: how much more often price DECREASED vs baseline decrease rate
        baseline_down_pct = (baseline_down / len(valid)) * 100
        edge_bearish = bearish_down_pct - baseline_down_pct
        edge_bearish_str = f"{edge_bearish:+.1f} pts"

        print(f"  {'Bearish':<22} {bearish_up_pct:>9.1f}% {bearish_down_pct:>9.1f}% {edge_bearish_str:>14}")
        print(f"  Baseline up-rate:  {baseline_up_pct:.1f}%  |  Baseline down-rate: {baseline_down_pct:.1f}%")
        if edge_bearish > 0:
            print("  => Bearish signals correctly preceded price DECREASES more often than baseline")
        elif edge_bearish < 0:
            print("  => Bearish signals did NOT predict decreases better than baseline")
        else:
            print("  => Bearish signals matched the baseline")
    else:
        print("  No Bearish signals with a valid 3-day forward window.")


if __name__ == "__main__":
    main()
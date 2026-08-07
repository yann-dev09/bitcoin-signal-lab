import csv
import json

CSV_FILE = "btc_data.csv"
OUTPUT_FILE = "hammers.json"


def read_ohlc(filename=CSV_FILE):
    """Read OHLC data from CSV into a list of dicts."""
    candles = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                {
                    "date": row["date"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
    return candles


def is_hammer(candle):
    """Detect a hammer candlestick pattern.

    Hammer criteria:
      - Small body (open/close close together) in the upper part of the range
      - Long lower wick (at least 2x the body size)
      - Little to no upper wick
    """
    open_price = candle["open"]
    high = candle["high"]
    low = candle["low"]
    close = candle["close"]

    body = abs(close - open_price)
    total_range = high - low

    if total_range == 0 or body == 0:
        return False

    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low

    # Small body: body is at most 1/3 of the total range
    small_body = body <= total_range / 3

    # Long lower wick: at least 2x the body size
    long_lower_wick = lower_wick >= 2 * body

    # Little to no upper wick: upper wick is at most 1/3 of the body
    little_upper_wick = upper_wick <= body / 3

    return small_body and long_lower_wick and little_upper_wick


def main():
    candles = read_ohlc()

    print(f"Scanning {len(candles)} daily candles for hammer patterns...")
    print()

    matches = [c for c in candles if is_hammer(c)]

    # Save results to hammers.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)

    if not matches:
        print("No hammer patterns found.")
        print(f"Results saved to {OUTPUT_FILE} (empty list).")
        return

    for candle in matches:
        print(
            f"{candle['date']}: "
            f"Open={candle['open']:.2f} High={candle['high']:.2f} "
            f"Low={candle['low']:.2f} Close={candle['close']:.2f}"
        )

    print(f"\n{len(matches)} hammer(s) saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
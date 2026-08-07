import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone

OHLC_API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
BINANCE_KLINES_API_URL = "https://api.binance.com/api/v3/klines"
OUTPUT_CSV = "btc_data.csv"
DEFAULT_DAYS = 30
MAX_DAYS = 730

# CoinGecko OHLC endpoint returns true daily candles only up to 90 days.
# Beyond that it switches to 4-day candles, so use Binance klines instead.
OHLC_DAILY_LIMIT = 90

# Binance klines returns up to 1000 candles per request.
BINANCE_MAX_LIMIT = 1000
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1d"


def fetch_json(url):
    """Fetch and parse JSON from a URL with a User-Agent header."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_btc_ohlc(days=30):
    """Fetch daily BTC/USD OHLC data.

    For days <= 90, uses the CoinGecko OHLC endpoint (true daily candles).
    For days > 90, uses the Binance public klines API which returns true
    daily candles in a single request (up to 1000 candles).
    """
    if days <= OHLC_DAILY_LIMIT:
        url = f"{OHLC_API_URL}?vs_currency=usd&days={days}"
        return fetch_json(url)

    return fetch_long_range_from_binance(days)


def fetch_long_range_from_binance(days):
    """Fetch daily BTC/USDT klines from Binance public API.

    Binance klines response format for each candle:
      [open_time, open, high, low, close, volume, close_time, ...]

    Returns a list of [timestamp_ms, open, high, low, close] candles.
    """
    start_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - days * 86400 * 1000
    url = (
        f"{BINANCE_KLINES_API_URL}"
        f"?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}"
        f"&startTime={start_ms}&limit={BINANCE_MAX_LIMIT}"
    )
    data = fetch_json(url)

    candles = []
    for kline in data:
        candles.append(
            [
                kline[0],                    # open time (ms)
                float(kline[1]),             # open
                float(kline[2]),             # high
                float(kline[3]),             # low
                float(kline[4]),             # close
            ]
        )

    return candles


def aggregate_daily(ohlc_data):
    """Aggregate OHLC candle list into a daily dict keyed by date."""
    daily = {}
    for candle in ohlc_data:
        timestamp_ms, open_price, high, low, close = candle
        date_str = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        if date_str not in daily:
            daily[date_str] = {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        else:
            day = daily[date_str]
            day["high"] = max(day["high"], high)
            day["low"] = min(day["low"], low)
            day["close"] = close  # last candle of the day

    return daily


def save_to_csv(ohlc_data, filename=OUTPUT_CSV):
    """Save daily OHLC data to a CSV file with date, open, high, low, close columns."""
    daily = aggregate_daily(ohlc_data)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close"])

        for date_str in sorted(daily):
            day = daily[date_str]
            writer.writerow(
                [
                    date_str,
                    f"{day['open']:.2f}",
                    f"{day['high']:.2f}",
                    f"{day['low']:.2f}",
                    f"{day['close']:.2f}",
                ]
            )


def main(days=DEFAULT_DAYS):
    source = "CoinGecko" if days <= OHLC_DAILY_LIMIT else "Binance"
    print(f"Fetching last {days} days of BTC/USD OHLC data from {source}...")
    try:
        ohlc_data = fetch_btc_ohlc(days=days)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if not ohlc_data:
        print("No OHLC data received.")
        return

    daily = aggregate_daily(ohlc_data)
    save_to_csv(ohlc_data)
    print(f"Saved {len(daily)} daily OHLC records to {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch BTC/USD daily OHLC data from CoinGecko and save to CSV."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days of history to fetch (default: {DEFAULT_DAYS}, max: {MAX_DAYS})",
    )
    args = parser.parse_args()

    if not 1 <= args.days <= MAX_DAYS:
        print(f"Error: --days must be between 1 and {MAX_DAYS}.")
        raise SystemExit(1)

    main(days=args.days)
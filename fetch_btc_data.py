import argparse
import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

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


FULL_HISTORY_START = "2018-02-01"
FULL_HISTORY_OUTPUT = "btc_data_full.csv"
DAY_MS = 86400 * 1000


def fetch_btc_history_paginated(start_date, end_date=None):
    """Fetch daily BTC/USDT klines from Binance from start_date onward.

    Binance returns at most 1000 candles per request, so this paginates using
    startTime: each request starts one day after the previous batch's last
    candle and we loop until a batch returns fewer than 1000 (the final
    partial batch) or startTime passes end_date.

    Args:
        start_date: 'YYYY-MM-DD' UTC string.
        end_date:   'YYYY-MM-DD' UTC string, or None to fetch up to the
                    latest available candle (no upper bound).

    Returns: list of [timestamp_ms, open, high, low, close] candles.
    """
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = None
    if end_date:
        end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_candles = []
    cur_start = start_ms
    while True:
        url = (
            f"{BINANCE_KLINES_API_URL}"
            f"?symbol={BINANCE_SYMBOL}&interval={BINANCE_INTERVAL}"
            f"&startTime={cur_start}&limit={BINANCE_MAX_LIMIT}"
        )
        data = fetch_json(url)
        if not data:
            break

        for kline in data:
            ts = int(kline[0])
            if ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            all_candles.append([ts, float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4])])

        if len(data) < BINANCE_MAX_LIMIT:
            # Reached the tail (final partial batch).
            break
        # Advance to one day after the last candle's open time.
        cur_start = int(data[-1][0]) + DAY_MS
        if end_ms is not None and cur_start > end_ms:
            break
        time.sleep(0.2)  # be gentle with the Binance rate limit
    return all_candles


def print_missing_days(daily):
    """Print calendar gaps in the aggregated daily dict (keys = 'YYYY-MM-DD')."""
    dates = sorted(daily.keys())
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    gaps = []
    for i in range(1, len(dts)):
        delta = (dts[i] - dts[i - 1]).days
        if delta > 1:
            for k in range(1, delta):
                gaps.append((dts[i - 1] + timedelta(days=k)).strftime("%Y-%m-%d"))
    if not gaps:
        print("  No missing days (consecutive daily series).")
    else:
        shown = gaps[:15]
        suffix = " ..." if len(gaps) > 15 else ""
        print(f"  Missing days ({len(gaps)}): {shown}{suffix}")


def main_full(start_date, end_date, output):
    """Fetch the full paginated daily history and save to a CSV."""
    end_label = end_date if end_date else "latest available"
    print(f"Fetching full daily BTC/USDT history from Binance "
          f"({start_date} .. {end_label}), paginating in 1000-candle chunks ...")
    candles = fetch_btc_history_paginated(start_date, end_date)

    if not candles:
        print("No OHLC data received.")
        return

    daily = aggregate_daily(candles)
    save_to_csv(candles, filename=output)

    print(f"Saved {len(daily)} daily OHLC records to {output}")
    print(f"Date range: {min(daily)} to {max(daily)}")
    print("  Missing days:")
    print_missing_days(daily)


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
        description=(
            "Fetch BTC/USD daily OHLC data and save to CSV. "
            "Default (no flags) fetches a short recent window to btc_data.csv; "
            "--full fetches the complete paginated history to btc_data_full.csv."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of days of history to fetch (default: {DEFAULT_DAYS}, max: {MAX_DAYS})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Fetch the FULL daily history (2018-02-01 .. latest) by paginating "
            "Binance in 1000-candle chunks and save to btc_data_full.csv. "
            "Ignores --days."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=FULL_HISTORY_START,
        help=f"Start date for --full in YYYY-MM-DD (default: {FULL_HISTORY_START})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for --full in YYYY-MM-DD (default: latest available)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for --full (default: btc_data_full.csv)",
    )
    args = parser.parse_args()

    if args.full:
        output = args.output or FULL_HISTORY_OUTPUT
        main_full(args.start, args.end, output)
    else:
        if not 1 <= args.days <= MAX_DAYS:
            print(f"Error: --days must be between 1 and {MAX_DAYS}.")
            raise SystemExit(1)
        main(days=args.days)
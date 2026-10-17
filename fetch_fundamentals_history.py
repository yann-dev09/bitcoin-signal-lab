"""Fetch 2 years of historical on-chain Bitcoin metrics from Blockchain.com.

Downloads two daily chart series over a 2-year timespan:
  * n-unique-addresses  -> "unique_addresses"
  * n-transactions        -> "transactions"

Both series are merged by date (inner join on dates present in BOTH series)
and saved to fundamentals_history.csv with columns:
  date, unique_addresses, transactions

This is the historical analogue of fetch_fundamentals.py (which only snapshots
the most recent 30 days).
"""

import csv
import json
import urllib.request
from datetime import datetime, timezone

OUTPUT_FILE = "fundamentals_history.csv"

# Blockchain.com "charts" API. timespan=2years returns daily data points.
TIMESPAN = "2years"
METRICS = {
    "unique_addresses": f"https://api.blockchain.info/charts/n-unique-addresses?timespan={TIMESPAN}&format=json",
    "transactions": f"https://api.blockchain.info/charts/n-transactions?timespan={TIMESPAN}&format=json",
}


def fetch_json(url):
    """Fetch and parse JSON from a URL with a User-Agent header."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def format_date(timestamp):
    """Convert a unix timestamp (seconds, UTC) to a YYYY-MM-DD date string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_series(url, label):
    """Fetch one Blockchain.com chart series and return {date_str: value}."""
    print(f"Fetching {label} ...")
    data = fetch_json(url)
    values = data.get("values", [])
    if not values:
        raise RuntimeError(f"No data received for {label}")
    series = {}
    for v in values:
        series[format_date(v["x"])] = v["y"]
    print(f"  -> {len(series)} daily data points ({min(series)} to {max(series)})")
    return series


def main():
    print(f"Fetching {TIMESPAN} of on-chain Bitcoin metrics from Blockchain.com...")
    print()

    series = {}
    for key, url in METRICS.items():
        series[key] = fetch_series(url, key)

    # Merge by date: keep only dates present in BOTH series (inner join).
    common_dates = sorted(set(series["unique_addresses"]) & set(series["transactions"]))
    if not common_dates:
        print("ERROR: no overlapping dates between the two series.")
        return

    rows = [
        (d, series["unique_addresses"][d], series["transactions"][d])
        for d in common_dates
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "unique_addresses", "transactions"])
        for date_str, addr, tx in rows:
            writer.writerow([date_str, f"{addr:.0f}", f"{tx:.0f}"])

    start_date = common_dates[0]
    end_date = common_dates[-1]
    print()
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Columns: date, unique_addresses, transactions")


if __name__ == "__main__":
    main()

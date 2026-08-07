import json
import urllib.request
from datetime import datetime, timezone

OUTPUT_FILE = "fundamentals.json"

METRICS = {
    "unique_addresses": {
        "name": "Total Unique Addresses",
        "url": "https://api.blockchain.info/charts/n-unique-addresses?timespan=30days&format=json",
    },
    "transactions_per_day": {
        "name": "Confirmed Transactions Per Day",
        "url": "https://api.blockchain.info/charts/n-transactions?timespan=30days&format=json",
    },
}


def fetch_metric(url):
    """Fetch a single metric series from Blockchain.com public API."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data


def format_date(timestamp):
    """Convert a unix timestamp to a UTC date string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


def pct_change(values):
    """Calculate percentage change between the first and last value in a list."""
    if len(values) < 2:
        return None
    first = values[0]
    last = values[-1]
    if first == 0:
        return None
    return ((last - first) / first) * 100


def main():
    print("Fetching on-chain Bitcoin metrics from Blockchain.com...")
    print()

    fundamentals = {}
    for key, meta in METRICS.items():
        try:
            data = fetch_metric(meta["url"])
        except Exception as e:
            print(f"Error fetching {meta['name']}: {e}")
            return

        values = data.get("values", [])
        if not values:
            print(f"No data received for {meta['name']}.")
            return

        # Build the series with formatted dates
        series = [
            {"date": format_date(v["x"]), "value": v["y"]}
            for v in values
        ]

        fundamentals[key] = {
            "name": meta["name"],
            "unit": data.get("unit", ""),
            "series": series,
        }

        # Print most recent value and 7-day percentage change
        recent = series[-1]
        print(f"{meta['name']}:")
        print(f"  Most recent ({recent['date']}): {recent['value']:,.0f} {data.get('unit', '')}".strip())

        # Last 7 days of data (or as many as available)
        last_7 = series[-7:] if len(series) >= 7 else series
        change = pct_change([v["value"] for v in last_7])
        if change is not None:
            print(f"  Change over last {len(last_7)} days: {change:+.2f}%")
        else:
            print("  Change over last 7 days: N/A")
        print()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(fundamentals, f, indent=2)

    print(f"Results saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
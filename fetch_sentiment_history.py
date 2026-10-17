"""Fetch the full Fear & Greed Index (FNG) history from alternative.me.

The CoinDesk RSS feed used by sentiment.py only returns the most recent
few headlines, so it cannot provide multi-year daily sentiment history.
The Fear & Greed Index is used here as a PROXY for daily market sentiment.

IMPORTANT (also stated in README.md): the Fear & Greed Index is an aggregate
market-sentiment indicator derived from volatility, market momentum, social
media, and Bitcoin dominance. It is NOT the same data as the VADER news-
sentiment scores produced by sentiment.py on CoinDesk headlines. It is a
proxy that gives a 0-100 value which we rescale to a -1..+1 score so it is
roughly comparable in magnitude to a VADER compound score.

API: https://api.alternative.me/fng/?limit=0&format=json
  - limit=0 returns the FULL available history.
  - response -> { "name", "data": [{"value":"55","value_classification":"Greed",
      "timestamp":"1792195200", ...}, ...], "metadata": {...} }
  - "value" is a string "0".."100"
  - "timestamp" is a unix epoch in seconds (string)

Saved to fng_history.csv with columns: date, value, sentiment_score
where sentiment_score = (value - 50) / 50  (maps 0..100 -> -1..+1).
"""

import csv
import json
import urllib.request
from datetime import datetime, timezone

OUTPUT_FILE = "fng_history.csv"
FNG_API_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_json(url):
    """Fetch and parse JSON from a URL with a User-Agent header."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def format_date(timestamp):
    """Convert a unix timestamp (seconds, UTC) to a YYYY-MM-DD date string."""
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")


def main():
    print("Fetching full Fear & Greed Index history from alternative.me ...")
    data = fetch_json(FNG_API_URL)

    points = data.get("data", [])
    if not points:
        print("ERROR: no data points received.")
        return

    rows = []
    for p in points:
        date_str = format_date(p["timestamp"])
        value = int(p["value"])
        sentiment_score = (value - 50) / 50.0
        rows.append((date_str, value, sentiment_score))

    # The API returns data most-recent-first; sort to ascending date order.
    rows.sort(key=lambda r: r[0])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value", "sentiment_score"])
        for date_str, value, score in rows:
            writer.writerow([date_str, value, f"{score:.6f}"])

    start_date = rows[0][0]
    end_date = rows[-1][0]
    print()
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Columns: date, value, sentiment_score")
    print("sentiment_score = (value - 50) / 50  (rescaled to -1..+1)")
    print()
    print("NOTE: This is a PROXY for news sentiment (Fear & Greed Index),")
    print("NOT the VADER compound scores from sentiment.py.")


if __name__ == "__main__":
    main()

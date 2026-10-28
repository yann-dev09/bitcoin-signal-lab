"""Fetch historical on-chain Bitcoin metrics from Blockchain.com.

Downloads two chart series over Blockchain.com's longest available timespan
(timespan=all, spanning 2009 -> today; granularity is ~4-day sampled across the
full span) and cuts the merged result to 2018-02-01 onward:
  * n-unique-addresses  -> "unique_addresses"
  * n-transactions        -> "transactions"

Both series are merged by the UNION of their dates, with each metric forward-filled
onto the union grid (a plain inner join would discard most samples because
Blockchain.com coarsens the two series onto slightly offset 4-day grids under
timespan=all). Saved to fundamentals_history.csv with columns:
  date, unique_addresses, transactions

Note on granularity: timespan=all returns ~4-day-spaced samples across the whole
range (Blockchain.com auto-coarsens very long spans). The downstream
data_utils.load_merged_dataset() forward-fills these points onto the daily BTC
index so the features become daily-valued. This coarser on-chain cadence is only
used by the Step 3 walk-forward model; the 5 rules in Step 2 use no on-chain data.

This is the historical analogue of fetch_fundamentals.py (which only snapshots
the most recent 30 days).
"""

import csv
import json
import urllib.request
from datetime import datetime, timedelta, timezone

OUTPUT_FILE = "fundamentals_history.csv"

# Cutoff so every data file shares the same analysis window (2018-02-01 -> today).
CUTOFF_DATE = "2018-02-01"

# Blockchain.com "charts" API. timespan=all returns the full available range
# (2009 -> today) at ~4-day sampling for long spans.
TIMESPAN = "all"
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


def print_missing_days(dates):
    """Print calendar gaps in a sorted list of 'YYYY-MM-DD' date strings."""
    dates = sorted(dates)
    if len(dates) < 2:
        print("  (single point, cannot compute gaps)")
        return
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    gaps = 0
    examples = []
    for i in range(1, len(dts)):
        delta = (dts[i] - dts[i - 1]).days
        if delta > 1:
            gaps += delta - 1
            if len(examples) < 5:
                examples.append(
                    f"{dts[i - 1].strftime('%Y-%m-%d')} -> {dts[i].strftime('%Y-%m-%d')} (gap={delta}d)"
                )
    if gaps == 0:
        print("  No missing days (consecutive daily series).")
    else:
        print(f"  {gaps} missing calendar days "
              "(coarse sampling; forward-filled to daily downstream).")
        print(f"  Examples: {examples}")


def merge_union_forward_fill(series):
    """Union-merge {metric: {date: value}} series with per-metric forward-fill.

    Returns (all_dates, ff) where all_dates is the sorted union of dates from
    every series and ff[metric][date] holds each metric's value forward-filled
    onto that union grid (so every union date has a value for every metric).

    This is used instead of an inner join because Blockchain.com's coarsened
    timespan=all grid is slightly offset between metrics, which would make an
    inner join discard the majority of samples.
    """
    all_dates = sorted(set().union(*(s.keys() for s in series.values())))
    ff = {}
    for metric, values in series.items():
        ff_for_metric = {}
        last = None
        for d in all_dates:
            if d in values:
                last = values[d]
            ff_for_metric[d] = last
        ff[metric] = ff_for_metric
    return all_dates, ff


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

    # Merge by union of dates with per-metric forward-fill.
    all_dates, ff = merge_union_forward_fill(series)
    if not all_dates:
        print("ERROR: no overlapping dates between the two series.")
        return

    # Cut to the shared analysis window (2018-02-01 -> today).
    rows = [
        (d, ff["unique_addresses"][d], ff["transactions"][d])
        for d in all_dates
        if d >= CUTOFF_DATE
    ]
    if not rows:
        print(f"ERROR: no on-chain dates on or after {CUTOFF_DATE}.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "unique_addresses", "transactions"])
        for date_str, addr, tx in rows:
            writer.writerow([date_str, f"{addr:.0f}", f"{tx:.0f}"])

    start_date = rows[0][0]
    end_date = rows[-1][0]
    print()
    print(f"Saved {len(rows)} rows to {OUTPUT_FILE}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Cutoff applied: {CUTOFF_DATE} onward")
    print("Columns: date, unique_addresses, transactions")
    print("  Missing days:")
    print_missing_days([r[0] for r in rows])


if __name__ == "__main__":
    main()

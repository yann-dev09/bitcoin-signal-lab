"""Verify the three historical data files: ranges, row counts, missing days.

Reports the date range, row count, and number of missing calendar days for
btc_data_full.csv, fng_history.csv, and fundamentals_history.csv so the
2018-02-01 -> latest analysis window can be sanity-checked in one place.
"""

import csv
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    ("BTC daily (btc_data_full.csv)", os.path.join(BASE_DIR, "btc_data_full.csv")),
    ("FNG daily (fng_history.csv)", os.path.join(BASE_DIR, "fng_history.csv")),
    ("On-chain (fundamentals_history.csv)", os.path.join(BASE_DIR, "fundamentals_history.csv")),
]

CUTOFF = "2018-02-01"


def load_dates(path):
    dates = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dates.append(row["date"])
    return sorted(dates)


def missing_days(dates):
    if len(dates) < 2:
        return 0, []
    dts = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    total = 0
    examples = []
    for i in range(1, len(dts)):
        gap = (dts[i] - dts[i - 1]).days - 1
        if gap > 0:
            total += gap
            if len(examples) < 5:
                examples.append(f"{dts[i-1].date()} -> {dts[i].date()} ({gap}d)")
    return total, examples


def report(label, path):
    print(f"--- {label} ---")
    if not os.path.exists(path):
        print("  FILE MISSING")
        print()
        return
    dates = load_dates(path)
    print(f"  rows: {len(dates)}")
    if not dates:
        print("  (empty)")
        print()
        return
    print(f"  range: {dates[0]} to {dates[-1]}")
    in_window = [d for d in dates if d >= CUTOFF]
    print(f"  rows >= {CUTOFF}: {len(in_window)}")
    if in_window:
        print(f"  in-window range: {in_window[0]} to {in_window[-1]}")
    gaps, examples = missing_days(in_window)
    if gaps == 0:
        print("  missing days in window: 0 (consecutive daily)")
    else:
        print(f"  missing days in window: {gaps} (coarse sampling; forward-filled downstream)")
        if examples:
            print(f"  examples: {examples}")
    print()


def main():
    print("=== Data verification ===\n")
    for label, path in FILES:
        report(label, path)
    print("Note: on-chain (fundamentals) is 4-day sampled by Blockchain.com and")
    print("is forward-filled to daily inside data_utils.load_merged_dataset().\n")


if __name__ == "__main__":
    main()

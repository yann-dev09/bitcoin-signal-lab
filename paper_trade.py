"""Daily forward paper-trader for Bitcoin (no exchange API, no real money).

Runs once a day (e.g. via Windows Task Scheduler, see README "Paper-trading"
section). Each run:

  1. refreshes btc_data_full.csv with the latest daily candles (reuses
     fetch_btc_data.main_full) and refreshes fng_history.csv (reuses
     fetch_sentiment_history.main) - best-effort: a network failure is warned
     and the run continues from the on-disk CSVs (never blocks logging).
  2. for every newly-closed daily candle NOT yet in paper_log.csv (skips dates
     already logged, never rewrites past rows), computes the MA100 & R2 signals,
     decides the next-day target position, and appends ONE row.

Columns: date, close, ma100, signal_ma100, signal_r2, position, fng, rsi_14,
         paper_equity_ma100, paper_equity_r2, paper_equity_buyhold,
         trades_ma100, trades_r2

Position / equity convention (see RULES_FROZEN.md):
  - `position` = next-day target for the LIVE MA100 rule = signal_ma100[date]
    (1 = long BTC / 0 = cash), decided at the close of `date`.
  - paper_equity_* = portfolio value at the CLOSE of `date`, reflecting the
    position HELD during `date` = the signal decided on `date-1`
    (no same-day lookahead; matches backtest_variants `signal.shift(1)`).
  - R2 and Buy & Hold are tracked as counterfactual paper portfolios.
  - A daily candle for date D is treated closed once D+1 (UTC) begins, so only
    dates strictly before "today" (UTC) are ever logged.

Cost model: 0.15% per side on every entry AND every exit (0.30% round-trip).
Starting virtual capital: $1000.00 per strategy.

Usage:
    python paper_trade.py            # refresh + log
    python paper_trade.py --dry-run  # preview, write nothing
"""

import argparse
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

import data_utils as du
import fetch_btc_data
import fetch_sentiment_history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_LOG = os.path.join(BASE_DIR, "paper_log.csv")
START_CAPITAL = 1000.0
COST_PER_SIDE = 0.0015  # 0.1% fee + 0.05% slippage, per side

LOG_COLUMNS = [
    "date", "close", "ma100", "signal_ma100", "signal_r2", "position",
    "fng", "rsi_14", "paper_equity_ma100", "paper_equity_r2",
    "paper_equity_buyhold", "trades_ma100", "trades_r2",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def today_utc():
    return datetime.now(timezone.utc).date()


def refresh_data():
    """Best-effort refresh of the on-disk CSVs (reuses existing fetch code)."""
    ok = True
    try:
        fetch_btc_data.main_full(
            fetch_btc_data.FULL_HISTORY_START, None,
            fetch_btc_data.FULL_HISTORY_OUTPUT,
        )
    except Exception as e:  # network / API failure -> keep on-disk data
        print(f"[warn] BTC refresh failed ({e}); keeping existing btc_data_full.csv")
        ok = False
    try:
        fetch_sentiment_history.main()
    except Exception as e:
        print(f"[warn] FNG refresh failed ({e}); keeping existing fng_history.csv")
        ok = False
    return ok


def load_dataframe():
    """Merged daily dataset (datetime index) + the signals we need."""
    df = du.load_merged_dataset()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    close = df["close"]
    df["ma100"] = close.rolling(100).mean()
    df["signal_ma100"] = (close > df["ma100"]).astype(int)
    df["signal_r2"] = ((close > df["ma_90"]) & (df["momentum_7"] > 0)).astype(int)
    # FNG 0-100 map for the informational `fng` column.
    fng_map = {}
    try:
        fng = pd.read_csv(os.path.join(BASE_DIR, "fng_history.csv"))
        fng_map = dict(zip(fng["date"].astype(str), fng["value"].astype(float)))
    except Exception:
        pass
    return df, fng_map


def existing_log():
    """Read paper_log.csv (strings, no rewrite of past rows). Empty DF if absent."""
    if not os.path.exists(PAPER_LOG):
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(PAPER_LOG, dtype={"date": str})
    if log.empty:
        return pd.DataFrame(columns=LOG_COLUMNS)
    return log.drop_duplicates(subset=["date"], keep="last").sort_values("date")


def _sig(sigcol, d_str, df):
    """Signal value (1/0) of `sigcol` at date d_str (missing/NaN -> 0)."""
    v = df.loc[d_str, sigcol] if d_str in df.index else np.nan
    return 0 if pd.isna(v) else int(v)


def run(dry_run=False, no_refresh=False):
    today = pd.Timestamp(today_utc())
    print(f"paper_trade run @ {datetime.now(timezone.utc).isoformat()}Z "
          f"| today(utc)={today.date()}")
    if not no_refresh:
        refresh_data()
    df, fng_map = load_dataframe()

    # Last CLOSED daily candle = max date strictly before today (UTC).
    closed = df.index.normalize()
    mask = closed < today
    if not mask.any():
        print("No closed candle available yet (today's candle is not closed).")
        return
    last_closed = closed[mask].max()

    log = existing_log()
    if log.empty:
        start = last_closed
        eq_ma100 = eq_r2 = eq_bh = START_CAPITAL
        tr_ma100 = tr_r2 = 0
        first_run = True
        logged = set()
    else:
        last_dt = pd.Timestamp(log["date"].iloc[-1])
        start = last_dt + pd.Timedelta(days=1)
        eq_ma100 = float(log["paper_equity_ma100"].iloc[-1])
        eq_r2 = float(log["paper_equity_r2"].iloc[-1])
        eq_bh = float(log["paper_equity_buyhold"].iloc[-1])
        tr_ma100 = int(log["trades_ma100"].iloc[-1])
        tr_r2 = int(log["trades_r2"].iloc[-1])
        first_run = False
        logged = set(log["date"].astype(str))

    if start > last_closed:
        print(f"Up to date: last closed candle {last_closed.date()} already logged.")
        return

    rng = pd.date_range(start=start, end=last_closed, freq="D")
    new_rows = []
    for d in rng:
        ds = d.strftime("%Y-%m-%d")
        if ds in logged:
            continue  # idempotent: never rewrite an already-logged date
        if ds not in df.index:
            continue
        ret = df.loc[ds, "returns_1d"]
        if pd.isna(ret):
            continue
        dp1 = (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        dp2 = (d - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        h_m100 = _sig("signal_ma100", dp1, df)   # held during day d = signal[d-1]
        h_r2 = _sig("signal_r2", dp1, df)
        if first_run and d == rng[0]:
            p_m100, p_r2 = 0, 0
        else:
            p_m100 = _sig("signal_ma100", dp2, df)   # held during d-1 = signal[d-2]
            p_r2 = _sig("signal_r2", dp2, df)

        def evolve(h, p, prev_eq):
            txn = COST_PER_SIDE if (h != p) else 0.0
            eq = prev_eq * (1.0 + (h * float(ret) - txn))
            entries = 1 if (p == 0 and h == 1) else 0
            return eq, entries

        eq_ma100, em = evolve(h_m100, p_m100, eq_ma100)
        eq_r2, er = evolve(h_r2, p_r2, eq_r2)
        eq_bh = eq_bh * (1.0 + float(ret))           # buy & hold: long, no cost
        tr_ma100 += em
        tr_r2 += er
        first_run = False
        fng_d = fng_map.get(ds, np.nan)
        new_rows.append({
            "date": ds,
            "close": round(float(df.loc[ds, "close"]), 2),
            "ma100": round(float(df.loc[ds, "ma100"]), 2) if not pd.isna(df.loc[ds, "ma100"]) else "",
            "signal_ma100": int(df.loc[ds, "signal_ma100"]),
            "signal_r2": int(df.loc[ds, "signal_r2"]),
            "position": int(df.loc[ds, "signal_ma100"]),  # next-day target (live rule)
            "fng": round(float(fng_d), 2) if not pd.isna(fng_d) else "",
            "rsi_14": round(float(df.loc[ds, "rsi_14"]), 2) if not pd.isna(df.loc[ds, "rsi_14"]) else "",
            "paper_equity_ma100": round(eq_ma100, 2),
            "paper_equity_r2": round(eq_r2, 2),
            "paper_equity_buyhold": round(eq_bh, 2),
            "trades_ma100": tr_ma100,
            "trades_r2": tr_r2,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
        out = (pd.concat([log, new_df], ignore_index=True)
               if not log.empty else new_df)
        out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        if dry_run:
            print(f"[dry-run] would append {len(new_rows)} row(s):\n")
            print(out.tail(len(new_rows)).to_string(index=False))
        else:
            out.to_csv(PAPER_LOG, index=False)
            print(f"Appended {len(new_rows)} row(s) to {PAPER_LOG} "
                  f"(last closed logged: {out['date'].iloc[-1]})")
    else:
        print("No new closed candles to log.")


def main():
    p = argparse.ArgumentParser(
        description="Daily BTC paper-trader: MA100 live rule + R2/BH counterfactuals.")
    p.add_argument("--dry-run", action="store_true",
                   help="preview what would be logged without writing paper_log.csv")
    p.add_argument("--no-refresh", action="store_true",
                   help="skip BTC/FNG refresh and use the on-disk CSVs")
    args = p.parse_args()
    run(dry_run=args.dry_run, no_refresh=args.no_refresh)


if __name__ == "__main__":
    main()



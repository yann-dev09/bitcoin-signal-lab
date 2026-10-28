"""Performance report for the paper-trading ledger (paper_log.csv).

Prints, per strategy (MA100 live, R2, Buy & Hold): ending equity, total return,
annualized return, Sharpe, max drawdown, and trade count - vs Buy & Hold.
Also prints the start / evaluation (start + 90 days) dates.

Usage:
    python paper_report.py
"""

import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_LOG = os.path.join(BASE_DIR, "paper_log.csv")
START_CAPITAL = 1000.0
TD = 365.0


def stats(equity, trades, first_date, last_date):
    """Equity series (list) -> stats dict. equity[-1] is last day's value."""
    e = np.asarray(equity, dtype=float)
    e = e[~np.isnan(e)]
    if len(e) == 0:
        return dict(end=START_CAPITAL, total=0.0, ann=0.0, sharpe=0.0,
                    mdd=0.0, trades=int(trades))
    curve = np.concatenate([[START_CAPITAL], e])      # include the start
    n_days = (last_date - first_date).days + 1
    end = float(e[-1])
    total = end / START_CAPITAL - 1.0
    ann = (1.0 + total) ** (TD / n_days) - 1.0 if total > -1 else float("nan")
    r = pd.Series(curve).pct_change().dropna().to_numpy()
    n_ret = len(r)
    if n_ret >= 2:
        mu, sig = r.mean(), r.std(ddof=1)
        sharpe = (mu / sig) * np.sqrt(TD) if sig > 0 else float("nan")
    else:
        sharpe = float("nan")
    if n_days >= 30:
        ann = (1.0 + total) ** (TD / n_days) - 1.0 if total > -1 else float("nan")
    else:
        ann = float("nan")  # annualization is not meaningful yet
    mdd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    return dict(end=end, total=total * 100, ann=ann * 100, sharpe=sharpe,
                mdd=mdd * 100, trades=int(trades), n_days=n_days, n_ret=n_ret)


def _f(x, nd=2, w=10):
    """Format a float, rendering NaN/None as right-aligned 'n/a'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a".rjust(w)
    return f"{x:.{nd}f}".rjust(w)


def main():
    if not os.path.exists(PAPER_LOG):
        print(f"No ledger yet: {PAPER_LOG} does not exist.")
        print("Run `python paper_trade.py` first.")
        return
    log = pd.read_csv(PAPER_LOG, dtype={"date": str}).sort_values("date")
    if log.empty:
        print("Ledger is empty. Run `python paper_trade.py` first.")
        return

    first_date = datetime.strptime(log["date"].iloc[0], "%Y-%m-%d").date()
    last_date = datetime.strptime(log["date"].iloc[-1], "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    eval_date = first_date + timedelta(days=90)

    rows = []
    for name, col, tcol in [
        ("MA100 live", "paper_equity_ma100", "trades_ma100"),
        ("R2", "paper_equity_r2", "trades_r2"),
        ("Buy & Hold", "paper_equity_buyhold", None),
    ]:
        tr = int(log[tcol].iloc[-1]) if tcol else 0
        rows.append(stats(log[col].tolist(), tr, first_date, last_date))

    bh = rows[2]
    print("=" * 78)
    print("PAPER-TRADING REPORT")
    print("=" * 78)
    print(f"Start date   : {first_date}")
    print(f"Last logged  : {last_date}  (rows: {len(log)})")
    print(f"Today (UTC)  : {today}")
    print(f"Evaluation   : {eval_date}  (= start + 90 days)")
    print(f"Days to eval : {(eval_date - today).days}  "
          f"(evaluation window: {eval_date - first_date} total)")
    print("-" * 78)
    hdr = f"{'strategy':<14}{'end $':>10}{'total %':>10}{'ann %':>10}" \
          f"{'sharpe':>9}{'maxDD %':>10}{'trades':>8}{'ret-vs-BH %':>13}"
    print(hdr)
    print("-" * 78)
    for name, s in zip(["MA100 live", "R2", "Buy & Hold"], rows):
        vs = _f(s['total'] - bh['total'], nd=2, w=12) + " %" if name != "Buy & Hold" else ""
        print(f"{name:<14}{_f(s['end'], nd=2):>10}{_f(s['total'], nd=2):>10}"
              f"{_f(s['ann'], nd=1):>10}{_f(s['sharpe'], nd=3):>9}"
              f"{_f(s['mdd'], nd=2):>10}{s['trades']:>8}{vs:>13}")
    print("-" * 78)
    if len(log) < 30:
        print(f"Note: only {len(log)} day(s) logged so far - annualized "
              "metrics / Sharpe are shown as n/a until ~30+ days.")
    print(f"Starting capital per strategy: ${START_CAPITAL:.2f}")
    print("Costs: 0.15% per side on entry/exit (0.30% round-trip). "
          "See RULES_FROZEN.md.")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""Full combined-signal backtest over historical BTC data.

Merges three data sources by date:
  * btc_data.csv               (date, open, high, low, close)         -- CoinGecko/Binance
  * fundamentals_history.csv   (date, unique_addresses, transactions) -- Blockchain.com
  * fng_history.csv            (date, value, sentiment_score)         -- alternative.me FNG

The BTC daily OHLC series is used as the master (consecutive daily) calendar.
On-chain metrics and the FNG sentiment score are reindexed onto that calendar
with forward-fill.  Forward-fill is NOT look-ahead: a missing day simply
inherits the most recent *past* observation, exactly as a trader would have
seen it.

For every day that has enough history (RSI available, >=7 days of on-chain
data, and a full forward window) the script:
  1. computes RSI(14) with Wilder's smoothing,
  2. detects a hammer via detect_hammer.is_hammer,
  3. computes the 7-day % change of unique addresses and transactions,
  4. calls signal_combiner.determine_signal(...) using ONLY same-day info,
  5. evaluates the 1-, 3- and 7-day forward return (close[t+N]/close[t]-1).

It then reports, per signal class (Bullish / Bearish / Cautious / Neutral)
and per horizon: number of signals, up-rate, average and median forward
return, the baseline (all days) up-rate, the edge vs baseline, and a
two-sided binomial-test p-value (H0: signal up-rate == baseline up-rate).

Finally it reports the same metrics for a train/test split (first 70% of
dates = train, last 30% = test).

The script is deliberately honest: it prints an explicit verdict when no
signal class shows a statistically significant edge (p < 0.05).
"""

import pandas as pd
from scipy.stats import binomtest

from detect_hammer import is_hammer
from signal_combiner import determine_signal

BTC_CSV = "btc_data.csv"
FUND_CSV = "fundamentals_history.csv"
FNG_CSV = "fng_history.csv"

HORIZONS = [1, 3, 7]
SIGNAL_ORDER = ["Bullish", "Bearish", "Cautious", "Neutral"]
ALPHA = 0.05  # significance level

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


def calculate_rsi(close, length=14):
    """RSI using Wilder's smoothing (alpha = 1/length).  Matches analyze.py."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_dataset():
    """Load and merge the three CSVs into one consecutive-daily dataframe."""
    btc = pd.read_csv(BTC_CSV, parse_dates=["date"])
    btc = btc.sort_values("date").reset_index(drop=True)
    btc = btc.set_index("date")

    fund = pd.read_csv(FUND_CSV, parse_dates=["date"]).set_index("date")
    fng = pd.read_csv(FNG_CSV, parse_dates=["date"]).set_index("date")

    # Reindex on-chain + sentiment onto the BTC daily calendar, forward-filling.
    # No look-ahead: only past values are carried forward.
    fund_on_btc = fund.reindex(btc.index).ffill()
    fng_on_btc = fng.reindex(btc.index).ffill()

    df = btc.copy()
    df["unique_addresses"] = fund_on_btc["unique_addresses"]
    df["transactions"] = fund_on_btc["transactions"]
    df["sentiment_score"] = fng_on_btc["sentiment_score"]
    df = df.reset_index()  # date back to a column

    # --- indicators (same-day / past only) ---
    df["rsi"] = calculate_rsi(df["close"], 14)
    df["is_hammer"] = df.apply(lambda r: is_hammer(r.to_dict()), axis=1)

    # 7-day % change (calendar-correct because the merged frame is consecutive daily)
    df["addr_change_7d"] = df["unique_addresses"].pct_change(periods=7) * 100.0
    df["tx_change_7d"] = df["transactions"].pct_change(periods=7) * 100.0
    df["sentiment"] = df["sentiment_score"]

    # --- combined signal (same-day inputs only) ---
    def _signal_for(row):
        rsi = None if pd.isna(row["rsi"]) else float(row["rsi"])
        addr = None if pd.isna(row["addr_change_7d"]) else float(row["addr_change_7d"])
        tx = None if pd.isna(row["tx_change_7d"]) else float(row["tx_change_7d"])
        sent = None if pd.isna(row["sentiment"]) else float(row["sentiment"])
        return determine_signal(rsi, bool(row["is_hammer"]), sent, addr, tx)["signal"]

    df["signal"] = df.apply(_signal_for, axis=1)

    # --- forward returns (the OUTCOME, not an input) ---
    for n in HORIZONS:
        fwd = df["close"].shift(-n)
        df[f"fwd_return_{n}"] = fwd / df["close"] - 1.0
        df[f"fwd_up_{n}"] = df[f"fwd_return_{n}"] > 0

    return df


def eval_window(df, horizon):
    """Return per-signal + baseline stats for a given forward horizon."""
    ret_col = f"fwd_return_{horizon}"
    up_col = f"fwd_up_{horizon}"
    valid = df.dropna(subset=[ret_col]).copy()

    n_all = len(valid)
    up_all = int(valid[up_col].sum())
    base_up_rate = up_all / n_all if n_all else float("nan")
    base_avg = valid[ret_col].mean() * 100 if n_all else float("nan")
    base_med = valid[ret_col].median() * 100 if n_all else float("nan")

    rows = []
    rows.append({
        "signal": "Baseline (all)",
        "n": n_all,
        "up_rate": base_up_rate,
        "up_count": up_all,
        "avg": base_avg,
        "median": base_med,
        "edge": 0.0,
        "p_value": None,
    })

    for sig in SIGNAL_ORDER:
        sub = valid[valid["signal"] == sig]
        n = len(sub)
        if n == 0:
            rows.append({"signal": sig, "n": 0, "up_rate": None, "up_count": 0,
                         "avg": None, "median": None, "edge": None, "p_value": None})
            continue
        up = int(sub[up_col].sum())
        up_rate = up / n
        avg = sub[ret_col].mean() * 100
        med = sub[ret_col].median() * 100
        edge = (up_rate - base_up_rate) * 100  # percentage points

        # Two-sided binomial test: is the signal up-rate different from baseline?
        p_value = None
        if 0 < base_up_rate < 1:
            try:
                p_value = binomtest(up, n, p=base_up_rate,
                                    alternative="two-sided").pvalue
            except Exception:
                p_value = None

        rows.append({
            "signal": sig, "n": n, "up_rate": up_rate, "up_count": up,
            "avg": avg, "median": med, "edge": edge, "p_value": p_value,
        })
    return rows, valid


def fmt_pct_or_none(x):
    """Format a 0..1 proportion as a percentage, or a placeholder."""
    return "  --- " if x is None else f"{x * 100:5.1f}%"


def print_section(title, rows, baseline_up_rate, horizon):
    print("=" * 72)
    print(f"{title}  (forward window = {horizon} day{'s' if horizon > 1 else ''})")
    print(f"baseline up-rate = {baseline_up_rate * 100:.1f}%")
    print("=" * 72)
    print(f"  {'signal':<16} {'n':>5} {'up-rate':>8} {'avg':>9} {'median':>9} "
          f"{'edge':>9} {'p-value':>10} {'sig':>4}")
    for r in rows:
        if r["signal"] == "Baseline (all)":
            upr = f"{r['up_rate'] * 100:5.1f}%"
        else:
            upr = fmt_pct_or_none(r["up_rate"])
        avg_s = "  --- " if r["avg"] is None else f"{r['avg']:+8.2f}%"
        med_s = "  --- " if r["median"] is None else f"{r['median']:+8.2f}%"
        edge_s = "  --- " if r["edge"] is None else f"{r['edge']:+8.1f}pts"
        p_s = "    ---" if r["p_value"] is None else f"{r['p_value']:10.4f}"
        sig_s = ""
        if r["signal"] != "Baseline (all)" and r["p_value"] is not None:
            sig_s = " *" if r["p_value"] < ALPHA else "  "
        print(f"  {r['signal']:<16} {r['n']:>5} {upr:>8} {avg_s:>9} {med_s:>9} "
              f"{edge_s:>9} {p_s:>10} {sig_s:>4}")
    print("  (* = significant at p < 0.05)")
    print()


def print_train_test(df):
    print("=" * 72)
    print("TRAIN / TEST SPLIT  (first 70% of dates = train, last 30% = test)")
    print("=" * 72)
    split_at = int(len(df) * 0.70)
    train = df.iloc[:split_at].copy()
    test = df.iloc[split_at:].copy()
    print(f"  Train: {train['date'].iloc[0].strftime('%Y-%m-%d')} "
          f".. {train['date'].iloc[-1].strftime('%Y-%m-%d')}  ({len(train)} days)")
    print(f"  Test : {test['date'].iloc[0].strftime('%Y-%m-%d')} "
          f".. {test['date'].iloc[-1].strftime('%Y-%m-%d')}  ({len(test)} days)")
    print()

    for horizon in HORIZONS:
        print(f"\n--- HORIZON: {horizon} day{'s' if horizon > 1 else ''} ---")
        tr_rows, _ = eval_window(train, horizon)
        te_rows, _ = eval_window(test, horizon)

        print(f"  {'signal':<16} {'TR n':>5} {'TR up%':>8} {'TR p':>9}   "
              f"{'TE n':>5} {'TE up%':>8} {'TE p':>9}   {'persist?':>9}")
        tr_map = {r["signal"]: r for r in tr_rows}
        te_map = {r["signal"]: r for r in te_rows}
        base_tr = tr_map.get("Baseline (all)")
        base_te = te_map.get("Baseline (all)")
        for sig in SIGNAL_ORDER:
            tr = tr_map.get(sig)
            te = te_map.get(sig)
            tr_up = "  --- " if (not tr or tr["up_rate"] is None) else f"{tr['up_rate']*100:5.1f}%"
            te_up = "  --- " if (not te or te["up_rate"] is None) else f"{te['up_rate']*100:5.1f}%"
            tr_p = "   ---" if (not tr or tr["p_value"] is None) else f"{tr['p_value']:7.4f}"
            te_p = "   ---" if (not te or te["p_value"] is None) else f"{te['p_value']:7.4f}"
            tr_n = tr['n'] if tr else 0
            te_n = te['n'] if te else 0
            persist = ""
            if tr and te and tr["up_rate"] is not None and te["up_rate"] is not None:
                tr_edge = tr["up_rate"] - (base_tr["up_rate"] if base_tr else 0.0)
                te_edge = te["up_rate"] - (base_te["up_rate"] if base_te else 0.0)
                persist = "same dir" if (tr_edge * te_edge > 0) else "diff dir"
            print(f"  {sig:<16} {tr_n:>5} {tr_up:>8} {tr_p:>9}   "
                  f"{te_n:>5} {te_up:>8} {te_p:>9}   {persist:>9}")
        base_up_tr = base_tr["up_rate"] * 100 if base_tr else 0
        base_up_te = base_te["up_rate"] * 100 if base_te else 0
        print(f"  {'Baseline':<16} {base_tr['n'] if base_tr else 0:>5} "
              f"{base_up_tr:>7.1f}% {'':>9}   "
              f"{base_te['n'] if base_te else 0:>5} {base_up_te:>7.1f}% {'':>9}   ")
        print("  (persist = whether train edge direction matches the test period)")
    print()


def main():
    df = build_dataset()
    print(f"\nMerged dataset: {len(df)} daily rows")
    print(f"Date range: {df['date'].iloc[0].strftime('%Y-%m-%d')} "
          f"to {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print("Columns: date, open, high, low, close, unique_addresses, "
          "transactions, sentiment_score, rsi, is_hammer, addr_change_7d, "
          "tx_change_7d, sentiment, signal, fwd_return_*, fwd_up_*")
    print()

    # Full-sample signal counts (at 7-day horizon, the richest signal set)
    valid7 = df.dropna(subset=["fwd_return_7"])
    counts = valid7["signal"].value_counts()
    print("Signal counts (days with a valid 7-day forward window):")
    total = len(valid7)
    for sig in SIGNAL_ORDER:
        c = int(counts.get(sig, 0))
        pct = (c / total * 100) if total else 0.0
        print(f"  {sig:<16}: {c:>4}  ({pct:5.1f}%)")
    print(f"  Total valid days: {total}")
    print()

    # On-chain availability note
    onchain_days = int(df["addr_change_7d"].notna().sum())
    print(f"Days with on-chain 7-day change available: {onchain_days} of {len(df)}")
    print()

    # Per-horizon full-sample report
    for horizon in HORIZONS:
        rows, valid = eval_window(df, horizon)
        base = next((r for r in rows if r["signal"] == "Baseline (all)"), None)
        base_up = base["up_rate"] if base else float("nan")
        print_section(f"FULL SAMPLE - {horizon}-day forward", rows, base_up, horizon)

    # Train / test split
    print_train_test(df)

    # ---- Honest verdict ----
    print("=" * 72)
    print("VERDICT: STATISTICAL SIGNIFICANCE")
    print("=" * 72)
    any_sig = False
    for horizon in HORIZONS:
        rows, _ = eval_window(df, horizon)
        for r in rows:
            if r["signal"] != "Baseline (all)" and r["p_value"] is not None:
                if r["p_value"] < ALPHA:
                    any_sig = True
                    print(f"  SIGNIFICANT: {r['signal']} {horizon}-day up-rate "
                          f"{r['up_rate']*100:.1f}% vs baseline "
                          f"(+{r['edge']:+.1f}pts, p={r['p_value']:.4f})")
    if not any_sig:
        print("  No signal class reached statistical significance (p >= 0.05)")
        print("  at any horizon. The apparent edge is NOT statistically significant.")
        print("  Treat any observed differences as due to chance / small-sample noise.")
    print()
    print("(p-values are two-sided binomial tests of each signal up-rate")
    print(" against the baseline up-rate; H0: signal up-rate == baseline up-rate)")


if __name__ == "__main__":
    main()





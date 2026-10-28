"""Step 2: five rule-based trading strategies on the BTC dataset.

Reuses data_utils.load_merged_dataset() as the single source of truth.
Each rule emits a daily 0/1 "long" signal from information available at the
close of day t; the position is taken the NEXT day (signal[t] -> position[t+1])
so the strategy earns returns_1d[t+1] with no same-day lookahead.

Default run evaluates on the ANALYSIS window (first 80% of days).
--holdout evaluates ONLY on the hold-out window (last 20%) for the single
final forward report (do not use until the final validation step).
"""

import argparse

import numpy as np
import pandas as pd

import data_utils as du

TRADING_DAYS = 365.0


def compute_signals(df):
    """Return a DataFrame of 0/1 long-signals (cols = rule names), indexed like df.

    Every condition uses only data known at the close of day t:
      R1: 30-day momentum > 2.5%  AND  30-day volatility < 50%
      R2: 7-day momentum > 0    AND  price above 90-day MA
      R3: 30-day momentum > 5%  AND  14-day RSI < 70 (not overbought)
      R4: 7-day on-chain change > 0 AND  Fear&Greed sentiment > 0
      R5: price above 30-day MA AND  7-day momentum > 0 AND  sentiment > 0
    """
    c = df["close"]
    s = pd.DataFrame(index=df.index)
    s["R1: Mom30>2.5% & VOL30<50%"] = ((df["momentum_30"] > 0.025) & (df["vol_30"] < 0.50)).astype(int)
    s["R2: Mom7>0 & price>MA90"] = ((df["momentum_7"] > 0) & (c > df["ma_90"])).astype(int)
    s["R3: Mom30>5% & RSI14<70"] = ((df["momentum_30"] > 0.05) & (df["rsi_14"] < 70)).astype(int)
    s["R4: Onchain>0 & sentiment>0"] = ((df["addr_7d"] > 0) & (df["sentiment_score"] > 0)).astype(int)
    s["R5: price>MA30 & Mom7>0 & sent>0"] = ((c > df["ma_30"]) & (df["momentum_7"] > 0) & (df["sentiment_score"] > 0)).astype(int)
    return s


def portfolio_metrics(strat_ret, label, long_days=None):
    """Total/annualized return, Sharpe, win rate of a daily return stream."""
    ret = np.asarray(strat_ret, dtype=float)
    ret = ret[~np.isnan(ret)]
    n = len(ret)
    out = {"rule": label, "days": n}
    if n == 0:
        out.update({"total_return_%": 0.0, "annual_return_%": 0.0, "sharpe": 0.0, "win_rate_%": 0.0, "long_days": 0})
        return out
    total = np.prod(1 + ret) - 1.0
    ann = (1 + total) ** (TRADING_DAYS / n) - 1.0 if total > -1 else float("nan")
    mu, sigma = ret.mean(), ret.std(ddof=1)
    sharpe = (mu / sigma) * np.sqrt(TRADING_DAYS) if sigma > 0 else float("nan")
    out.update({
        "total_return_%": round(total * 100, 2),
        "annual_return_%": round(ann * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate_%": round((ret > 0).mean() * 100, 1),
        "long_days": int(long_days) if long_days is not None else n,
    })
    return out


def signal_quality(df, signal):
    """7-day-forward-return quality of a 0/1 signal (on days it fires)."""
    fr = df["future_return_7d"]
    tgt = df["target_7d"]
    fired = (signal == 1)
    n_sig = int(fired.sum())
    if n_sig == 0:
        return {"n_signals": 0, "avg_7d_forward_return_%": 0.0, "win_rate_7d_%": 0.0, "precision_long_%": 0.0}
    fr7 = fr[fired].dropna()
    precision = tgt[fired].mean()  # P(next-7d up | long)
    return {
        "n_signals": n_sig,
        "avg_7d_forward_return_%": round(fr7.mean() * 100, 2) if len(fr7) else 0.0,
        "win_rate_7d_%": round((fr7 > 0).mean() * 100, 1) if len(fr7) else 0.0,
        "precision_long_%": round(precision * 100, 1),
    }


def evaluate(df):
    """Compute signal matrix, next-day positions, and per-rule stats."""
    signals = compute_signals(df)
    # position[t] = signal[t-1]  -> earns returns_1d[t], no same-day lookahead
    pos = signals.shift(1).fillna(0).astype(int)
    strat = pos.mul(df["returns_1d"], axis=0)   # align on rows, not columns
    strat = strat.dropna()

    rows = []
    # Baseline: buy & hold (long every day).
    base = portfolio_metrics(df["returns_1d"].to_numpy(), "Buy & Hold", long_days=len(df))
    rows.append(base)
    for col in signals.columns:
        r = portfolio_metrics(strat[col].to_numpy(), col, long_days=int(pos[col].sum()))
        r.update(signal_quality(df, signals[col]))
        rows.append(r)

    return rows, signals, strat


def print_table(rows):
    cols = ["rule", "days", "long_days", "n_signals", "total_return_%",
            "annual_return_%", "sharpe", "win_rate_%", "win_rate_7d_%",
            "avg_7d_forward_return_%", "precision_long_%"]
    out = pd.DataFrame(rows)[cols]
    for c in cols:
        if c != "rule":
            out[c] = out[c].apply(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Step 2: rule-based strategies on the BTC dataset.")
    parser.add_argument("--holdout", action="store_true",
                        help="Evaluate ONLY on the hold-out window (final fifth). Final-step use only.")
    args = parser.parse_args()

    df = du.load_merged_dataset()
    need = du.FEATURE_COLUMNS + ["returns_1d", "future_return_7d", "target_7d"]
    df = df.dropna(subset=need)

    analysis, holdout = du.split_dataset(df)
    if args.holdout:
        window = holdout
        tag = "HOLD-OUT window (last 20%%: %s -> %s)" % (window.index[0], window.index[-1])
    else:
        window = analysis
        tag = "ANALYSIS window (first 80%%: %s -> %s)" % (window.index[0], window.index[-1])

    print("=== Step 2: rule-based variants ===")
    print(tag)
    print(f"Rules retrained/evaluated on {len(window)} days")
    print()
    rows, signals, strat = evaluate(window)
    print_table(rows)
    print()
    print("Note: long signals fire on day t, position taken day t+1 (no lookahead).")


if __name__ == "__main__":
    main()

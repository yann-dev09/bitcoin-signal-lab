"""Shared dataset loader + feature engineering for the BTC prediction pipeline
(used by backtest_variants.py and model_walkforward.py).

Loads the three historical data files into a single daily, forward-filled
DataFrame spanning 2018-02-01 .. latest:
    * btc_data_full.csv        (daily BTC OHLC, from fetch_btc_data.py --full)
    * fng_history.csv          (daily Fear&Greed sentiment, fetch_sentiment_history.py)
    * fundamentals_history.csv (4-day on-chain; forward-filled to daily by this module,
                                produced by fetch_fundamentals_history.py with timespan=all)

Conventions:
  * All features at day t use only information available at the close of day t
    (no future leakage into the features).
  * The prediction TARGET is the sign of the 7-day-forward return
    (close[t+7] / close[t] - 1 > 0  -> 1, else 0).
  * By default the dataset is split by date: the first ANALYSIS_SPLIT of days
    are the "analysis" window (tuning / walk-forward reporting) and the last 20%
    is the "hold-out" window (single final forward report).
"""

import csv
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ANALYSIS_CSV = os.path.join(BASE_DIR, "btc_data_full.csv")
FNG_CSV = os.path.join(BASE_DIR, "fng_history.csv")
FUNDAMENTALS_CSV = os.path.join(BASE_DIR, "fundamentals_history.csv")

CUTOFF_DATE = "2018-02-01"   # shared analysis-window start
HORIZON = 7                  # 7-day forward-return prediction horizon
ANALYSIS_SPLIT = 0.80        # first 80% of days = analysis; last 20% = hold-out

# Canonical feature set used by the logistic-regression model.
FEATURE_COLUMNS = [
    "returns_1d",
    "ma_ratio_7_30",
    "ma_ratio_7_90",
    "ma_ratio_30_90",
    "rsi_14",
    "vol_30",
    "momentum_7",
    "momentum_30",
    "sentiment_score",
    "addr_7d",
    "tx_7d",
]


# --------------------------------------------------------------------------- #
# Low-level CSV loaders
# --------------------------------------------------------------------------- #
def _load_btc():
    rows = []
    with open(ANALYSIS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "date": r["date"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("date").sort_index()
    return df[df.index >= CUTOFF_DATE]


def _load_fng():
    rows = []
    with open(FNG_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["date"], float(r["sentiment_score"])))
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in rows})


def _load_fundamentals():
    rows = []
    with open(FUNDAMENTALS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["date"],
                         int(float(r["unique_addresses"])),
                         int(float(r["transactions"]))))
    if not rows:
        return pd.DataFrame(columns=["unique_addresses", "transactions"])
    df = pd.DataFrame(rows, columns=["date", "unique_addresses", "transactions"])
    return df.set_index("date").sort_index()


def _forward_fill_onchain(series, full_index):
    """Reindex an on-chain series to the daily BTC index and forward-fill it
    (early days before the first on-chain sample are back-filled with that
    first sample so the 7-day on-chain change has no leading NaNs)."""
    s = series.reindex(full_index)
    s = s.ffill()
    if s.isna().any():
        first_valid = series.dropna()
        if not first_valid.empty:
            s = s.fillna(first_valid.iloc[0])
    return s


def _rsi(series, window=14):
    """Wilder RSI over a close series (fully known at the close of day t)."""
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_dataset():
    """Build the merged daily DataFrame with features + the 7-day target.

    Returns a DataFrame indexed by date (string) with columns:
    open, high, low, close, returns_1d, ma_7, ma_30, ma_90, rsi_14, vol_30,
    momentum_7, momentum_30, sentiment_score, unique_addresses, transactions,
    addr_7d, tx_7d, future_return_7d, target_7d.
    """
    btc = _load_btc()
    if btc.empty:
        raise RuntimeError(
            "btc_data_full.csv is empty or missing; run 'py fetch_btc_data.py --full'."
        )
    full_index = btc.index

    fng = _load_fng().reindex(full_index).ffill()
    fund = _load_fundamentals()

    addr = (_forward_fill_onchain(fund["unique_addresses"], full_index)
            if "unique_addresses" in fund.columns else pd.Series(np.nan, index=full_index))
    tx = (_forward_fill_onchain(fund["transactions"], full_index)
          if "transactions" in fund.columns else pd.Series(np.nan, index=full_index))

    df = btc.copy()
    df["sentiment_score"] = fng
    df["unique_addresses"] = addr
    df["transactions"] = tx

    close = df["close"]
    df["returns_1d"] = close.pct_change()
    df["ma_7"] = close.rolling(7).mean()
    df["ma_30"] = close.rolling(30).mean()
    df["ma_90"] = close.rolling(90).mean()
    # Stationary moving-average ratios (short vs longer MAs) instead of raw
    # price levels, which are non-stationary and otherwise dominate the fit.
    df["ma_ratio_7_30"] = df["ma_7"] / df["ma_30"]
    df["ma_ratio_7_90"] = df["ma_7"] / df["ma_90"]
    df["ma_ratio_30_90"] = df["ma_30"] / df["ma_90"]
    df["rsi_14"] = _rsi(close, 14)
    df["vol_30"] = df["returns_1d"].rolling(30).std() * np.sqrt(365)
    df["momentum_7"] = close.pct_change(7)
    df["momentum_30"] = close.pct_change(30)
    df["addr"] = addr
    df["addr_7d"] = addr.pct_change(7)
    df["tx_7d"] = tx.pct_change(7)

    # 7-day forward return + binary target (the quantity the model predicts).
    df["future_close_7d"] = close.shift(-HORIZON)
    df["future_return_7d"] = df["future_close_7d"] / close - 1.0
    df["target_7d"] = (df["future_return_7d"] > 0).astype(float)

    return df


def load_merged_dataset():
    """Public entry point: the merged daily DataFrame (features + target)."""
    return build_dataset()


def split_dataset(df, split=ANALYSIS_SPLIT):
    """Split by date: first `split` fraction of rows = analysis, rest = hold-out."""
    n = len(df)
    cut = int(n * split)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def feature_matrix(df):
    """Return (X, y) with X = feature columns (float, no NaNs), y = target_7d.

    Rows with any NaN in features or target are dropped (lookback warmup at the
    start of the series and the final 7 days which have no forward return).
    """
    cols = FEATURE_COLUMNS + ["target_7d"]
    valid = df.dropna(subset=cols)
    X = valid[FEATURE_COLUMNS].astype(float)
    y = valid["target_7d"].astype(float)
    return X, y


def strategy_from_position(df, position):
    """Map a 0/1 daily `position` array (aligned to df) to strategy daily returns.

    strat_ret[t] = position[t] * returns_1d[t].
    """
    pos = np.asarray(position, dtype=float)
    ret = df["returns_1d"].fillna(0.0).to_numpy()
    return pos * ret

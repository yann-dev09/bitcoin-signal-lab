"""Step 3: expanding-window logistic-regression walk-forward model.

Predicts the sign of the next 7-day BTC return (data_utils.target_7d) from
price, momentum, volatility, Fear&Greed sentiment, and on-chain features.

Walk-forward protocol (expanding window, no lookahead):
  * The model is re-trained every RETRAIN_DAYS on ALL data seen up to day t-1.
  * Between retrains the same model is used to score each subsequent day.
  * A long-only position is taken when the predicted probability of "up"
    exceeds a threshold.

Default run evaluates on the ANALYSIS window (first 80% of days): it reports
classification metrics (accuracy, precision, recall, F1, AUC), picks the
probability threshold that maximizes F1 on the analysis predictions, and shows
the long-only strategy metrics.

--holdout evaluates the SAME pipeline on the hold-out window (last 20%), scoring
each hold-out day with a model trained on the analysis window + all earlier
hold-out days (each prediction uses only strictly-prior data). The threshold
reused there is the one chosen on the analysis window. Use --holdout only for
the single final validation run.
"""

import argparse
import os

import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data_utils as du

# Fitting is done inside an expanding walk-forward loop; suppress the lbfgs
# iteration warnings that can arise from borderline-convergent refits.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRADING_DAYS = 365.0
HORIZON = du.HORIZON          # 7-day prediction horizon
RETRAIN_DAYS = 30             # re-fit the model every 30 days (expanding window)
WARMUP_DAYS = 90             # skip the first days of a no-pretrain window
PRED_CSV = "model_walkforward_predictions.csv"
CHART_PNG = "walkforward_equity.png"


def _build_model():
    """Standardized logistic regression (lbfgs). Scaling is essential because
    the features span very different scales (price levels vs. returns vs. RSI)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="lbfgs"),
    )


def walk_forward(df_train, df_predict, retrain_days=RETRAIN_DAYS, warmup=0):
    """Expanding-window walk-forward over df_predict (train on df_train first).

    Returns (dates, y_true, returns_1d, proba) for the predicted days only.
    A model is fit at the start and every `retrain_days` rows thereafter on all
    rows strictly before the current row (no future data in training).
    """
    cols = du.FEATURE_COLUMNS + ["target_7d"]
    train = df_train.dropna(subset=cols) if df_train is not None and not df_train.empty else pd.DataFrame()
    predict = df_predict.dropna(subset=cols)
    n_train = len(train)

    comb = pd.concat([train, predict]).sort_index()
    X = comb[du.FEATURE_COLUMNS].astype(float).to_numpy()
    y = comb["target_7d"].astype(float).to_numpy()
    ret = comb["returns_1d"].fillna(0.0).to_numpy()
    dates = comb.index

    n = len(y)
    # When there is no pretraining, skip a warmup prefix before scoring.
    predict_start = n_train if n_train > 0 else warmup
    proba = np.full(n, np.nan)
    model = None
    last_fit = -1

    for i in range(predict_start, n):
        need_refit = model is None or (i - last_fit >= retrain_days)
        if need_refit and i > 0 and np.unique(y[:i]).size >= 2:
            model = _build_model()
            model.fit(X[:i], y[:i])
            last_fit = i
        if model is not None:
            proba[i] = model.predict_proba(X[i:i + 1])[0, 1]

    mask = ~np.isnan(proba)
    return dates[mask], y[mask], ret[mask], proba[mask]


def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, zero_division=0)
    p1, r1, f1_1 = p[1], r[1], f1[1]
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": round(acc, 3), "precision": round(p1, 3), "recall": round(r1, 3),
        "f1": round(f1_1, 3), "auc": round(auc, 3),
    }


def best_threshold(y_true, proba):
    """Probability threshold (in 0.05..0.95) that maximizes F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 1.0, 0.05):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(round(best_t, 2)), float(best_f1)


def strategy_metrics(ret, proba, threshold):
    """Long-only daily-position strategy metrics at a given probability threshold."""
    pos = (proba >= threshold).astype(float)
    strat = pos * ret
    n = len(strat)
    total = float(np.prod(1.0 + strat) - 1.0) if n else float("nan")
    ann = (1 + total) ** (TRADING_DAYS / n) - 1.0 if (n and total > -1) else float("nan")
    mu, sig = float(strat.mean()), float(strat.std(ddof=1))
    sharpe = (mu / sig) * np.sqrt(TRADING_DAYS) if sig > 0 else float("nan")
    return {
        "long_days": int(pos.sum()),
        "total_return_%": round(total * 100, 2),
        "annual_return_%": round(ann * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate_%": round((strat > 0).mean() * 100, 1),
    }


def buy_hold_metrics(ret):
    n = len(ret)
    total = float(np.prod(1.0 + ret) - 1.0) if n else float("nan")
    ann = (1 + total) ** (TRADING_DAYS / n) - 1.0 if (n and total > -1) else float("nan")
    mu, sig = float(ret.mean()), float(ret.std(ddof=1))
    sharpe = (mu / sig) * np.sqrt(TRADING_DAYS) if sig > 0 else float("nan")
    return {"long_days": n, "total_return_%": round(total * 100, 2),
            "annual_return_%": round(ann * 100, 2), "sharpe": round(sharpe, 3),
            "win_rate_%": round((ret > 0).mean() * 100, 1)}


def save_predictions(path, dates, y, ret, proba, threshold):
    pos = (proba >= threshold).astype(float)
    strat = pos * ret
    df = pd.DataFrame({
        "date": dates,
        "actual_up_7d": y.astype(int),
        "proba_up": np.round(proba, 4),
        "predicted_long": (proba >= threshold).astype(int),
        "position": pos,
        "daily_return": np.round(ret, 6),
        "strategy_return": np.round(strat, 6),
    })
    df.to_csv(path, index=False)
    print(f"  Saved per-day predictions to {path}")


def plot_equity(path, dates, strat, bh, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (equity chart skipped: {e})")
        return
    eq_model = np.cumprod(1.0 + strat) - 1.0
    eq_bh = np.cumprod(1.0 + bh) - 1.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(dates)), eq_model * 100, label="Model long-only", lw=1.6)
    ax.plot(range(len(dates)), eq_bh * 100, label="Buy & hold", lw=1.4, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("trading day (predicted window)")
    ax.set_ylabel("cumulative return (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  Saved equity chart to {path}")


def _report(prefix, dates, y, ret, proba, threshold):
    cls = classification_metrics(y, proba, threshold)
    strat_m = strategy_metrics(ret, proba, threshold)
    strat_05 = strategy_metrics(ret, proba, 0.5)
    bh_m = buy_hold_metrics(ret)
    print(f"--- {prefix} walk-forward ({len(y)} scored days, {dates[0]} -> {dates[-1]}) ---")
    print(f"    threshold (max-F1 on analysis): {threshold}")
    print(f"    AUC (threshold-independent, no-skill baseline ~0.50): {cls['auc']}")
    print("    classification @ threshold: " + ", ".join(f"{k}={v}" for k, v in cls.items() if k != "auc"))
    print("    model strategy @F1-thr: " + ", ".join(f"{k}={v}" for k, v in strat_m.items()))
    print("    model strategy @0.50:   " + ", ".join(f"{k}={v}" for k, v in strat_05.items()))
    print("    buy & hold:              " + ", ".join(f"{k}={v}" for k, v in bh_m.items()))


def print_feature_importance(analysis_df):
    X, y = du.feature_matrix(analysis_df)
    if np.unique(y).size < 2:
        return
    m = _build_model().fit(X, y)
    coef = m.named_steps["logisticregression"].coef_[0]
    coefs = sorted(zip(X.columns, coef), key=lambda kv: kv[1], reverse=True)
    print("  Model coefficients (full-analysis fit, standardized features):")
    for name, c in coefs:
        print(f"    {name:18s} {c:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Step 3 walk-forward logistic-regression model.")
    parser.add_argument("--holdout", action="store_true",
                        help="Final forward report on the hold-out window (final fifth).")
    parser.add_argument("--retrain-days", type=int, default=RETRAIN_DAYS,
                        help="Retrain interval in days (default 30).")
    args = parser.parse_args()

    df = du.load_merged_dataset()
    cols = du.FEATURE_COLUMNS + ["target_7d", "returns_1d"]
    df = df.dropna(subset=cols)
    analysis, holdout = du.split_dataset(df)

    print("=== Step 3: expanding-window logistic regression walk-forward ===")
    print(f"  features ({len(du.FEATURE_COLUMNS)}): {du.FEATURE_COLUMNS}")
    print(f"  model: LogisticRegression | retrain every {args.retrain_days}d | horizon={HORIZON}d")
    print(f"  split: analysis {analysis.index[0]} -> {analysis.index[-1]} "
          f"({len(analysis)}d) | hold-out {holdout.index[0]} -> {holdout.index[-1]} ({len(holdout)}d)")
    print()

    if args.holdout:
        # 1) threshold selection on the analysis window
        a_dates, a_y, a_ret, a_proba = walk_forward(None, analysis, args.retrain_days, WARMUP_DAYS)
        threshold, f1 = best_threshold(a_y, a_proba)
        _report("ANALYSIS (threshold selection)", a_dates, a_y, a_ret, a_proba, threshold)
        # 2) forward-predict the hold-out with the analysis-trained threshold
        h_dates, h_y, h_ret, h_proba = walk_forward(analysis, holdout, args.retrain_days, 0)
        _report("HOLD-OUT (forward)", h_dates, h_y, h_ret, h_proba, threshold)
        save_predictions(PRED_CSV, h_dates, h_y, h_ret, h_proba, threshold)
        strat = (h_proba >= threshold).astype(float) * h_ret
        plot_equity(CHART_PNG, h_dates, strat, h_ret,
                    "Model vs Buy & Hold - hold-out window")
    else:
        dates, y, ret, proba = walk_forward(None, analysis, args.retrain_days, WARMUP_DAYS)
        threshold, f1 = best_threshold(y, proba)
        _report("ANALYSIS", dates, y, ret, proba, threshold)
        save_predictions(PRED_CSV, dates, y, ret, proba, threshold)
        strat = (proba >= threshold).astype(float) * ret
        plot_equity(CHART_PNG, dates, strat, ret, "Model vs Buy & Hold - analysis window")
        print()
        print_feature_importance(analysis)

    print("\nProtocol: expanding window, retrain every %d days on data strictly prior to day t."
          % args.retrain_days)


if __name__ == "__main__":
    main()

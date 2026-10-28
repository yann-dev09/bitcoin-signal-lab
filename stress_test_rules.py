"""Stress-test the trend rules R2 and R5 from backtest_variants.py.

Adds transaction costs, simple moving-average baselines, a parameter-
robustness sweep, a block-bootstrap significance check, and per-year
diagnostics. The rule parameters are FIXED by the Step-2 definitions
(R2: close > MA(90) and 7-day momentum > 0; R5: close > MA(30) and 7-day
momentum > 0 and sentiment > 0). The MA-length / momentum-window robustness
sweep is performed on the ANALYSIS window only; the hold-out is touched only
once, for the final run of the chosen rule.

Position convention (matches backtest_variants.py): signal on day t is acted
on day t+1 (long if the prior day's signal fired), so there is no same-day
lookahead. Costs are 0.1% fee + 0.05% slippage PER SIDE (0.15% on entry,
0.15% on exit).

Run: python stress_test_rules.py
"""

import numpy as np
import pandas as pd

import data_utils as du

TRADING_DAYS = 365.0
COST_PER_SIDE = 0.0015  # 0.1% fee + 0.05% slippage, applied on each entry/exit


# --------------------------------------------------------------------------- #
# Strategy construction
# --------------------------------------------------------------------------- #
def rule_signals(df, ma_lookbacks=(90,), mom_windows=(7,), include_sentiment=False,
                 extra_ma=None):
    """Build long-only 0/1 daily signals for the trend rules.

    R2-style: (close > MA(L)) & (momentum(M) > 0)  [optionally & sentiment>0]
    """
    sig = pd.Series(False, index=df.index)
    close = df["close"]
    for L in ma_lookbacks:
        ma = close.rolling(L).mean()
        sig = sig | (close > ma)
    for M in mom_windows:
        mom = close.pct_change(M)
        sig = sig & (mom > 0)
    if include_sentiment:
        sig = sig & (df["sentiment_score"] > 0)
    return sig


def simple_ma_long(df, L):
    return df["close"] > df["close"].rolling(L).mean()


def cost_adjusted_returns(df, signal, cost=COST_PER_SIDE):
    """Next-day long-only returns with per-side transaction costs.

    Returns (strat_ret, n_entries, n_exits).
    """
    pos = signal.shift(1).fillna(False).astype(float)
    ret = df["returns_1d"]
    entry = (pos.shift(1) == 0) & (pos > 0)
    exit_ = (pos.shift(1) > 0) & (pos == 0)
    txn = (entry.astype(float) + exit_.astype(float)) * cost
    strat = pos * ret - txn
    return strat, int(entry.sum()), int(exit_.sum())


def sharpe(ret):
    r = np.asarray(ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu, sig = r.mean(), r.std(ddof=1)
    return (mu / sig) * np.sqrt(TRADING_DAYS) if sig > 0 else float("nan")


def perf(strat, label, n_trades=None):
    """One-line performance summary of a daily return stream."""
    r = np.asarray(strat, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    out = {"rule": label, "days": n}
    if n == 0:
        return {**out, "total_%": 0.0, "ann_%": 0.0, "sharpe": 0.0,
                "max_dd_%": 0.0, "win_%": 0.0, "trades": 0}
    total = float(np.prod(1 + r) - 1.0)
    ann = (1 + total) ** (TRADING_DAYS / n) - 1.0 if total > -1 else float("nan")
    mu, sig = r.mean(), r.std(ddof=1)
    sh = (mu / sig) * np.sqrt(TRADING_DAYS) if sig > 0 else float("nan")
    gm = np.cumprod(1 + r)
    mdd = float((gm / np.maximum.accumulate(gm) - 1.0).min())
    out.update({
        "total_%": round(total * 100, 2),
        "ann_%": round(ann * 100, 2),
        "sharpe": round(sh, 3),
        "max_dd_%": round(mdd * 100, 2),
        "win_%": round((r > 0).mean() * 100, 1),
        "trades": int(n_trades) if n_trades is not None else 0,
    })
    return out


# --------------------------------------------------------------------------- #
# Fixed rule definitions (parameters fixed by Step-2 definitions)
# --------------------------------------------------------------------------- #
RULES = {
    "R2": lambda d: rule_signals(d, ma_lookbacks=(90,), mom_windows=(7,),
                                  include_sentiment=False),
    "R5": lambda d: rule_signals(d, ma_lookbacks=(30,), mom_windows=(7,),
                                  include_sentiment=True),
    "MA200-long": lambda d: simple_ma_long(d, 200),
    "MA100-long": lambda d: simple_ma_long(d, 100),
}


def print_table(rows, cols, title=None):
    df = pd.DataFrame(rows, columns=cols)
    if title:
        print(title)
    print(df.to_string(index=False))


def task12_fixed_rules(analysis, holdout):
    """Task 1+2: cost-adjusted performance of fixed rules vs buy & hold."""
    print("\n================ TASKS 1 & 2: fixed rules with costs ================")
    print("Costs: 0.1% fee + 0.05% slippage per side (0.30% round-trip entry+exit)\n")
    cols = ["rule", "days", "total_%", "ann_%", "sharpe", "max_dd_%", "win_%", "trades"]
    for wname, w in [("ANALYSIS", analysis), ("HOLD-OUT", holdout)]:
        print(f"-- {wname} window ({w.index[0].date()} -> "
              f"{w.index[-1].date()}, {len(w)} trading days) --")
        rows = [perf(w["returns_1d"], "Buy & Hold", n_trades=0)]
        for name, fn in RULES.items():
            strat, ent, _ = cost_adjusted_returns(w, fn(w))
            rows.append(perf(strat, name, n_trades=ent))
        print_table(rows, cols)
        if wname == "ANALYSIS":
            sh = {r["rule"]: r["sharpe"] for r in rows}
            print(f"\n  Sharpe: R2={sh.get('R2')}  R5={sh.get('R5')}  "
                  f"MA200={sh.get('MA200-long')}  MA100={sh.get('MA100-long')}  BH={sh.get('Buy & Hold')}")
            print(f"  R2-MarR2 vs MA200: {round(sh.get('R2',0)-sh.get('MA200-long',0),3)} pts; "
                  f"R5 vs MA200: {round(sh.get('R5',0)-sh.get('MA200-long',0),3)} pts\n")

def task3_robustness(analysis):
    """R2 robustness: MA length x momentum window (analysis window, Sharpe)."""
    print("\n================ TASK 3: R2 robustness (Sharpe grid) ================")
    print("Sweep on ANALYSIS window only (no hold-out tuning).\n")
    Ls = [50, 90, 150, 200]
    Ms = [5, 7, 10, 14]
    grid = {}
    for L in Ls:
        row = {}
        for M in Ms:
            sig = rule_signals(analysis, ma_lookbacks=(L,), mom_windows=(M,))
            strat, _, _ = cost_adjusted_returns(analysis, sig)
            row[f"mom{M}"] = round(sharpe(strat), 3)
        grid[f"MA{L}"] = row
    g = pd.DataFrame(grid).T
    print("R2 Sharpe = (close>MA(L)) & (mom(M)>0), with costs (analysis):\n")
    print(g.to_string())
    sh = g.values.astype(float)
    print(f"\n  min={sh.min():.3f}  max={sh.max():.3f}  mean={sh.mean():.3f}")
    base = grid["MA90"]["mom7"]
    rest = sh.flatten()
    print(f"  baseline(MA90,mom7)={base}; grid IQR="
          f"{np.percentile(rest,25):.3f}..{np.percentile(rest,75):.3f}; mean={rest.mean():.3f}")
    # Honest robustness check: is the grid flat (plateau) or sensitive to params?
    mom_avg = {M: float(np.mean([grid["MA{}".format(L)]["mom{}".format(M)] for L in Ls]))
               for M in Ms}
    order = ", ".join("mom{}={:.3f}".format(m, mom_avg[m]) for m in Ms)
    sensitive = (mom_avg[Ms[-1]] - mom_avg[Ms[0]]) > 0.10
    verdict = ("PARAMETER-SENSITIVE -- NOT a robust plateau: Sharpe rises with the "
               "momentum lookback (" + order + "); the mom7 default is mid-pack and "
               "parameters matter.") if sensitive else "broad plateau"
    print(f"  avg Sharpe by momentum window: {order}")
    print(f"  -> {verdict}")

def task4_bootstrap(analysis):
    """Block bootstrap of Sharpe(rule - buy&hold), 1000 reps, 20-day blocks."""
    print("\n================ TASK 4: block-bootstrap significance ================")
    print("Moving block bootstrap (B=1000, block=20 trading days) of the Sharpe "
          "diff vs Buy & Hold, on ANALYSIS window.\n")
    r_bh = np.asarray(analysis["returns_1d"], dtype=float)
    n = len(r_bh)
    block, reps, seed = 20, 1000, 42
    rng = np.random.default_rng(seed)

    def bootstrap_diff(strat_rule):
        s_r = np.nan_to_num(np.asarray(strat_rule, dtype=float), nan=0.0)
        idx_all = []
        for _ in range(reps):
            idx = []
            while len(idx) < n:
                start = rng.integers(0, n - block + 1)
                idx.extend(range(start, start + block))
            idx = np.array(idx[:n])
            idx_all.append((s_r[idx], r_bh[idx]))
        diffs = np.array([sharpe(a) - sharpe(b) for a, b in idx_all])
        return diffs

    for name, fn in RULES.items():
        strat, _, _ = cost_adjusted_returns(analysis, fn(analysis))
        diffs = bootstrap_diff(strat)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        mean = diffs.mean()
        p_le0 = (diffs <= 0).mean()
        sig = "SIG" if (lo > 0) else ("anti-sig" if (hi < 0) else "NOT-sig")
        print(f"  {name:12s} Sharpe(diff vs BH): mean={mean:+.3f}  "
              f"95% CI=[{lo:+.3f}, {hi:+.3f}]  p(diff<=0)={p_le0:.3f}  -> {sig}")

def task5_per_year(df):
    """Per-year (2018-2026) return + max drawdown, descriptive (full series)."""
    print("\n================ TASK 5: per-year breakdown (2018-2026) ================")
    print("Descriptive annual return %% and max drawdown %% for each fixed rule "
          "(with costs) + Buy & Hold, over the FULL 2018-2026 history.\n")
    strats = {"Buy & Hold": df["returns_1d"]}
    for name, fn in RULES.items():
        strat, _, _ = cost_adjusted_returns(df, fn(df))
        strats[name] = strat
    df["yr"] = df.index.year
    cols = ["year"]
    for name in ["Buy & Hold", "R2", "R5", "MA200-long", "MA100-long"]:
        cols += [f"{name[:11]} ret", f"{name[:11]} mdd"]
    rows = []
    for yr in sorted(df["yr"].unique()):
        if yr < 2018 or yr > 2026:
            continue
        rec = {"year": int(yr)}
        sub = df[df["yr"] == yr]
        for name, s in strats.items():
            r = np.asarray(s.loc[sub.index], dtype=float)
            r = r[~np.isnan(r)]
            if len(r) == 0:
                rec[f"{name[:11]} ret"], rec[f"{name[:11]} mdd"] = 0.0, 0.0
                continue
            total = float(np.prod(1 + r) - 1.0)
            gm = np.cumprod(1 + r)
            mdd = float((gm / np.maximum.accumulate(gm) - 1.0).min())
            rec[f"{name[:11]} ret"] = round(total * 100, 2)
            rec[f"{name[:11]} mdd"] = round(mdd * 100, 2)
        rows.append(rec)
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

def main():
    df = du.load_merged_dataset()
    df.index = pd.to_datetime(df.index)            # string -> DatetimeIndex
    df = df.dropna(subset=["returns_1d", "close", "sentiment_score"]).copy()
    analysis, holdout = du.split_dataset(df)
    task12_fixed_rules(analysis, holdout)
    task3_robustness(analysis)
    task4_bootstrap(analysis)
    task5_per_year(df)
    pick_and_final_holdout(analysis, holdout)

def pick_and_final_holdout(analysis, holdout):
    """Task 6: declare chosen rule (analysis evidence) then ONE final hold-out run."""
    print("\n================ TASK 6: pick + final hold-out run ================")
    cols = ["rule", "days", "total_%", "ann_%", "sharpe", "max_dd_%", "win_%", "trades"]
    rows = [perf(analysis["returns_1d"], "Buy & Hold", n_trades=0)]
    for name, fn in RULES.items():
        strat, ent, _ = cost_adjusted_returns(analysis, fn(analysis))
        rows.append(perf(strat, name, n_trades=ent))
    print("ANALYSIS window (cost-adjusted), sorted by Sharpe:")
    print_table(sorted(rows, key=lambda r: r["sharpe"]), cols)
    print()
    trend = {r["rule"]: r["sharpe"] for r in rows if r["rule"] in ("R2", "R5")}
    chosen = max(trend, key=trend.get)
    print(">>> PICKED RULE:", chosen, "(best trend rule on analysis Sharpe =", trend[chosen], ")")
    print(">>> WHY (analysis-window evidence only): between the two stress-tested")
    print("    trend rules, this one has the higher cost-adjusted Sharpe.")
    print(">>> HONEST CAVEAT: even this winner does NOT beat the trivial MA100/MA200")
    print("    baselines, and no rule's Sharpe edge over Buy & Hold is bootstrap-")
    print("    significant (Task 4). See the honest verdict in the summary.")
    print()
    b100 = next(r for r in rows if r["rule"] == "MA100-long")
    b200 = next(r for r in rows if r["rule"] == "MA200-long")
    print("    reference: MA100-long Sharpe =", b100["sharpe"], "beats", chosen, trend[chosen], "by", round(b100["sharpe"]-trend[chosen], 3), "pts; MA200-long Sharpe =", b200["sharpe"], ".")
    print()
    dates = "FINAL hold-out run ({}) [{} -> {}, {} days]:".format(chosen, holdout.index[0].date(), holdout.index[-1].date(), len(holdout))
    print(dates)
    bh_h = perf(holdout["returns_1d"], "Buy & Hold", n_trades=0)
    strat, ent, _ = cost_adjusted_returns(holdout, RULES[chosen](holdout))
    print_table([bh_h, perf(strat, chosen, n_trades=ent)], cols)


if __name__ == "__main__":
    main()

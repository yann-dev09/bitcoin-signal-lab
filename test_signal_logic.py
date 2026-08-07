"""
Temporary test script to verify the "Cautious" fundamentals-override rule.

Replicates the exact signal combiner logic from index.html (determineSignal +
computeFundamentalChanges) in Python, loads the REAL fundamentals.json, and
feeds it FAKED bullish inputs (RSI=25, recent hammer=True, positive sentiment)
to confirm that a >10% drop in transactions forces the signal to "Cautious".

This script does NOT modify any real app files.
"""

import json


def compute_fundamental_changes(fundamentals):
    """Mirror of computeFundamentalChanges() from index.html."""
    if not fundamentals:
        return {"addrChange": None, "txChange": None}

    def last_seven_change(metric_key):
        series = fundamentals.get(metric_key, {}).get("series")
        if not series or len(series) < 2:
            return None
        last7 = series[-7:]
        first = last7[0]["value"]
        last = last7[-1]["value"]
        if first == 0:
            return None
        return ((last - first) / first) * 100

    return {
        "addrChange": last_seven_change("unique_addresses"),
        "txChange": last_seven_change("transactions_per_day"),
    }


def determine_signal(rsi, has_recent_hammer, overall_sentiment, fundamentals):
    """Mirror of determineSignal() from index.html.

    Args:
        rsi: faked RSI value (or None)
        has_recent_hammer: faked bool for whether the last candle is a hammer
        overall_sentiment: faked overall average sentiment score
        fundamentals: real fundamentals dict loaded from fundamentals.json
    """
    is_positive = overall_sentiment >= 0.05
    is_strongly_negative = overall_sentiment <= -0.2

    # Fundamentals: % change over the last 7 days (from real data)
    changes = compute_fundamental_changes(fundamentals)
    addr_change = changes["addrChange"]
    tx_change = changes["txChange"]
    tx_dropped_over_10 = tx_change is not None and tx_change < -10
    fundamentals_strong = (
        addr_change is not None
        and tx_change is not None
        and addr_change > 0
        and tx_change > 0
    )

    # Evaluate bullish technicals/sentiment conditions (faked inputs)
    bullish_reasons = []

    if rsi is not None and rsi < 30 and is_positive:
        bullish_reasons.append(
            f"RSI {rsi:.1f} < 30 (oversold) + positive sentiment "
            f"({overall_sentiment:+.2f})"
        )

    if has_recent_hammer and not is_strongly_negative:
        bullish_reasons.append(
            f"Hammer pattern on FAKE_DATE + sentiment not strongly negative "
            f"({overall_sentiment:+.2f})"
        )

    if bullish_reasons:
        if tx_dropped_over_10:
            signal = "Cautious"
            reason = (
                f"Bullish technicals but weakening on-chain activity "
                f"(transactions {tx_change:.1f}% over 7d)"
            )
        elif fundamentals_strong:
            signal = "Bullish"
            reason = (
                "; ".join(bullish_reasons)
                + f" + strong on-chain activity (addresses {addr_change:+.1f}%, "
                f"transactions {tx_change:+.1f}% over 7d)"
            )
        else:
            signal = "Bullish"
            reason = "; ".join(bullish_reasons)
    else:
        signal = "Neutral"
        reason = (
            "No strong bullish confluence (RSI not oversold with positive "
            "sentiment, or no recent hammer)"
        )

    return {
        "signal": signal,
        "reason": reason,
        "rsi": rsi,
        "hasRecentHammer": has_recent_hammer,
        "overall": overall_sentiment,
        "addrChange": addr_change,
        "txChange": tx_change,
    }


def main():
    # Load REAL fundamentals data
    with open("fundamentals.json", encoding="utf-8") as f:
        fundamentals = json.load(f)

    # Print real fundamentals context
    changes = compute_fundamental_changes(fundamentals)
    print("=== REAL fundamentals.json (7-day changes) ===")
    print(f"  Unique addresses 7d: {changes['addrChange']:+.2f}%")
    print(f"  Transactions 7d:    {changes['txChange']:+.2f}%")
    print()

    # FAKED bullish inputs
    rsi = 25.0            # oversold -> bullish
    has_recent_hammer = True
    overall_sentiment = 0.30   # positive -> bullish

    print("=== FAKED bullish inputs ===")
    print(f"  RSI = {rsi} (oversold, < 30)")
    print(f"  Recent hammer = {has_recent_hammer}")
    print(f"  Overall sentiment = {overall_sentiment:+.2f} (positive)")
    print()

    result = determine_signal(rsi, has_recent_hammer, overall_sentiment, fundamentals)

    print("=== RESULT ===")
    print(f"  Signal: {result['signal']}")
    print(f"  Reason: {result['reason']}")
    print()

    # Assert the expected behavior
    tx_change = result["txChange"]
    if tx_change is not None and tx_change < -10:
        expected = "Cautious"
    else:
        expected = "Bullish"

    print(f"  Expected signal (transactions {tx_change:+.1f}%): {expected}")
    if result["signal"] == expected:
        print("  ✓ PASS — fundamentals override works as expected")
    else:
        print("  ✗ FAIL — signal does not match expected behavior")


if __name__ == "__main__":
    main()
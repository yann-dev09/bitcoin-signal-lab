"""Combined signal combiner.

This module reproduces the signal-decision logic of `determineSignal()` in
index.html (JavaScript) and ports it to Python for historical backtesting.

The exact logic from index.html (lines ~617-666) is:

    isPositive          = overall >= 0.05
    isStronglyNegative    = overall <= -0.2     // hammer requires !isStronglyNegative

    // bullish reasons
    reason 1: rsi < 30  AND  sentiment >= 0.05
    reason 2: has_hammer AND  sentiment > -0.2        (== !isStronglyNegative)

    if any bullish reason:
        if transactions dropped > 10% over 7d   -> "Cautious"   (on-chain veto)
        elif addresses & transactions BOTH up    -> "Bullish"    (stronger)
        else                                     -> "Bullish"
    else:
        -> "Neutral"

ADDITIONAL Bearish branch (requested by the task spec, NOT present in the JS):
    rsi > 70  AND  sentiment <= 0.05  -> "Bearish"

The Bearish branch is checked FIRST.  Rationale: RSI > 70 (overbought) and
RSI < 30 (oversold) are mutually exclusive, so the RSI-driven Bearish condition
never clashes with the RSI-driven Bullish reason.  The only possible overlap
with the existing logic is a *hammer* (weak reversal) co-occurring with an
overbought + non-positive-sentiment day; in that case Bearish rightly wins
because overbought + no positive sentiment is a much stronger signal than a
bare hammer with weak sentiment.

Function signature (changes are precomputed, as required by the task):

    determine_signal(rsi, has_hammer, sentiment,
                     addr_change_7d, tx_change_7d) -> dict
"""

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SENTIMENT_POSITIVE = 0.05      # isPositive threshold (>=)
SENTIMENT_STRONGLY_NEGATIVE = -0.2  # hammer requires strictly above this (> -0.2)
TX_DROP_CAUTION = -10          # transactions drop > 10% => Cautious


def determine_signal(rsi, has_hammer, sentiment, addr_change_7d, tx_change_7d):
    """Determine the combined market signal for a single day.

    Args:
        rsi:            RSI(14) value (Wilder). None if not yet available.
        has_hammer:     bool - whether today's candle is a hammer pattern.
        sentiment:      float - sentiment score on -1..+1 scale (FNG proxy).
        addr_change_7d: float or None - 7-day % change in unique addresses.
        tx_change_7d:   float or None - 7-day % change in transactions.

    Returns:
        dict with keys: signal ("Bullish"|"Bearish"|"Cautious"|"Neutral"),
        reason, rsi, has_hammer, sentiment, addr_change_7d, tx_change_7d.
    """
    # ------------------------------------------------------------------
    # NEW Bearish branch (added per task; checked first, see module docstring)
    # ------------------------------------------------------------------
    if rsi is not None and rsi > RSI_OVERBOUGHT and sentiment <= SENTIMENT_POSITIVE:
        return {
            "signal": "Bearish",
            "reason": (
                f"RSI {rsi:.1f} > 70 (overbought) and sentiment "
                f"{sentiment:+.2f} <= +0.05 (not positive)"
            ),
            "rsi": rsi,
            "has_hammer": has_hammer,
            "sentiment": sentiment,
            "addr_change_7d": addr_change_7d,
            "tx_change_7d": tx_change_7d,
        }

    # ------------------------------------------------------------------
    # Exact reproduction of determineSignal() bullish / Cautious / Neutral
    # ------------------------------------------------------------------
    is_positive = sentiment >= SENTIMENT_POSITIVE
    # isStronglyNegative is `sentiment <= -0.2`; hammer needs `!isStronglyNegative`
    # i.e. `sentiment > -0.2`.
    sentiment_not_strongly_negative = sentiment > SENTIMENT_STRONGLY_NEGATIVE

    tx_dropped_over_10 = tx_change_7d is not None and tx_change_7d < TX_DROP_CAUTION
    fundamentals_strong = (
        addr_change_7d is not None
        and tx_change_7d is not None
        and addr_change_7d > 0
        and tx_change_7d > 0
    )

    bullish_reasons = []

    # reason 1: RSI oversold + positive sentiment
    if rsi is not None and rsi < RSI_OVERSOLD and is_positive:
        bullish_reasons.append(
            f"RSI {rsi:.1f} < 30 (oversold) + positive sentiment ({sentiment:+.2f})"
        )

    # reason 2: hammer + sentiment not strongly negative
    if has_hammer and sentiment_not_strongly_negative:
        bullish_reasons.append(
            f"Hammer pattern + sentiment not strongly negative ({sentiment:+.2f})"
        )

    if bullish_reasons:
        if tx_dropped_over_10:
            signal = "Cautious"
            reason = (
                f"Bullish technicals but weakening on-chain activity "
                f"(transactions {tx_change_7d:+.1f}% over 7d)"
            )
        elif fundamentals_strong:
            signal = "Bullish"
            reason = (
                "; ".join(bullish_reasons)
                + f" + strong on-chain activity "
                f"(addresses {addr_change_7d:+.1f}%, transactions "
                f"{tx_change_7d:+.1f}% over 7d)"
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
        "has_hammer": has_hammer,
        "sentiment": sentiment,
        "addr_change_7d": addr_change_7d,
        "tx_change_7d": tx_change_7d,
    }

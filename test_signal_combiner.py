"""Unit tests for signal_combiner.determine_signal().

Run directly:  py test_signal_combiner.py
Or with pytest: pytest test_signal_combiner.py

These cases mirror the exact logic of determineSignal() in index.html plus the
additional Bearish branch (RSI > 70 and sentiment <= 0.05 -> Bearish).
"""

from signal_combiner import determine_signal

ALL_UP = 5.0    # both addresses and tx up 5% (not a >10% drop, weakly positive)
TX_DROP = -15.0  # transactions dropped 15% (> 10% drop threshold)
TX_FLAT_NEG = -5.0  # small transaction decline (below caution threshold)


def test_strong_bullish():
    """RSI < 30 + positive sentiment + both metrics up -> Bullish (strong)."""
    r = determine_signal(rsi=25.0, has_hammer=False, sentiment=0.30,
                         addr_change_7d=3.0, tx_change_7d=4.0)
    assert r["signal"] == "Bullish", r
    assert "strong on-chain" in r["reason"], r


def test_cautious_via_tx_drop():
    """RSI < 30 + positive sentiment but transactions dropped > 10% -> Cautious."""
    r = determine_signal(rsi=25.0, has_hammer=False, sentiment=0.30,
                         addr_change_7d=3.0, tx_change_7d=TX_DROP)
    assert r["signal"] == "Cautious", r


def test_bullish_weak_fundamentals():
    """RSI < 30 + positive sentiment, tx drop small -> Bullish (not Cautious)."""
    r = determine_signal(rsi=25.0, has_hammer=False, sentiment=0.30,
                         addr_change_7d=3.0, tx_change_7d=TX_FLAT_NEG)
    assert r["signal"] == "Bullish", r
    # fundamentals are NOT strong (tx still <= 0) so reason should not claim
    # "strong on-chain activity"
    assert "strong on-chain" not in r["reason"], r


def test_hammer_bullish():
    """Hammer + sentiment not strongly negative -> Bullish, regardless of RSI."""
    r = determine_signal(rsi=50.0, has_hammer=True, sentiment=0.01,
                         addr_change_7d=ALL_UP, tx_change_7d=ALL_UP)
    assert r["signal"] == "Bullish", r


def test_hammer_with_strongly_negative_sentiment():
    """Hammer + strongly negative sentiment (<= -0.2) -> NOT bullish -> Neutral."""
    r = determine_signal(rsi=50.0, has_hammer=True, sentiment=-0.25,
                         addr_change_7d=ALL_UP, tx_change_7d=ALL_UP)
    # sentiment <= 0.05 and RSI not > 70, so not Bearish; no bullish reason
    assert r["signal"] == "Neutral", r


def test_bearish_overbought():
    """RSI > 70 + sentiment <= 0.05 -> Bearish (new branch)."""
    r = determine_signal(rsi=75.0, has_hammer=False, sentiment=0.0,
                         addr_change_7d=3.0, tx_change_7d=4.0)
    assert r["signal"] == "Bearish", r


def test_bearish_overbought_positive_sentiment_not_bearish():
    """RSI > 70 but sentiment > 0.05 -> NOT Bearish -> Neutral (no bullish reason)."""
    r = determine_signal(rsi=75.0, has_hammer=False, sentiment=0.10,
                         addr_change_7d=3.0, tx_change_7d=4.0)
    assert r["signal"] == "Neutral", r


def test_bearish_beats_hammer():
    """RSI > 70 + sentiment <= 0.05 + hammer -> Bearish wins (checked first)."""
    r = determine_signal(rsi=72.0, has_hammer=True, sentiment=-0.10,
                         addr_change_7d=3.0, tx_change_7d=4.0)
    assert r["signal"] == "Bearish", r


def test_plain_neutral():
    """RSI mid-range, no hammer, neutral sentiment, no strong drivers -> Neutral."""
    r = determine_signal(rsi=50.0, has_hammer=False, sentiment=0.0,
                         addr_change_7d=ALL_UP, tx_change_7d=ALL_UP)
    assert r["signal"] == "Neutral", r


def test_none_rsi_neutral():
    """rsi=None -> cannot be oversold/overbought; hammer only -> Bullish here."""
    r = determine_signal(rsi=None, has_hammer=True, sentiment=0.0,
                         addr_change_7d=ALL_UP, tx_change_7d=ALL_UP)
    assert r["signal"] == "Bullish", r


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print()
    print(f"{passed} passed, {failed} failed ({len(tests)} total)")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)

"""
Strategy Correlation Engine test — run after test_fusion.py passes.

Usage:
    python scripts/test_correlation.py

What this tests:
  1. Correlation matrix computation — synthetic correlated data
  2. Insufficient history — returns None gracefully
  3. Save + retrieve matrix from DB
  4. Dynamic penalty formula (Section 23.1: scales 0.7→1.0 maps to 0→0.2)
  5. Static co-fire penalty (Trend + Breakout)
  6. Combined penalty capped at 50%
  7. apply_correlation_penalty() — no matrix exists (safe no-op)
  8. apply_correlation_penalty() — full integration with stored matrix
  9. CorrelationEngine.run() batch job
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import numpy as np

from app.correlation.engine import (
    compute_correlation_matrix, save_correlation_matrix, get_correlation_matrix,
    apply_correlation_penalty, _dynamic_penalty, _static_cofire_penalty,
    CorrelationEngine, STRATEGIES,
    CORRELATION_THRESHOLD, MAX_PENALTY_AT_R1, MAX_COMBINED_PENALTY,
    MIN_SAMPLES_REQUIRED,
)
from app.db import get_sync_db


def cleanup_test_data(stock: str):
    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM strategy_correlations WHERE stock = %s", (stock,))
            cur.execute("DELETE FROM daily_signals WHERE stock = %s", (stock,))
    except Exception as e:
        print(f"  ⚠️  Cleanup failed: {e}")


def seed_correlated_scores(stock: str, n_days: int = 65):
    """
    Insert synthetic daily_signals rows where trend and breakout scores
    are highly correlated (r > 0.9), to test the dynamic penalty path.
    """
    today = date.today()
    rng = np.random.default_rng(42)
    base = rng.normal(0, 30, n_days)              # shared underlying signal
    trend_scores    = base + rng.normal(0, 3, n_days)     # highly correlated with base
    breakout_scores = base + rng.normal(0, 3, n_days)     # highly correlated with base
    momentum_scores = rng.normal(0, 30, n_days)            # independent
    reversion_scores= rng.normal(0, 30, n_days)            # independent
    volume_scores   = rng.normal(0, 30, n_days)            # independent

    try:
        with get_sync_db() as conn:
            cur = conn.cursor()
            for i in range(n_days):
                d = today - timedelta(days=n_days - i)
                cur.execute(
                    """
                    INSERT INTO daily_signals (
                        stock, date, signal, quant_score, confidence_pct, regime,
                        trend_score, momentum_score, reversion_score,
                        breakout_score, volume_score, valid_until
                    ) VALUES (
                        %s, %s, 'HOLD'::signal_type, 0, 50, 'UNCERTAIN'::regime_type,
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (date, stock) DO UPDATE SET
                        trend_score=EXCLUDED.trend_score,
                        breakout_score=EXCLUDED.breakout_score,
                        momentum_score=EXCLUDED.momentum_score,
                        reversion_score=EXCLUDED.reversion_score,
                        volume_score=EXCLUDED.volume_score
                    """,
                    (stock, d,
                     float(trend_scores[i]), float(momentum_scores[i]),
                     float(reversion_scores[i]), float(breakout_scores[i]),
                     float(volume_scores[i]), d),
                )
    except Exception as e:
        print(f"  ❌ Seeding failed: {e}")
        raise


def test_dynamic_penalty_formula():
    print("\n── Test 1: Dynamic penalty formula (Section 23.1) ───")
    cases = [
        (0.5, 0.0),    # below threshold → no penalty
        (0.7, 0.0),    # exactly at threshold → no penalty
        (0.85, 0.1),   # midpoint → half of max penalty
        (1.0, 0.2),    # full correlation → max penalty
    ]
    for r, expected in cases:
        penalty = _dynamic_penalty(r)
        status = "✅" if abs(penalty - expected) < 0.001 else "❌"
        print(f"  {status} r={r} → penalty={penalty:.3f} (expected {expected:.3f})")
        assert abs(penalty - expected) < 0.001


def test_static_cofire():
    print("\n── Test 2: Static co-fire penalty (Trend+Breakout) ───")
    # Both fire together → penalty applies to both
    p1 = _static_cofire_penalty("trend", {"trend", "breakout", "momentum"})
    p2 = _static_cofire_penalty("breakout", {"trend", "breakout", "momentum"})
    assert p1 == 0.20 and p2 == 0.20
    print(f"  ✅ Trend+Breakout co-firing → both get {p1:.0%} penalty")

    # Only trend fires, no breakout → no penalty
    p3 = _static_cofire_penalty("trend", {"trend", "momentum"})
    assert p3 == 0.0
    print(f"  ✅ Trend alone (no breakout) → {p3:.0%} penalty")


def test_insufficient_history():
    print("\n── Test 3: Insufficient history → graceful None ─────")
    cleanup_test_data("TEST_NOHIST")
    matrix = compute_correlation_matrix("TEST_NOHIST", date.today())
    assert matrix is None
    print(f"  ✅ No history → returns None (not an error)")


def test_compute_and_save_matrix():
    print("\n── Test 4: Compute + save correlation matrix ────────")
    stock = "TEST_CORR"
    cleanup_test_data(stock)
    seed_correlated_scores(stock, n_days=65)

    matrix = compute_correlation_matrix(stock, date.today())
    assert matrix is not None
    print(f"  ✅ Matrix computed with {MIN_SAMPLES_REQUIRED}+ days history")

    trend_breakout_r = matrix["trend"]["breakout"]
    print(f"  Trend↔Breakout correlation: {trend_breakout_r:.3f} (seeded to be high)")
    assert trend_breakout_r > 0.7, "Synthetic data should show high correlation"
    print(f"  ✅ High correlation detected as expected (r > 0.7)")

    trend_momentum_r = matrix["trend"]["momentum"]
    print(f"  Trend↔Momentum correlation: {trend_momentum_r:.3f} (seeded independent)")
    assert abs(trend_momentum_r) < 0.5, "Independent series should show low correlation"
    print(f"  ✅ Low correlation for independent strategies")

    save_correlation_matrix(stock, date.today(), matrix)
    retrieved = get_correlation_matrix(stock, date.today())
    assert retrieved is not None
    assert abs(retrieved["trend"]["breakout"] - trend_breakout_r) < 0.001
    print(f"  ✅ Matrix saved and retrieved correctly from DB")

    return stock


def test_apply_penalty_no_matrix():
    print("\n── Test 5: apply_correlation_penalty — no matrix (no-op) ──")
    cleanup_test_data("TEST_EMPTY")
    base_weights = {"trend": 0.30, "momentum": 0.25, "reversion": 0.10,
                     "breakout": 0.20, "volume": 0.10, "risk": 0.05}
    # NOTE: active_strategies deliberately excludes both trend and breakout
    # together, since Section 8's static co-fire penalty applies independent
    # of whether a dynamic correlation matrix exists (design doc: "applied
    # ... in addition to the static co-firing penalty from Section 8").
    # This set isolates the "no matrix" path with zero static penalty too,
    # so it's a true no-op test.
    adjusted, notes = apply_correlation_penalty(
        stock="TEST_EMPTY",
        base_weights=base_weights,
        active_strategies={"momentum", "reversion", "volume"},
    )
    assert adjusted == base_weights
    assert notes == []
    print(f"  ✅ No stored matrix, no co-fire pair active → weights unchanged (safe no-op)")


def test_apply_penalty_with_matrix(stock: str):
    print("\n── Test 6: apply_correlation_penalty — full integration ──")
    base_weights = {"trend": 0.30, "momentum": 0.25, "reversion": 0.10,
                     "breakout": 0.20, "volume": 0.10, "risk": 0.05}

    adjusted, notes = apply_correlation_penalty(
        stock=stock,
        base_weights=base_weights,
        active_strategies={"trend", "momentum", "reversion", "breakout", "volume"},
    )

    print(f"  Base weights:     {base_weights}")
    print(f"  Adjusted weights: {adjusted}")
    for n in notes:
        print(f"    → {n}")

    # Trend and Breakout should both be penalized (dynamic + static combine)
    assert adjusted["trend"] < base_weights["trend"]
    assert adjusted["breakout"] < base_weights["breakout"]
    print(f"  ✅ Trend and Breakout both penalized (correlated + co-fire)")

    # Combined penalty never exceeds 50%, and is NOT diluted by renormalization
    trend_reduction = 1 - (adjusted["trend"] / base_weights["trend"])
    assert trend_reduction <= MAX_COMBINED_PENALTY + 0.01
    print(f"  ✅ Combined penalty capped at {MAX_COMBINED_PENALTY:.0%} "
          f"(actual: {trend_reduction:.1%})")

    # Untouched strategies (no correlation, no co-fire pair) must be
    # byte-identical to base weights — no more silent redistribution.
    # "risk" especially must never move: it's a penalty subtractor in the
    # final score (Section 9), not a strategy weight.
    for strat in ("momentum", "reversion", "volume", "risk"):
        assert adjusted[strat] == base_weights[strat], (
            f"{strat} should be untouched but changed: "
            f"{base_weights[strat]} → {adjusted[strat]}"
        )
    print(f"  ✅ Untouched strategies unchanged (no silent redistribution, "
          f"'risk' weight intact)")


def test_correlation_engine_batch(stock: str):
    print("\n── Test 7: CorrelationEngine batch run ──────────────")
    engine = CorrelationEngine(run_date=date.today())
    results = engine.run([stock, "TEST_NOHIST", "NONEXISTENT_STOCK"])

    print(f"  Results: {results}")
    assert results[stock] == True
    assert results["TEST_NOHIST"] == False
    print(f"  ✅ Batch run: stock with history updated, "
          f"stocks without history skipped gracefully")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Correlation Engine Test")
    print(f"{'='*60}")

    try:
        test_dynamic_penalty_formula()
        test_static_cofire()
        test_insufficient_history()
        stock = test_compute_and_save_matrix()
        test_apply_penalty_no_matrix()
        test_apply_penalty_with_matrix(stock)
        test_correlation_engine_batch(stock)

        print(f"\n{'='*60}")
        print("  ✅ Strategy Correlation Engine verified")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_data("TEST_CORR")
        cleanup_test_data("TEST_NOHIST")
        cleanup_test_data("TEST_EMPTY")


if __name__ == "__main__":
    main()
"""
Cross-Stock Correlation in Portfolio Manager — Section 23.2 test.

Distinct from scripts/test_correlation.py, which tests intra-stock
strategy correlation (Section 23.1, fusion layer). This tests the
portfolio-layer cross-stock price correlation added to
app/portfolio/manager.py.

Usage:
    python scripts/test_portfolio_correlation.py

Requires market_data already ingested for at least 2 correlated stocks
(run scripts/test_ingestion.py first if you haven't).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

from app.portfolio.manager import (
    _pairwise_correlation, _cross_stock_penalty,
    apply_cross_stock_correlation_penalty,
    CROSS_STOCK_CORR_THRESHOLD, CROSS_STOCK_MAX_PENALTY,
)


def test_penalty_formula():
    print("\n── Test 1: Cross-stock penalty formula ──────────────")
    cases = [
        (0.5,  0.0),
        (0.7,  0.0),
        (0.85, CROSS_STOCK_MAX_PENALTY / 2),
        (1.0,  CROSS_STOCK_MAX_PENALTY),
    ]
    for r, expected in cases:
        pen = _cross_stock_penalty(r)
        status = "✅" if abs(pen - expected) < 0.001 else "❌"
        print(f"  {status} r={r} → penalty={pen:.3f} (expected {expected:.3f})")
        assert abs(pen - expected) < 0.001


def test_real_pairwise_correlation():
    print("\n── Test 2: Real pairwise correlation (needs ingested data) ──")
    today = date.today()

    pairs = [
        ("RELIANCE", "TCS"),
        ("RELIANCE", "HDFCBANK"),
        ("TCS", "HDFCBANK"),
    ]
    any_computed = False
    for a, b in pairs:
        r = _pairwise_correlation(a, b, today)
        if r is None:
            print(f"  ⚠️  {a} vs {b}: insufficient history (run test_ingestion.py first)")
            continue
        any_computed = True
        pen = _cross_stock_penalty(r)
        flag = " ← penalty applies" if pen > 0 else ""
        print(f"  ✅ {a} vs {b}: r={r:+.3f} → penalty={pen:.1%}{flag}")

    if not any_computed:
        print("  ⚠️  Skipped — no market data found. Run test_ingestion.py first.")
    return any_computed


def test_apply_cross_stock_penalty():
    print("\n── Test 3: apply_cross_stock_correlation_penalty() ──")
    today = date.today()

    result = apply_cross_stock_correlation_penalty(
        candidates=["TCS", "HDFCBANK"],
        accepted_so_far=["RELIANCE"],
        as_of_date=today,
    )
    for stock, (pen, worst_with) in result.items():
        print(f"  {stock}: penalty={pen:.1%} worst_with={worst_with}")
    print("  ✅ Function runs without error against real DB data")


def test_empty_accepted_is_noop():
    print("\n── Test 4: No accepted positions yet → zero penalty (no-op) ──")
    result = apply_cross_stock_correlation_penalty(
        candidates=["RELIANCE", "TCS"],
        accepted_so_far=[],
        as_of_date=date.today(),
    )
    for stock, (pen, worst_with) in result.items():
        assert pen == 0.0 and worst_with is None
    print("  ✅ Empty accepted_so_far → all penalties zero, safe no-op")


def main():
    print(f"\n{'='*60}")
    print("  Portfolio Manager — Cross-Stock Correlation (§23.2) Test")
    print(f"{'='*60}")

    test_penalty_formula()
    has_data = test_real_pairwise_correlation()
    test_apply_cross_stock_penalty()
    test_empty_accepted_is_noop()

    print(f"\n{'='*60}")
    if has_data:
        print("  ✅ Cross-Stock Correlation Penalty verified (real data)")
    else:
        print("  ⚠️  Verified formula + no-op paths only — ingest data for full test")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
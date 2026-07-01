"""
Cross-Stock Correlation in Portfolio Acceptance Engine — Section 23.2 test.

Distinct from scripts/test_correlation.py, which tests intra-stock
strategy correlation (Section 23.1, fusion layer). This tests the
portfolio-layer cross-stock price correlation, now consolidated into
app/portfolio/acceptance.py's PortfolioAcceptanceEngine (previously
lived as standalone functions in app/portfolio/manager.py — see that
module's docstring for why the consolidation happened).

Usage:
    python scripts/test_portfolio_correlation.py

Requires market_data already ingested for at least 2 correlated stocks
(run scripts/test_ingestion.py first if you haven't).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

from app.portfolio.acceptance import (
    _pairwise_correlation, _cross_stock_penalty,
    AcceptanceCandidate, PortfolioAcceptanceEngine, PortfolioLedger,
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


def _make_candidate(symbol: str, entry=1000.0, stop=950.0, score=65.0) -> AcceptanceCandidate:
    sector_map = {"RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking"}
    return AcceptanceCandidate(
        symbol=symbol, sector=sector_map.get(symbol, "Unknown"),
        fused_score=score, entry_date=date.today(),
        entry_price=entry, stop_loss=stop,
    )


def test_engine_applies_correlation_penalty():
    """
    Replaces the old standalone apply_cross_stock_correlation_penalty()
    test — exercises the same behavior through the consolidated engine.
    Accept RELIANCE first, then offer TCS: if they're correlated above
    threshold, TCS's effective risk should be grossed up by the penalty
    (visible as TCS's accepted candidate having effective_risk_pct >
    its raw position.risk_pct_portfolio).
    """
    print("\n── Test 3: Engine applies correlation penalty on 2nd accept ──")
    today = date.today()

    ledger = PortfolioLedger()
    engine = PortfolioAcceptanceEngine(ledger, correlation_fn=_pairwise_correlation)

    # Accept RELIANCE alone first (nothing to correlate against yet)
    r1 = engine.evaluate_batch(today, [_make_candidate("RELIANCE")])
    assert len(r1.accepted) == 1, "RELIANCE should be accepted with nothing else in portfolio"
    print(f"  RELIANCE accepted | effective_risk={r1.accepted[0].effective_risk_pct:.3%}")

    # Now offer TCS — engine checks correlation against RELIANCE (now in ledger)
    r2 = engine.evaluate_batch(today, [_make_candidate("TCS")])
    if not r2.accepted:
        print(f"  ⚠️  TCS rejected: {r2.rejected[0].reason if r2.rejected else 'unknown'}")
        print("  (expected if correlation data unavailable or risk cap hit first)")
        return False

    tcs = r2.accepted[0]
    raw_risk = tcs.position.risk_pct_portfolio
    print(f"  TCS accepted | raw_risk={raw_risk:.3%} | "
          f"effective_risk={tcs.effective_risk_pct:.3%}")
    if tcs.effective_risk_pct > raw_risk:
        print(f"  ✅ Correlation penalty applied — effective risk grossed up "
              f"({tcs.effective_risk_pct/raw_risk - 1:.1%} increase)")
    else:
        print(f"  ℹ️  No correlation penalty fired (RELIANCE/TCS not correlated "
              f"above {CROSS_STOCK_CORR_THRESHOLD} threshold, or insufficient data)")
    return True


def test_empty_ledger_is_noop():
    print("\n── Test 4: Empty ledger → zero correlation penalty (no-op) ──")
    ledger = PortfolioLedger()
    engine = PortfolioAcceptanceEngine(ledger, correlation_fn=_pairwise_correlation)

    result = engine.evaluate_batch(date.today(), [_make_candidate("RELIANCE")])
    assert len(result.accepted) == 1
    assert result.accepted[0].effective_risk_pct == result.accepted[0].position.risk_pct_portfolio, (
        "First acceptance into an empty ledger should have zero correlation penalty"
    )
    print("  ✅ Empty ledger → first accept has zero correlation penalty, safe no-op")


def main():
    print(f"\n{'='*60}")
    print("  Portfolio Acceptance Engine — Cross-Stock Correlation (§23.2) Test")
    print(f"{'='*60}")

    test_penalty_formula()
    has_data = test_real_pairwise_correlation()
    test_engine_applies_correlation_penalty()
    test_empty_ledger_is_noop()

    print(f"\n{'='*60}")
    if has_data:
        print("  ✅ Cross-Stock Correlation Penalty verified (real data)")
    else:
        print("  ⚠️  Verified formula + no-op paths only — ingest data for full test")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
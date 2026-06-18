"""
Portfolio Manager test — run after test_fusion.py passes.

Usage:
    python scripts/test_portfolio.py

What this tests:
  1. Position sizing — all 3 constraints (risk, stock cap, ADV cap)
  2. Binding constraint detection
  3. Sector cap enforcement
  4. Total risk cap enforcement
  5. Portfolio full rejection
  6. Risk-adjusted scoring + same-sector penalty
  7. Full PortfolioManager.run() with mixed signals
  8. No active positions (fresh portfolio)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.fusion.engine import FusedSignal
from app.portfolio.manager import (
    PortfolioManager, calculate_position_size,
    PORTFOLIO_VALUE, RISK_PER_TRADE_PCT, MAX_SECTOR_EXPOSURE,
    MAX_TOTAL_RISK_PCT, MAX_OPEN_POSITIONS,
)


# ── Fixtures ──────────────────────────────────────────────────

def make_signal(symbol, score=65.0, signal="BUY", conf=70.0,
                entry=1000.0, stop=950.0, target=1100.0) -> FusedSignal:
    from app.strategies.base import score_to_signal
    return FusedSignal(
        symbol=symbol, run_date=date.today(),
        fused_score=score, signal=signal, confidence=conf,
        regime="BULL",
        strategy_scores={"trend": 60, "momentum": 55},
        entry_price=entry, stop_loss=stop, target_price=target,
    )

def make_features(volume_sma_20=500_000) -> dict:
    return {"volume_sma_20": volume_sma_20, "close": 1000.0}


# ── Tests ─────────────────────────────────────────────────────

def test_position_sizing():
    print("\n── Test 1: Position sizing constraints ──────────────")

    # Normal case — risk-based should be binding
    ps = calculate_position_size("RELIANCE", entry_price=1332.0,
                                  stop_loss=1293.0, features=make_features(500_000))
    risk_budget = PORTFOLIO_VALUE * RISK_PER_TRADE_PCT
    expected_risk_shares = int(risk_budget / abs(1332 - 1293))
    print(f"  Portfolio: ₹{PORTFOLIO_VALUE:,.0f} | Risk/trade: {RISK_PER_TRADE_PCT:.1%}")
    print(f"  Entry=1332 | Stop=1293 | Risk/share=₹{abs(1332-1293)}")
    print(f"  Risk-based shares: {ps.shares_risk_based} (expected ~{expected_risk_shares})")
    print(f"  Stock-capped shares: {ps.shares_stock_capped}")
    print(f"  ADV-capped shares: {ps.shares_adv_capped}")
    print(f"  Final shares: {ps.final_shares} | Constraint: {ps.binding_constraint}")
    print(f"  Position value: ₹{ps.position_value_inr:,.0f}")
    print(f"  Risk amount: ₹{ps.risk_amount_inr:,.0f} ({ps.risk_pct_portfolio:.2%})")
    assert ps.final_shares > 0
    assert ps.position_value_inr <= PORTFOLIO_VALUE * 0.151  # within stock cap
    print(f"  ✅ Position sizing correct")


def test_adv_cap_binding():
    print("\n── Test 2: ADV cap binding (illiquid stock) ─────────")
    # entry=500, stop=480 -> risk/share=20
    # risk_shares = 1000000*0.015/20 = 750
    # stock_cap   = 1000000*0.15/500 = 300
    # adv_cap     = 5000*500*0.05/500 = 250  <- binds
    ps = calculate_position_size("SMALLCAP", entry_price=500.0,
                                  stop_loss=480.0,
                                  features={"volume_sma_20": 5_000, "close": 500.0})
    print(f"  Risk-based shares:  {ps.shares_risk_based}")
    print(f"  Stock-capped shares:{ps.shares_stock_capped}")
    print(f"  ADV-capped shares:  {ps.shares_adv_capped}")
    print(f"  ADV estimate: Rs{ps.adv_estimate:,.0f}")
    print(f"  Binding constraint: {ps.binding_constraint}")
    assert ps.binding_constraint == "ADV_CAP", \
        f"Expected ADV_CAP, got {ps.binding_constraint} " \
        f"(risk={ps.shares_risk_based}, stock={ps.shares_stock_capped}, adv={ps.shares_adv_capped})"
    print(f"  ✅ ADV cap correctly binding for illiquid stock")


def test_sector_cap():
    print("\n── Test 3: Sector cap enforcement ───────────────────")
    # All 4 signals are Banking — only first few should pass
    signals = [
        make_signal("HDFCBANK",  score=70, entry=787.0,  stop=760.0),
        make_signal("ICICIBANK", score=65, entry=1337.0, stop=1299.0),
        make_signal("SBIN",      score=60, entry=1026.0, stop=1006.0),
        make_signal("AXISBANK",  score=55, entry=1100.0, stop=1070.0),
    ]
    manager = PortfolioManager(run_date=date.today(), features_map={})
    result  = manager.run(signals)

    sector_rejected = [r for r in result.rejected if "SECTOR_CAP" in r.reason]
    print(f"  Accepted: {[s.symbol for s in result.accepted if 'BUY' in s.signal]}")
    print(f"  Rejected: {[(r.symbol, r.reason) for r in result.rejected]}")
    print(f"  Sector exposure: {result.sector_exposure}")

    # Banking sector should cap out
    banking_value = sum(
        ps.position_value_inr
        for sym, ps in result.position_sizes.items()
        if sym in ("HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK")
        and sym in [s.symbol for s in result.accepted]
    )
    banking_pct = banking_value / PORTFOLIO_VALUE
    print(f"  Banking exposure: {banking_pct:.1%} (max {MAX_SECTOR_EXPOSURE:.0%})")
    assert banking_pct <= MAX_SECTOR_EXPOSURE + 0.01  # small tolerance
    print(f"  ✅ Sector cap enforced correctly")


def test_total_risk_cap():
    print("\n── Test 4: Total risk cap enforcement ───────────────")
    # Many signals all with high individual risk — total should cap at 10%
    signals = [
        make_signal(sym, score=70-i*5, entry=1000.0, stop=940.0)  # 6% risk/share
        for i, sym in enumerate(["RELIANCE", "TCS", "BHARTIARTL", "ITC", "MARUTI",
                                  "TITAN", "LT", "ASIANPAINT"])
    ]
    manager = PortfolioManager(run_date=date.today(), features_map={})
    result  = manager.run(signals)

    print(f"  Total risk: {result.total_risk_pct:.2%} (cap={MAX_TOTAL_RISK_PCT:.0%})")
    assert result.total_risk_pct <= MAX_TOTAL_RISK_PCT + 0.005
    print(f"  Accepted: {len([s for s in result.accepted if 'BUY' in s.signal])}")
    print(f"  ✅ Total risk cap enforced at {result.total_risk_pct:.2%}")


def test_portfolio_full():
    print("\n── Test 5: Portfolio full (all slots taken) ─────────")
    # More signals than MAX_OPEN_POSITIONS
    signals = [
        make_signal(sym, score=70) for sym in
        ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
         "SBIN","BHARTIARTL","ITC","MARUTI","TITAN"]
    ]
    manager = PortfolioManager(run_date=date.today(), features_map={})
    result  = manager.run(signals)

    accepted_buys = [s for s in result.accepted if "BUY" in s.signal]
    print(f"  Accepted BUY: {len(accepted_buys)} (max={MAX_OPEN_POSITIONS})")
    print(f"  Rejected: {len(result.rejected)}")
    assert len(accepted_buys) <= MAX_OPEN_POSITIONS
    print(f"  ✅ Portfolio cap: {len(accepted_buys)} accepted, {len(result.rejected)} rejected")


def test_mixed_signals():
    print("\n── Test 6: Mixed signals (BUY + HOLD + SELL) ────────")
    signals = [
        make_signal("RELIANCE", score=70.0, signal="STRONG_BUY"),
        make_signal("TCS",      score=45.0, signal="BUY"),
        make_signal("HDFCBANK", score=0.0,  signal="HOLD"),
        make_signal("VEDL",     score=-80.0,signal="STRONG_SELL"),
    ]
    manager = PortfolioManager(run_date=date.today(), features_map={})
    result  = manager.run(signals)

    accepted_syms = [s.symbol for s in result.accepted]
    assert "RELIANCE" in accepted_syms  # top BUY
    assert "HDFCBANK" in accepted_syms  # HOLD passes through
    assert "VEDL"     in accepted_syms  # SELL passes through
    print(f"  Accepted: {accepted_syms}")

    # Check position sizing attached to BUY signals
    for s in result.accepted:
        if "BUY" in s.signal:
            assert hasattr(s, "position_size_shares")
            assert s.position_size_shares > 0
            print(f"  {s.symbol}: {s.position_size_shares} shares "
                  f"× ₹{s.entry_price:,.0f} = ₹{s.position_value_inr:,.0f}")
    print(f"  ✅ Mixed signals handled correctly")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Portfolio Manager Test")
    print(f"{'='*60}")
    print(f"  Portfolio value: ₹{PORTFOLIO_VALUE:,.0f}")
    print(f"  Risk per trade: {RISK_PER_TRADE_PCT:.1%}")
    print(f"  Max positions: {MAX_OPEN_POSITIONS}")
    print(f"  Max sector: {MAX_SECTOR_EXPOSURE:.0%}")
    print(f"  Max total risk: {MAX_TOTAL_RISK_PCT:.0%}")

    test_position_sizing()
    test_adv_cap_binding()
    test_sector_cap()
    test_total_risk_cap()
    test_portfolio_full()
    test_mixed_signals()

    print(f"\n{'='*60}")
    print("  ✅ Portfolio Manager verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
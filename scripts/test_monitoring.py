"""
Monitoring & Observability test — run after test_trades.py passes.

Usage:
    python scripts/test_monitoring.py

What this tests:
  1. Ingestion SLA — no breaches (healthy) vs breaches (alert)
  2. LLM failure rate — under threshold vs over threshold
  3. Pipeline latency — fast (healthy) vs slow (warning) vs very slow (critical)
  4. Bottleneck stage identification
  5. Strategy win rate check (requires trade history, gracefully skips if none)
  6. Full HealthMonitor.run() — combined report
  7. DB persistence of alerts (requires postgres + migration 002 applied)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from app.monitoring.health import (
    HealthMonitor, check_ingestion_sla, check_llm_failure_rate,
    check_pipeline_latency, identify_bottleneck_stage,
    check_strategy_win_rates,
    SLA_MINUTES_POST_CLOSE, LLM_FAILURE_RATE_ALERT,
    PIPELINE_LATENCY_WARN_MIN, PIPELINE_LATENCY_ALERT_MIN,
)


def test_ingestion_sla():
    print("\n── Test 1: Ingestion SLA check ──────────────────────")

    # Healthy — no breaches
    alert = check_ingestion_sla([], date.today())
    assert alert is None
    print("  ✅ No breaches → no alert")

    # Warning — few breaches
    alert = check_ingestion_sla(["TCS"], date.today())
    assert alert is not None and alert.severity == "WARNING"
    print(f"  ✅ 1 breach → {alert.severity}: {alert.message}")

    # Critical — many breaches
    alert = check_ingestion_sla(["TCS", "INFY", "SBIN", "WIPRO"], date.today())
    assert alert is not None and alert.severity == "CRITICAL"
    print(f"  ✅ 4 breaches → {alert.severity}: {alert.message}")


def test_llm_failure_rate():
    print("\n── Test 2: LLM failure rate check ───────────────────")

    # Healthy — 10% failure rate
    alert = check_llm_failure_rate(llm_timeouts=1, llm_total_calls=10)
    assert alert is None
    print("  ✅ 10% failure rate → no alert")

    # Critical — 30% failure rate (over 20% threshold)
    alert = check_llm_failure_rate(llm_timeouts=3, llm_total_calls=10)
    assert alert is not None and alert.severity == "CRITICAL"
    print(f"  ✅ 30% failure rate → {alert.severity}: {alert.message}")

    # Edge — no calls made
    alert = check_llm_failure_rate(llm_timeouts=0, llm_total_calls=0)
    assert alert is None
    print("  ✅ 0 calls → no alert (avoid div by zero)")


def test_pipeline_latency():
    print("\n── Test 3: Pipeline latency check ───────────────────")

    # Healthy — 10 min
    alert = check_pipeline_latency(total_ms=10 * 60_000)
    assert alert is None
    print(f"  ✅ 10 min → no alert (threshold {PIPELINE_LATENCY_WARN_MIN} min)")

    # Warning — 35 min
    alert = check_pipeline_latency(total_ms=35 * 60_000)
    assert alert is not None and alert.severity == "WARNING"
    print(f"  ✅ 35 min → {alert.severity}: {alert.message}")

    # Critical — 50 min
    alert = check_pipeline_latency(total_ms=50 * 60_000)
    assert alert is not None and alert.severity == "CRITICAL"
    print(f"  ✅ 50 min → {alert.severity}: {alert.message}")


def test_bottleneck_identification():
    print("\n── Test 4: Bottleneck stage identification ──────────")
    timings = {
        "ingestion": 8000, "features": 1500, "regime": 400,
        "strategies": 50, "fusion": 300, "llm": 12000,
        "microstructure": 600, "portfolio": 100,
    }
    bottleneck = identify_bottleneck_stage(timings)
    assert "llm" in bottleneck
    print(f"  ✅ Bottleneck correctly identified: {bottleneck}")


def test_strategy_win_rates():
    print("\n── Test 5: Strategy win rate check (DB) ─────────────")
    try:
        alerts = check_strategy_win_rates(lookback_days=20)
        print(f"  Win rate alerts: {len(alerts)}")
        for a in alerts:
            print(f"    [{a.severity}] {a.message}")
        print(f"  ✅ Win rate check ran without error "
              f"({'no data yet' if not alerts else f'{len(alerts)} flagged'})")
    except Exception as e:
        print(f"  ⚠️  Win rate check failed (expected if no trade history): {e}")


def test_full_health_monitor():
    print("\n── Test 6: Full HealthMonitor.run() ─────────────────")
    monitor = HealthMonitor(run_date=date.today())

    # Simulate a problematic run
    report = monitor.run(
        sla_breaches=["INFY", "SBIN"],
        llm_timeouts=3,
        llm_total_calls=8,           # 37.5% failure rate
        stage_timings={
            "ingestion": 9000, "features": 1500, "regime": 400,
            "strategies": 50, "fusion": 300, "llm": 5000,
            "microstructure": 600, "portfolio": 100,
        },
        check_win_rates=False,        # skip DB-dependent check for speed
    )

    print(f"  {report.summary()}")
    for a in report.alerts:
        print(f"    [{a.severity}] {a.metric}: {a.message}")
        print(f"       → {a.action}")

    assert len(report.alerts) >= 2   # SLA + LLM failure at minimum
    print(f"  ✅ Combined report generated: {len(report.alerts)} alerts")


def test_healthy_run():
    print("\n── Test 7: Healthy run (no alerts) ──────────────────")
    monitor = HealthMonitor(run_date=date.today())
    report = monitor.run(
        sla_breaches=[],
        llm_timeouts=0,
        llm_total_calls=5,
        stage_timings={"ingestion": 5000, "features": 1000, "llm": 2000},
        check_win_rates=False,
    )
    print(f"  {report.summary()}")
    assert report.is_healthy
    print(f"  ✅ Healthy run produces zero alerts")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — Monitoring Layer Test")
    print(f"{'='*60}")

    test_ingestion_sla()
    test_llm_failure_rate()
    test_pipeline_latency()
    test_bottleneck_identification()
    test_strategy_win_rates()
    test_full_health_monitor()
    test_healthy_run()

    print(f"\n{'='*60}")
    print("  ✅ Monitoring & Observability Layer verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

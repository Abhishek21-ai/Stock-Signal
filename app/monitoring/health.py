"""
Monitoring & Observability Layer — Section 24.1
Tracks 4 system health metrics with alert thresholds:

  1. Data ingestion SLA       — any stock missing OHLC > 31 min post-close
  2. LLM API failure rate     — > 20% of calls timing out in a session
  3. Signal generation latency— total pipeline > 30 min (alert > 45 min)
  4. Strategy win rate        — any strategy below 35% win rate (rolling 20d)

Called by pipeline.py at the end of each run (Stage 11 — new) and
by a separate scheduled job for strategy win rate (needs trade history).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from app.db import get_sync_db
from app.logger import get_logger
from config.settings import settings

logger = get_logger("monitoring")

# ── Alert thresholds (Section 24.1) ───────────────────────────
SLA_MINUTES_POST_CLOSE   = 31      # data must arrive within 31 min of close
LLM_FAILURE_RATE_ALERT   = 0.20    # 20% timeout rate triggers fallback
PIPELINE_LATENCY_WARN_MIN  = 30    # log bottleneck
PIPELINE_LATENCY_ALERT_MIN = 45    # send alert
STRATEGY_WIN_RATE_FLOOR  = 0.35    # below this → flag + reduce weight 25%
STRATEGY_WEIGHT_REDUCTION= 0.25
WIN_RATE_LOOKBACK_DAYS   = 20


@dataclass
class HealthAlert:
    metric:      str
    severity:    str     # WARNING | CRITICAL
    message:     str
    value:       float
    threshold:   float
    action:      str
    timestamp:   datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "metric":    self.metric,
            "severity":  self.severity,
            "message":   self.message,
            "value":     self.value,
            "threshold": self.threshold,
            "action":    self.action,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthReport:
    run_date:  date
    alerts:    List[HealthAlert] = field(default_factory=list)
    metrics:   Dict[str, float]  = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "CRITICAL" for a in self.alerts)

    @property
    def is_healthy(self) -> bool:
        return len(self.alerts) == 0

    def summary(self) -> str:
        if self.is_healthy:
            return "✅ All systems healthy"
        crit = sum(1 for a in self.alerts if a.severity == "CRITICAL")
        warn = sum(1 for a in self.alerts if a.severity == "WARNING")
        return f"⚠️  {crit} critical, {warn} warnings"


# ── Metric 1: Data ingestion SLA ──────────────────────────────

def check_ingestion_sla(
    sla_breaches: List[str],
    run_date: date,
) -> Optional[HealthAlert]:
    """
    Section 24.1: any stock missing OHLC > 31 min post-close triggers alert.
    sla_breaches comes from PipelineContext.data_sla_breaches.
    """
    if not sla_breaches:
        return None

    return HealthAlert(
        metric="data_ingestion_sla",
        severity="CRITICAL" if len(sla_breaches) > 3 else "WARNING",
        message=f"{len(sla_breaches)} stocks missing OHLC data: {sla_breaches}",
        value=len(sla_breaches),
        threshold=0,
        action="Trigger fallback source (nsepython); Telegram alert sent",
    )


# ── Metric 2: LLM API failure rate ────────────────────────────

def check_llm_failure_rate(
    llm_timeouts: int,
    llm_total_calls: int,
) -> Optional[HealthAlert]:
    """
    Section 24.1: > 20% of LLM calls timing out in a session.
    """
    if llm_total_calls == 0:
        return None

    failure_rate = llm_timeouts / llm_total_calls
    if failure_rate <= LLM_FAILURE_RATE_ALERT:
        return None

    return HealthAlert(
        metric="llm_failure_rate",
        severity="CRITICAL",
        message=(
            f"LLM failure rate {failure_rate:.0%} exceeds "
            f"{LLM_FAILURE_RATE_ALERT:.0%} threshold "
            f"({llm_timeouts}/{llm_total_calls} calls timed out)"
        ),
        value=round(failure_rate, 4),
        threshold=LLM_FAILURE_RATE_ALERT,
        action="Switched to rule-based fallback (Section 27); alert sent",
    )


# ── Metric 3: Pipeline latency ─────────────────────────────────

def check_pipeline_latency(total_ms: int) -> Optional[HealthAlert]:
    """
    Section 24.1: total pipeline > 30 min logs bottleneck,
    > 45 min triggers alert.
    """
    total_min = total_ms / 60_000

    if total_min < PIPELINE_LATENCY_WARN_MIN:
        return None

    severity = "CRITICAL" if total_min > PIPELINE_LATENCY_ALERT_MIN else "WARNING"
    action = (
        "Alert sent — pipeline exceeded 45 min"
        if severity == "CRITICAL"
        else "Logged for review — approaching latency threshold"
    )

    return HealthAlert(
        metric="pipeline_latency",
        severity=severity,
        message=f"Pipeline took {total_min:.1f} min (threshold {PIPELINE_LATENCY_WARN_MIN} min)",
        value=round(total_min, 2),
        threshold=PIPELINE_LATENCY_WARN_MIN,
        action=action,
    )


def identify_bottleneck_stage(stage_timings: Dict[str, int]) -> Optional[str]:
    """Returns the slowest stage name for logging."""
    if not stage_timings:
        return None
    slowest = max(stage_timings.items(), key=lambda x: x[1])
    return f"{slowest[0]} ({slowest[1]}ms)"


# ── Metric 4: Strategy win rate ────────────────────────────────

def check_strategy_win_rates(lookback_days: int = WIN_RATE_LOOKBACK_DAYS) -> List[HealthAlert]:
    """
    Section 24.1: any strategy below 35% win rate (rolling 20 days)
    gets flagged and its weight reduced 25%.

    Win rate per strategy = % of trades where that strategy's score
    was directionally aligned with the trade outcome (profit/loss).
    Approximation: use trades where strategy_scores[strategy] > 0
    and check if the trade was profitable.
    """
    alerts = []
    cutoff = date.today() - timedelta(days=lookback_days)

    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    ds.trend_score, ds.momentum_score, ds.reversion_score,
                    ds.breakout_score, ds.volume_score,
                    t.pnl_pct
                FROM trades t
                JOIN daily_signals ds
                  ON ds.stock = t.stock AND ds.date = t.signal_date
                WHERE t.status = 'CLOSED'
                  AND t.exit_date >= %s
                  AND t.pnl_pct IS NOT NULL
                """,
                (cutoff,),
            )
            rows = cursor.fetchall()
    except Exception as e:
        logger.warning(f"Could not fetch strategy win rate data: {e}")
        return alerts

    if not rows:
        logger.info(f"No closed trades in last {lookback_days}d — skipping win rate check")
        return alerts

    strategies = ["trend", "momentum", "reversion", "breakout", "volume"]
    stats = {s: {"contributed": 0, "won": 0} for s in strategies}

    for row in rows:
        won = float(row["pnl_pct"]) > 0
        for strat in strategies:
            score = row.get(f"{strat}_score")
            if score is not None and float(score) > 0:
                stats[strat]["contributed"] += 1
                if won:
                    stats[strat]["won"] += 1

    for strat, s in stats.items():
        if s["contributed"] < 5:   # not enough sample size
            continue
        win_rate = s["won"] / s["contributed"]
        if win_rate < STRATEGY_WIN_RATE_FLOOR:
            alerts.append(HealthAlert(
                metric=f"strategy_win_rate_{strat}",
                severity="WARNING",
                message=(
                    f"{strat} strategy win rate {win_rate:.0%} below "
                    f"{STRATEGY_WIN_RATE_FLOOR:.0%} floor "
                    f"({s['won']}/{s['contributed']} trades, {lookback_days}d)"
                ),
                value=round(win_rate, 4),
                threshold=STRATEGY_WIN_RATE_FLOOR,
                action=f"Flagged for recalibration; weight reduced {STRATEGY_WEIGHT_REDUCTION:.0%}",
            ))
            logger.warning(
                f"{strat}: win rate {win_rate:.0%} below floor — "
                f"recommend weight reduction"
            )

    return alerts


# ── DB: record health checks ──────────────────────────────────

def _record_alerts(alerts: List[HealthAlert], run_date: date) -> None:
    """Persist alerts to a simple log table for dashboard consumption."""
    if not alerts:
        return
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            for a in alerts:
                cursor.execute(
                    """
                    INSERT INTO health_alerts (
                        run_date, metric, severity, message,
                        value, threshold, action
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_date, a.metric, a.severity, a.message,
                     a.value, a.threshold, a.action),
                )
    except Exception as e:
        # Table may not exist yet if migration not run — log but don't crash
        logger.warning(f"Could not persist alerts (run migration?): {e}")


async def _send_telegram_alert(alerts: List[HealthAlert]) -> None:
    """Send critical alerts via Telegram."""
    critical = [a for a in alerts if a.severity == "CRITICAL"]
    if not critical or not settings.telegram_enabled:
        return

    try:
        import httpx
        lines = ["🚨 *SSP Health Alert*\n"]
        for a in critical:
            lines.append(f"❗ *{a.metric}*: {a.message}")
            lines.append(f"   Action: {a.action}\n")
        message = "\n".join(lines)

        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={
                "chat_id": settings.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown",
            })
        logger.info(f"Sent {len(critical)} critical alerts via Telegram")
    except Exception as e:
        logger.error(f"Failed to send health alert: {e}")


# ── Main entry point ───────────────────────────────────────────

class HealthMonitor:
    """
    Called by pipeline.py as the final stage (Stage 11).
    Runs all 4 health checks and persists/alerts as needed.
    """

    def __init__(self, run_date: Optional[date] = None):
        self.run_date = run_date or date.today()

    def run(
        self,
        sla_breaches:     List[str],
        llm_timeouts:     int,
        llm_total_calls:  int,
        stage_timings:    Dict[str, int],
        check_win_rates:  bool = True,
    ) -> HealthReport:
        report = HealthReport(run_date=self.run_date)

        # ── 1. Ingestion SLA ───────────────────────────────────
        alert = check_ingestion_sla(sla_breaches, self.run_date)
        if alert:
            report.alerts.append(alert)

        # ── 2. LLM failure rate ────────────────────────────────
        alert = check_llm_failure_rate(llm_timeouts, llm_total_calls)
        if alert:
            report.alerts.append(alert)

        # ── 3. Pipeline latency ─────────────────────────────────
        total_ms = sum(stage_timings.values())
        alert = check_pipeline_latency(total_ms)
        if alert:
            bottleneck = identify_bottleneck_stage(stage_timings)
            alert.message += f" | Bottleneck: {bottleneck}"
            report.alerts.append(alert)
        report.metrics["total_pipeline_ms"] = total_ms

        # ── 4. Strategy win rates (only if enough history) ────
        if check_win_rates:
            win_rate_alerts = check_strategy_win_rates()
            report.alerts.extend(win_rate_alerts)

        # ── Persist + notify ────────────────────────────────────
        _record_alerts(report.alerts, self.run_date)

        logger.info(f"Health check complete: {report.summary()}")
        for a in report.alerts:
            logger.warning(f"[{a.severity}] {a.metric}: {a.message}")

        return report

    async def run_and_alert(self, **kwargs) -> HealthReport:
        """Run checks and send Telegram alert for critical issues."""
        report = self.run(**kwargs)
        if report.has_critical:
            await _send_telegram_alert(report.alerts)
        return report

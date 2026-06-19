"""
Rule-Based LLM Fallback — Section 27.1
Deterministic replication of the most critical LLM override behaviors,
used when Groq + Gemini + OpenAI have all failed (or session failure
rate exceeds the 20% threshold from Section 24.1).

Rule table (Section 27.1):
  1. Earnings within 3 days        → Override to HOLD regardless of quant score
  2. Realized volatility > 2x avg  → Reduce confidence by 20%
  3. 3+ BUY signals in same sector → Downgrade 3rd+ BUY in that sector to HOLD
  4. Macro event within 2 days     → Reduce confidence by 20%, flag macro_window=True

These rules require no API call — they read directly from the feature
vector and daily_signals table, making them instant and free.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

from app.db import get_sync_db
from app.fusion.engine import FusedSignal
from app.logger import get_logger
from app.regime.detector import RegimeResult
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("llm_fallback")

# ── Thresholds (Section 27.1) ─────────────────────────────────
EARNINGS_WINDOW_DAYS      = 3
VOLATILITY_MULTIPLE       = 2.0    # realized vol > 2x 6-month avg
MACRO_EVENT_WINDOW_DAYS   = 2
MACRO_CONFIDENCE_PENALTY  = 20.0
VOLATILITY_CONFIDENCE_PENALTY = 20.0
SECTOR_BUY_LIMIT          = 2      # 3rd+ BUY in same sector gets downgraded


def _check_earnings_window(features: Dict) -> Optional[str]:
    """
    Rule 1: Earnings within 3 days → HOLD override.
    Reads event_flag / days_to_earnings from feature vector if present.
    Returns reason string if triggered, else None.
    """
    days_to_earnings = features.get("days_to_earnings")
    event_flag       = features.get("event_flag")

    if days_to_earnings is not None:
        try:
            if 0 <= int(days_to_earnings) <= EARNINGS_WINDOW_DAYS:
                return f"Earnings in {days_to_earnings}d — override to HOLD"
        except (ValueError, TypeError):
            pass

    if event_flag == "EARNINGS":
        return "Earnings event flagged — override to HOLD"

    return None


def _check_volatility(features: Dict) -> Optional[str]:
    """
    Rule 2: Realized volatility > 2x 6-month average → reduce confidence 20%.
    Uses atr_pct vs a longer-window average if available, else skip.
    """
    atr_pct      = features.get("atr_pct")
    atr_pct_6m_avg = features.get("atr_pct_6m_avg")  # may not exist yet

    if atr_pct is not None and atr_pct_6m_avg:
        try:
            if float(atr_pct) > VOLATILITY_MULTIPLE * float(atr_pct_6m_avg):
                return (
                    f"Volatility {atr_pct:.1f}% > {VOLATILITY_MULTIPLE}x "
                    f"6mo avg ({atr_pct_6m_avg:.1f}%) — confidence reduced"
                )
        except (ValueError, TypeError):
            pass

    return None


def _check_sector_concentration(symbol: str, run_date: date) -> Optional[str]:
    """
    Rule 3: 3+ BUY signals in same sector today → downgrade 3rd+ to HOLD.
    Queries today's daily_signals for sector peers already marked BUY.
    """
    sector = WATCHLIST_WITH_SECTORS.get(symbol, "Unknown")
    if sector == "Unknown":
        return None

    sector_peers = [
        s for s, sec in WATCHLIST_WITH_SECTORS.items()
        if sec == sector and s != symbol
    ]
    if not sector_peers:
        return None

    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT stock FROM daily_signals
                WHERE date = %s
                  AND signal IN ('BUY', 'STRONG_BUY')
                  AND stock = ANY(%s)
                ORDER BY quant_score DESC
                """,
                (run_date, sector_peers),
            )
            existing_buys = [r["stock"] for r in cursor.fetchall()]
    except Exception as e:
        logger.warning(f"Sector concentration check failed for {symbol}: {e}")
        return None

    # If this would be the 3rd+ BUY in the sector today, downgrade
    if len(existing_buys) >= SECTOR_BUY_LIMIT:
        return (
            f"{sector} sector already has {len(existing_buys)} BUY signals "
            f"today ({existing_buys[:3]}) — downgrading to HOLD"
        )

    return None


def _check_macro_window(run_date: date) -> Optional[str]:
    """
    Rule 4: Macro event within 2 days → reduce confidence 20%, flag macro_window.
    """
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT event_name, event_date FROM macro_events
                WHERE event_date BETWEEN %s AND %s
                  AND expected_impact = 'HIGH'
                ORDER BY event_date
                LIMIT 1
                """,
                (run_date, run_date + timedelta(days=MACRO_EVENT_WINDOW_DAYS)),
            )
            row = cursor.fetchone()
            if row:
                return (
                    f"Macro event '{row['event_name']}' on {row['event_date']} "
                    f"within {MACRO_EVENT_WINDOW_DAYS}d — confidence reduced, macro_window=True"
                )
    except Exception as e:
        logger.warning(f"Macro window check failed: {e}")
    return None


def apply_rule_based_fallback(
    signal: FusedSignal,
    regime: RegimeResult,
    features: Dict,
):
    """
    Main entry point. Runs all 4 rules in order and returns an LLMVerdict-
    compatible object. Import is deferred to avoid circular import with
    override.py.
    """
    from app.llm.override import LLMVerdict   # deferred import

    triggered_rules = []

    # ── Rule 1: Earnings window — hard override to HOLD ────────
    earnings_reason = _check_earnings_window(features)
    if earnings_reason:
        logger.info(f"{signal.symbol}: rule-based VETO — {earnings_reason}")
        return LLMVerdict(
            verdict="VETO",
            explanation=f"[Rule-based] {earnings_reason}",
            llm_status="RULE_BASED",
            provider="rule_based",
        )

    # ── Rule 3: Sector concentration — hard override to HOLD ───
    sector_reason = _check_sector_concentration(signal.symbol, signal.run_date)
    if sector_reason:
        logger.info(f"{signal.symbol}: rule-based VETO — {sector_reason}")
        return LLMVerdict(
            verdict="VETO",
            explanation=f"[Rule-based] {sector_reason}",
            llm_status="RULE_BASED",
            provider="rule_based",
        )

    # ── Rule 2: Volatility — soft penalty ───────────────────────
    vol_reason = _check_volatility(features)
    if vol_reason:
        triggered_rules.append(vol_reason)

    # ── Rule 4: Macro event — soft penalty ──────────────────────
    macro_reason = _check_macro_window(signal.run_date)
    if macro_reason:
        triggered_rules.append(macro_reason)

    if triggered_rules:
        logger.info(
            f"{signal.symbol}: rule-based REDUCE_CONFIDENCE — "
            f"{'; '.join(triggered_rules)}"
        )
        return LLMVerdict(
            verdict="REDUCE_CONFIDENCE",
            explanation=f"[Rule-based] {'; '.join(triggered_rules)}",
            llm_status="RULE_BASED",
            provider="rule_based",
        )

    # ── No rules triggered → CONFIRM ────────────────────────────
    return LLMVerdict(
        verdict="CONFIRM",
        explanation="[Rule-based] No red flags detected — quant signal stands",
        llm_status="RULE_BASED",
        provider="rule_based",
    )

"""
DailyPipeline — full wired implementation.
Section 4: Data → Features → Regime → Strategies → Fusion → LLM →
           Microstructure → Execution Realism → Portfolio → Notifications

Run manually:
    python scripts/run_pipeline.py

Or via scheduler at 15:45 IST daily.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from app.logger import get_logger
from config.settings import settings

logger = get_logger("pipeline")


@dataclass
class PipelineContext:
    run_date:          date  = field(default_factory=date.today)
    run_uuid:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    stocks:            List[str] = field(default_factory=list)
    skipped_stocks:    List[str] = field(default_factory=list)
    data_sla_breaches: List[str] = field(default_factory=list)
    stage_timings:     Dict[str, int] = field(default_factory=dict)
    signals_generated: int = 0
    llm_overrides:     int = 0
    llm_timeouts:      int = 0
    errors:            List[str] = field(default_factory=list)

    # Inter-stage data (passed between stages via context)
    raw_data:          Dict = field(default_factory=dict)   # {symbol: DataFrame}
    features:          Dict = field(default_factory=dict)   # {symbol: feature_dict}
    regime:            Optional[object] = None              # RegimeResult
    strategy_results:  Dict = field(default_factory=dict)   # {symbol: [StrategyResult]}
    fused_signals:     List = field(default_factory=list)   # [FusedSignal]
    final_signals:     List = field(default_factory=list)   # after LLM + microstructure


class DailyPipeline:

    def __init__(self, stocks: Optional[List[str]] = None, run_date: Optional[date] = None):
        self.ctx = PipelineContext(
            stocks=stocks or list(settings.watchlist),
            run_date=run_date or date.today(),
        )

    async def run(self) -> PipelineContext:
        started_at = datetime.utcnow()
        status = "SUCCESS"

        try:
            # ── Gate: NSE holiday check ───────────────────────────
            if await self._is_market_closed():
                logger.info(f"NSE holiday on {self.ctx.run_date} — pipeline skipped")
                await self._record_run(started_at, "MARKET_CLOSED")
                return self.ctx

            logger.info(
                f"Pipeline start | date={self.ctx.run_date} | "
                f"stocks={len(self.ctx.stocks)} | uuid={self.ctx.run_uuid}"
            )

            await self._timed_stage("ingestion",         self._run_ingestion)
            await self._timed_stage("features",          self._run_features)
            await self._timed_stage("regime",            self._run_regime)
            await self._timed_stage("strategies",        self._run_strategies)
            await self._timed_stage("fusion",            self._run_fusion)
            await self._timed_stage("correlation",       self._run_correlation)
            await self._timed_stage("llm",               self._run_llm)
            await self._timed_stage("microstructure",    self._run_microstructure)
            await self._timed_stage("execution_realism", self._run_execution_realism)
            await self._timed_stage("portfolio",         self._run_portfolio)
            await self._timed_stage("notifications",     self._run_notifications)
            await self._timed_stage("monitoring",        self._run_monitoring)

        except Exception as e:
            status = "FAILED"
            self.ctx.errors.append(str(e))
            logger.exception(f"Pipeline failed: {e}")

        finally:
            total_ms = sum(self.ctx.stage_timings.values())
            logger.info(
                f"Pipeline complete | status={status} | "
                f"signals={self.ctx.signals_generated} | total={total_ms}ms"
            )
            await self._record_run(started_at, status)

        return self.ctx

    # ── Stage 1: Data Ingestion ───────────────────────────────

    async def _run_ingestion(self) -> None:
        from app.data.ingestor import DataIngestor
        ingestor = DataIngestor(
            stocks=self.ctx.stocks,
            run_date=self.ctx.run_date,
        )
        # Run sync ingestor in thread pool (yfinance is blocking)
        loop = asyncio.get_event_loop()
        self.ctx.raw_data = await loop.run_in_executor(None, ingestor.run)

        self.ctx.data_sla_breaches = ingestor.sla_breaches
        self.ctx.skipped_stocks    = ingestor.skipped

        # Remove stocks with no data from pipeline
        active = list(self.ctx.raw_data.keys())
        logger.info(
            f"Ingestion: {len(active)} stocks OK | "
            f"skipped={len(self.ctx.skipped_stocks)} | "
            f"sla_breaches={len(self.ctx.data_sla_breaches)}"
        )

    # ── Stage 2: Feature Engineering ─────────────────────────

    async def _run_features(self) -> None:
        from app.features.engineer import FeatureEngineer
        engineer = FeatureEngineer()
        loop = asyncio.get_event_loop()
        self.ctx.features = await loop.run_in_executor(
            None,
            engineer.run,
            list(self.ctx.raw_data.keys()),
            self.ctx.run_date,
        )
        logger.info(f"Features: computed for {len(self.ctx.features)} stocks")

    # ── Stage 3: Regime Detection ─────────────────────────────

    async def _run_regime(self) -> None:
        from app.regime.detector import RegimeDetector
        detector = RegimeDetector(run_date=self.ctx.run_date)
        loop = asyncio.get_event_loop()
        self.ctx.regime = await loop.run_in_executor(None, detector.run)
        logger.info(
            f"Regime: {self.ctx.regime.regime} ({self.ctx.regime.regime_confidence}) | "
            f"Nifty={self.ctx.regime.nifty_close:.0f} ADX={self.ctx.regime.nifty_adx:.1f}"
        )

    # ── Stage 4: Strategy Engines ────────────────────────────

    async def _run_strategies(self) -> None:
        from app.strategies.runner import StrategyRunner
        runner  = StrategyRunner()
        regime  = self.ctx.regime.regime if self.ctx.regime else "UNCERTAIN"

        loop    = asyncio.get_event_loop()

        def _run_all():
            results = {}
            for symbol, features in self.ctx.features.items():
                results[symbol] = runner.run(features, regime=regime)
            return results

        self.ctx.strategy_results = await loop.run_in_executor(None, _run_all)
        logger.info(f"Strategies: ran for {len(self.ctx.strategy_results)} stocks")

    # ── Stage 5: Signal Fusion ────────────────────────────────

    async def _run_fusion(self) -> None:
        from app.fusion.engine import FusionEngine
        if not self.ctx.regime:
            from app.regime.detector import RegimeResult, REGIME_WEIGHTS
            self.ctx.regime = RegimeResult(
                regime="UNCERTAIN", regime_confidence="NORMAL",
                fusion_weights=dict(REGIME_WEIGHTS["UNCERTAIN"]),
            )

        engine = FusionEngine(
            regime_result=self.ctx.regime,
            run_date=self.ctx.run_date,
        )
        loop = asyncio.get_event_loop()
        self.ctx.fused_signals = await loop.run_in_executor(
            None,
            lambda: engine.run(
                self.ctx.strategy_results,
                all_features=self.ctx.features,
                save_to_db=True,
            ),
        )
        self.ctx.signals_generated = len(self.ctx.fused_signals)
        logger.info(
            f"Fusion: {self.ctx.signals_generated} signals | "
            f"BUY={sum(1 for s in self.ctx.fused_signals if 'BUY' in s.signal)} | "
            f"SELL={sum(1 for s in self.ctx.fused_signals if 'SELL' in s.signal)} | "
            f"HOLD={sum(1 for s in self.ctx.fused_signals if s.signal == 'HOLD')}"
        )

    # ── Stage 5b: Strategy Correlation Update ─────────────────

    async def _run_correlation(self) -> None:
        """
        Section 23: Recompute rolling 60-day strategy correlation matrix
        for each stock and store it. The Fusion stage that already ran
        this cycle uses YESTERDAY's matrix (correlations are inherently
        backward-looking); this stage refreshes today's matrix for
        TOMORROW's fusion run.
        """
        from app.correlation.engine import CorrelationEngine

        def _run():
            engine = CorrelationEngine(run_date=self.ctx.run_date)
            return engine.run(self.ctx.stocks)

        results = await asyncio.get_event_loop().run_in_executor(None, _run)
        updated = sum(1 for v in results.values() if v)
        logger.info(
            f"Correlation: {updated}/{len(self.ctx.stocks)} matrices updated "
            f"(others have <60d history)"
        )


    # ── Stage 6: LLM Override ────────────────────────────────

    async def _run_llm(self) -> None:
        from app.llm.override import LLMOverrideEngine
        engine = LLMOverrideEngine(
            regime=self.ctx.regime,
            features_map=self.ctx.features,
            run_date=self.ctx.run_date,
        )
        self.ctx.fused_signals = await engine.run(self.ctx.fused_signals)

        # Read exact count from engine instead of string-matching `reasons`,
        # which is fragile (other stages can coincidentally contain similar text).
        self.ctx.llm_overrides = engine.override_count
        logger.info(f"LLM: {self.ctx.llm_overrides} overrides")

    # ── Stage 7: Microstructure Filters ──────────────────────

    async def _run_microstructure(self) -> None:
        from app.data.microstructure import apply_microstructure_filters

        loop = asyncio.get_event_loop()

        def _apply_all():
            for signal in self.ctx.fused_signals:
                features = self.ctx.features.get(signal.symbol, {})
                confidence = signal.confidence
                result = apply_microstructure_filters(
                    signal.symbol, self.ctx.run_date, confidence
                )
                if result["suppress_signal"]:
                    signal.signal    = "HOLD"
                    signal.fused_score = 0.0
                    signal.reasons.append(
                        f"Microstructure suppressed: circuit={result['circuit_hit']} "
                        f"gap={result['gap_open_type']} pledge={result['pledge_risk_flag']}"
                    )
                elif result["adjusted_confidence"] != confidence:
                    signal.confidence = result["adjusted_confidence"]
                    signal.reasons.append(
                        f"Confidence adjusted by microstructure filters "
                        f"({confidence:.0f}% → {signal.confidence:.0f}%)"
                    )

        await loop.run_in_executor(None, _apply_all)
        suppressed = sum(1 for s in self.ctx.fused_signals if s.signal == "HOLD" and s.fused_score == 0)
        logger.info(f"Microstructure: {suppressed} signals suppressed")

    # ── Stage 8: Execution Realism ────────────────────────────

    async def _run_execution_realism(self) -> None:
        from app.data.execution_realism import apply_execution_realism

        loop = asyncio.get_event_loop()

        def _apply_all():
            for signal in self.ctx.fused_signals:
                if signal.signal == "HOLD" or not signal.entry_price:
                    continue
                features  = self.ctx.features.get(signal.symbol, {})
                close     = features.get("close", signal.entry_price)
                avg_vol   = features.get("volume_sma_20", 1_000_000)
                pos_value = (features.get("close", 100) *
                             _position_shares(signal, features))
                result = apply_execution_realism(
                    symbol=signal.symbol,
                    theoretical_entry=signal.entry_price,
                    theoretical_target=signal.target_price or signal.entry_price * 1.06,
                    theoretical_stop=signal.stop_loss or signal.entry_price * 0.97,
                    position_value_inr=pos_value,
                    avg_daily_volume=avg_vol,
                    close_price=close,
                )
                # Attach realistic prices to signal reasons for transparency
                signal.reasons.append(
                    f"Realistic entry ₹{result['entry_price_realistic']} | "
                    f"slip={result['slippage_factor_pct']:.3%} | "
                    f"R:R realistic={result['rr_ratio_realistic']}"
                )

        await loop.run_in_executor(None, _apply_all)
        logger.info("Execution realism: applied to all BUY/SELL signals")

    # ── Stage 9: Portfolio Constraints ───────────────────────
    async def _run_portfolio(self) -> None:
        """
        Section 21: Full portfolio management — position sizing,
        sector caps, total risk cap, signal prioritization.
        Replaces the simple position count cap.
        """
        from app.portfolio.manager import PortfolioManager
        from app.trades.lifecycle import create_pending_trades

        def _run():
            manager = PortfolioManager(
                run_date=self.ctx.run_date,
                features_map=self.ctx.features,
            )
            return manager.run(self.ctx.fused_signals)

        result = await asyncio.get_event_loop().run_in_executor(None, _run)

        self.ctx.final_signals    = result.accepted
        self.ctx.signals_generated = len(
            [s for s in result.accepted if "BUY" in s.signal]
        )

        # Create PENDING trade rows for accepted BUY signals
        create_pending_trades(result.accepted, self.ctx.run_date)

        if result.rejected:
            logger.info(
                f"Portfolio: {len(result.rejected)} signals rejected — "
                + " | ".join(f"{r.symbol}:{r.reason}" for r in result.rejected)
            )

        logger.info(
            f"Portfolio: {self.ctx.signals_generated} BUY signals accepted | "
            f"total_risk={result.total_risk_pct:.1%} | "
            f"slots_used={result.active_positions + self.ctx.signals_generated}"
        )

    # ── Stage 10: Notifications ───────────────────────────────

    async def _run_notifications(self) -> None:
        """
        Dispatch signals via Telegram.
        Full dispatcher: app/notifications/dispatcher.py (next build layer).
        """
        actionable = [s for s in self.ctx.fused_signals if s.signal != "HOLD"]
        if not actionable:
            logger.info("Notifications: no actionable signals to send")
            return

        if settings.telegram_enabled:
            try:
                from app.notifications.telegram import send_signal_digest
                await send_signal_digest(actionable, self.ctx.regime, self.ctx.run_date)
            except Exception as e:
                logger.warning(f"Telegram notification failed: {e}")
        else:
            logger.info(
                f"Notifications: Telegram not configured | "
                f"{len(actionable)} actionable signals ready"
            )

    # ── Stage 11: Monitoring ───────────────────────────────

    async def _run_monitoring(self) -> None:
        """
        Section 24: post-run health check.
        Failures here are logged but never fail the overall pipeline —
        observability should never be a single point of failure.
        """
        try:
            from app.monitoring.health import HealthMonitor
            monitor = HealthMonitor(run_date=self.ctx.run_date)
            await monitor.run_and_alert(
                sla_breaches=self.ctx.data_sla_breaches,
                llm_timeouts=self.ctx.llm_timeouts,
                llm_total_calls=self.ctx.llm_overrides + self.ctx.llm_timeouts,
                stage_timings=self.ctx.stage_timings,
            )
        except Exception as e:
            logger.warning(f"Monitoring stage failed (non-fatal): {e}")


    # ── Helpers ───────────────────────────────────────────────

    async def _is_market_closed(self) -> bool:
        from app.data.market_calendar import is_nse_holiday
        return await is_nse_holiday(self.ctx.run_date)

    async def _timed_stage(self, name: str, coro) -> None:
        t0 = time.monotonic()
        try:
            await coro()
        except Exception as e:
            logger.error(f"Stage [{name}] failed: {e}")
            self.ctx.errors.append(f"{name}: {e}")
            raise
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.ctx.stage_timings[name] = elapsed_ms
        logger.info(f"Stage [{name}] ✓ {elapsed_ms}ms")

    async def _record_run(self, started_at: datetime, status: str) -> None:
        try:
            from app.db import get_sync_db
            with get_sync_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_uuid, run_date, started_at, completed_at,
                        ingestion_ms, features_ms, regime_ms, strategies_ms,
                        fusion_ms, llm_ms, notifications_ms, total_ms,
                        stocks_processed, stocks_skipped, signals_generated,
                        llm_overrides, llm_timeouts, data_sla_breaches,
                        status, error_message
                    ) VALUES (
                        %s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s
                    )
                    ON CONFLICT (run_date) DO UPDATE SET
                        completed_at=NOW(), status=EXCLUDED.status,
                        signals_generated=EXCLUDED.signals_generated,
                        llm_overrides=EXCLUDED.llm_overrides,
                        total_ms=EXCLUDED.total_ms,
                        error_message=EXCLUDED.error_message
                    """,
                    (
                        self.ctx.run_uuid, self.ctx.run_date, started_at,
                        self.ctx.stage_timings.get("ingestion"),
                        self.ctx.stage_timings.get("features"),
                        self.ctx.stage_timings.get("regime"),
                        self.ctx.stage_timings.get("strategies"),
                        self.ctx.stage_timings.get("fusion"),
                        self.ctx.stage_timings.get("llm"),
                        self.ctx.stage_timings.get("notifications"),
                        sum(self.ctx.stage_timings.values()),
                        len(self.ctx.stocks) - len(self.ctx.skipped_stocks),
                        len(self.ctx.skipped_stocks),
                        self.ctx.signals_generated,
                        self.ctx.llm_overrides,
                        self.ctx.llm_timeouts,
                        self.ctx.data_sla_breaches,
                        status,
                        "; ".join(self.ctx.errors) if self.ctx.errors else None,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to record pipeline run: {e}")


def _position_shares(signal, features: dict) -> int:
    """Rough position size for execution realism calc."""
    close = features.get("close", 100)
    risk_inr = settings.portfolio_value_inr * settings.risk_per_trade_pct
    atr = features.get("atr_14", close * 0.02)
    if atr <= 0 or close <= 0:
        return 0
    shares = int(risk_inr / (1.5 * atr))
    max_shares = int(settings.portfolio_value_inr * settings.max_single_stock_pct / close)
    return min(shares, max_shares)
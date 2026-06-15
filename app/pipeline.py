"""
DailyPipeline — orchestrates all layers in order.
Section 4 architecture: Data → Features → Regime → Strategies → Fusion → LLM → Microstructure → Execution Realism → Portfolio → Notifications

Each stage is timed and recorded in pipeline_runs table (Section 24).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from app.db import get_async_db
from app.logger import get_logger
from config.settings import settings

logger = get_logger("pipeline")


@dataclass
class PipelineContext:
    """Shared state passed through all pipeline stages."""
    run_date: date = field(default_factory=date.today)
    run_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    stocks: List[str] = field(default_factory=list)
    skipped_stocks: List[str] = field(default_factory=list)
    data_sla_breaches: List[str] = field(default_factory=list)
    stage_timings: Dict[str, int] = field(default_factory=dict)   # ms per stage
    signals_generated: int = 0
    llm_overrides: int = 0
    llm_timeouts: int = 0
    errors: List[str] = field(default_factory=list)


class DailyPipeline:
    """
    Runs the full daily signal generation pipeline.
    Stages map 1:1 to the architecture layers in Section 4.
    """

    def __init__(self):
        self.ctx = PipelineContext(stocks=list(settings.watchlist))

    async def run(self) -> None:
        started_at = datetime.utcnow()
        status = "SUCCESS"

        try:
            # ── Gate: NSE holiday check (Section 19.3) ────────────
            if await self._is_market_closed():
                logger.info(f"Market closed on {self.ctx.run_date} — skipping pipeline")
                await self._record_run(started_at, "MARKET_CLOSED")
                return

            # ── Stage 1: Data Ingestion ───────────────────────────
            await self._timed_stage("ingestion", self._run_ingestion)

            # ── Stage 2: Feature Engineering ─────────────────────
            await self._timed_stage("features", self._run_features)

            # ── Stage 3: Regime Detection ────────────────────────
            await self._timed_stage("regime", self._run_regime)

            # ── Stage 4: Strategy Engines ────────────────────────
            await self._timed_stage("strategies", self._run_strategies)

            # ── Stage 5: Signal Fusion ───────────────────────────
            await self._timed_stage("fusion", self._run_fusion)

            # ── Stage 6: LLM Override Layer ──────────────────────
            await self._timed_stage("llm", self._run_llm)

            # ── Stage 7: Microstructure Filters ─────────────────
            # (circuit breakers, gaps, macro window, pledge checks)
            await self._timed_stage("microstructure", self._run_microstructure)

            # ── Stage 8: Execution Realism ───────────────────────
            await self._timed_stage("execution_realism", self._run_execution_realism)

            # ── Stage 9: Portfolio Constraints ───────────────────
            await self._timed_stage("portfolio", self._run_portfolio)

            # ── Stage 10: Notifications ──────────────────────────
            await self._timed_stage("notifications", self._run_notifications)

        except Exception as e:
            status = "FAILED"
            self.ctx.errors.append(str(e))
            logger.exception(f"Pipeline failed: {e}")

        finally:
            total_ms = sum(self.ctx.stage_timings.values())
            logger.info(
                f"Pipeline complete | date={self.ctx.run_date} | "
                f"status={status} | signals={self.ctx.signals_generated} | "
                f"total={total_ms}ms"
            )
            await self._record_run(started_at, status)

    async def _timed_stage(self, name: str, coro) -> None:
        t0 = time.monotonic()
        await coro()
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.ctx.stage_timings[name] = elapsed_ms
        logger.info(f"Stage [{name}] completed in {elapsed_ms}ms")

    # ── Stage implementations (stubs — filled in next layers) ──

    async def _is_market_closed(self) -> bool:
        from app.data.market_calendar import is_nse_holiday
        return await is_nse_holiday(self.ctx.run_date)

    async def _run_ingestion(self) -> None:
        # Implemented in: app/data/ingestor.py
        logger.info(f"Ingesting data for {len(self.ctx.stocks)} stocks")

    async def _run_features(self) -> None:
        # Implemented in: app/features/engineer.py
        logger.info("Engineering features")

    async def _run_regime(self) -> None:
        # Implemented in: app/regime/detector.py
        logger.info("Detecting market regime")

    async def _run_strategies(self) -> None:
        # Implemented in: app/strategies/
        logger.info("Running strategy engines")

    async def _run_fusion(self) -> None:
        # Implemented in: app/fusion/engine.py
        logger.info("Running signal fusion")

    async def _run_llm(self) -> None:
        # Implemented in: app/llm/override.py
        logger.info("Running LLM override layer")

    async def _run_microstructure(self) -> None:
        # Implemented in: app/data/microstructure.py
        logger.info("Applying microstructure filters")

    async def _run_execution_realism(self) -> None:
        # Implemented in: app/data/execution_realism.py
        logger.info("Applying execution realism")

    async def _run_portfolio(self) -> None:
        # Implemented in: app/portfolio/manager.py
        logger.info("Applying portfolio constraints")

    async def _run_notifications(self) -> None:
        # Implemented in: app/notifications/dispatcher.py
        logger.info("Sending notifications")

    async def _record_run(self, started_at: datetime, status: str) -> None:
        """Write pipeline run record to pipeline_runs table (Section 24)."""
        try:
            async with get_async_db() as db:
                await db.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_uuid, run_date, started_at, completed_at,
                        ingestion_ms, features_ms, regime_ms, strategies_ms,
                        fusion_ms, llm_ms, notifications_ms, total_ms,
                        stocks_processed, stocks_skipped, signals_generated,
                        llm_overrides, llm_timeouts, data_sla_breaches, status, error_message
                    ) VALUES (
                        :run_uuid, :run_date, :started_at, NOW(),
                        :ingestion_ms, :features_ms, :regime_ms, :strategies_ms,
                        :fusion_ms, :llm_ms, :notifications_ms, :total_ms,
                        :stocks_processed, :stocks_skipped, :signals_generated,
                        :llm_overrides, :llm_timeouts, :sla_breaches, :status, :error
                    )
                    ON CONFLICT (run_date) DO UPDATE SET
                        completed_at = NOW(), status = EXCLUDED.status,
                        error_message = EXCLUDED.error_message
                    """,
                    {
                        "run_uuid": self.ctx.run_uuid,
                        "run_date": self.ctx.run_date,
                        "started_at": started_at,
                        "ingestion_ms": self.ctx.stage_timings.get("ingestion"),
                        "features_ms": self.ctx.stage_timings.get("features"),
                        "regime_ms": self.ctx.stage_timings.get("regime"),
                        "strategies_ms": self.ctx.stage_timings.get("strategies"),
                        "fusion_ms": self.ctx.stage_timings.get("fusion"),
                        "llm_ms": self.ctx.stage_timings.get("llm"),
                        "notifications_ms": self.ctx.stage_timings.get("notifications"),
                        "total_ms": sum(self.ctx.stage_timings.values()),
                        "stocks_processed": len(self.ctx.stocks) - len(self.ctx.skipped_stocks),
                        "stocks_skipped": len(self.ctx.skipped_stocks),
                        "signals_generated": self.ctx.signals_generated,
                        "llm_overrides": self.ctx.llm_overrides,
                        "llm_timeouts": self.ctx.llm_timeouts,
                        "sla_breaches": self.ctx.data_sla_breaches,
                        "status": status,
                        "error": "; ".join(self.ctx.errors) if self.ctx.errors else None,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to record pipeline run: {e}")

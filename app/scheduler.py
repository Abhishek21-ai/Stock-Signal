"""
APScheduler entry point — runs the daily signal pipeline.
Section 15: replaces Kafka for V1.

Schedule:
  - 15:45 IST (post market close): full signal pipeline
  - 08:30 IST: overnight gap check (Section 19.2)
  - 18:00 IST: news ingestion, FII/DII data scrape
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.logger import get_logger
from config.settings import settings

logger = get_logger("scheduler")

IST = pytz.timezone("Asia/Kolkata")


async def run_daily_pipeline() -> None:
    """Main daily signal generation pipeline — triggered post market close."""
    from app.pipeline import DailyPipeline
    logger.info("⏰ Starting daily signal pipeline")
    pipeline = DailyPipeline()
    await pipeline.run()


async def run_gap_check() -> None:
    """Pre-market open: check overnight gaps on active trades (Section 19.2)."""
    from app.data.gap_detector import check_overnight_gaps
    logger.info("🌅 Running overnight gap check")
    await check_overnight_gaps()


async def run_news_ingestion() -> None:
    """Evening: ingest news and FII/DII data for RAG layer."""
    from app.data.news_ingestor import ingest_news
    from app.data.fii_scraper import scrape_fii_dii
    logger.info("📰 Running evening data ingestion")
    await asyncio.gather(ingest_news(), scrape_fii_dii())


def main() -> None:
    scheduler = AsyncIOScheduler(timezone=IST)

    # Parse pipeline run time from settings (default 15:45)
    run_h, run_m = map(int, settings.pipeline_run_time.split(":"))

    scheduler.add_job(
        run_daily_pipeline,
        CronTrigger(hour=run_h, minute=run_m, timezone=IST),
        id="daily_pipeline",
        name="Daily Signal Pipeline",
        misfire_grace_time=300,
        coalesce=True,
    )

    scheduler.add_job(
        run_gap_check,
        CronTrigger(hour=8, minute=30, timezone=IST),
        id="gap_check",
        name="Overnight Gap Check",
        misfire_grace_time=120,
        coalesce=True,
    )

    scheduler.add_job(
        run_news_ingestion,
        CronTrigger(hour=18, minute=0, timezone=IST),
        id="news_ingest",
        name="Evening News + FII/DII Ingestion",
        misfire_grace_time=600,
        coalesce=True,
    )

    logger.info(
        f"Scheduler started | pipeline at {settings.pipeline_run_time} IST "
        f"| environment={settings.environment}"
    )
    scheduler.start()

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler shutting down")
        scheduler.shutdown()


if __name__ == "__main__":
    main()

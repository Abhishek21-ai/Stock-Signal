"""
FII/DII flow data scraper.
Section 19.4: scrape NSE post-close, store in fii_dii_flows.
"""
from __future__ import annotations

from app.logger import get_logger

logger = get_logger("fii_scraper")


async def scrape_fii_dii() -> None:
    """
    Scrape NSE FII/DII activity page and store in fii_dii_flows table.
    Full implementation: app/data/fii_scraper_impl.py (next build layer)
    """
    logger.info("FII/DII scrape stub — implementation in next layer")

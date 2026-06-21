"""
FII/DII flow data scraper — Section 19.4
Scrapes NSE post-close and stores it in the fii_dii_flows table.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import httpx

from app.db import get_sync_db
from app.logger import get_logger

logger = get_logger("fii_scraper")

NSE_BASE_URL = "https://www.nseindia.com"
NSE_FII_API = "https://www.nseindia.com/api/fiidiiTradeReact"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/reports/fii-dii-activity",
}

async def fetch_fii_dii_data() -> dict:
    """Fetches real-time FII/DII data from NSE."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: Hit base URL to get required NSE cookies
        await client.get(NSE_BASE_URL, headers=HEADERS)
        
        # Step 2: Hit the actual API with the session cookies
        response = await client.get(NSE_FII_API, headers=HEADERS)
        response.raise_for_status()
        return response.json()

def upsert_fii_data(fii_net: float, dii_net: float, run_date: datetime.date) -> float:
    """Upserts the flow data and calculates the 5-day rolling sum."""
    with get_sync_db() as conn:
        cursor = conn.cursor()
        
        # Upsert today's data
        cursor.execute(
            """
            INSERT INTO fii_dii_flows (date, fii_net_crore, dii_net_crore)
            VALUES (%s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                fii_net_crore = EXCLUDED.fii_net_crore,
                dii_net_crore = EXCLUDED.dii_net_crore
            """,
            (run_date, fii_net, dii_net),
        )
        
        # Calculate 5-day rolling sum of FII
        cursor.execute(
            """
            SELECT SUM(fii_net_crore) as rolling_5d
            FROM (
                SELECT fii_net_crore FROM fii_dii_flows
                WHERE date <= %s
                ORDER BY date DESC LIMIT 5
            ) as recent_5d
            """,
            (run_date,)
        )
        rolling_5d = cursor.fetchone()["rolling_5d"]
        
        # Update rolling sum
        cursor.execute(
            "UPDATE fii_dii_flows SET rolling_5d_fii_sum = %s WHERE date = %s",
            (rolling_5d, run_date)
        )
        return float(rolling_5d)

async def scrape_fii_dii() -> None:
    """Main orchestrator called by scheduler.py"""
    try:
        logger.info("Fetching FII/DII activity from NSE...")
        data = await fetch_fii_dii_data()
        
        # NSE JSON structure typically holds data in a list under a specific category
        # NSE API sometimes returns a direct list, and sometimes a wrapped dictionary
        if isinstance(data, list):
            items = data
        else:
            items = data.get("records", [])
        
        if not items:
            logger.warning("NSE returned empty FII/DII records.")
            return

        fii_net = 0.0
        dii_net = 0.0
        date_str = items[0].get("date")  # e.g., "17-Jun-2026"
        run_date = datetime.strptime(date_str, "%d-%b-%Y").date()

        for item in items:
            category = item.get("category", "")
            if "FII" in category:
                fii_net = float(item.get("buyValue", 0)) - float(item.get("sellValue", 0))
            elif "DII" in category:
                dii_net = float(item.get("buyValue", 0)) - float(item.get("sellValue", 0))

        rolling_5d = upsert_fii_data(fii_net, dii_net, run_date)
        logger.info(f"FII/DII Upserted. Date: {run_date} | FII Net: ₹{fii_net}Cr | 5d FII Sum: ₹{rolling_5d}Cr")

    except Exception as e:
        logger.error(f"FII/DII scraper failed: {e}")
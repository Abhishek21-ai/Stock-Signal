import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.fii_scraper import scrape_fii_dii
from app.db import get_sync_db

async def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — FII/DII Scraper Test")
    print(f"{'='*60}\n")
    
    print("1. Running NSE Scrape...")
    await scrape_fii_dii()
    
    print("\n2. Verifying database insertion...")
    try:
        with get_sync_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM fii_dii_flows ORDER BY date DESC LIMIT 3")
            rows = cursor.fetchall()
            
            print(f"  ✅ DB connected. Last {len(rows)} records:")
            for r in rows:
                print(f"     {r['date']} | FII: {r['fii_net_crore']} Cr | 5D Sum: {r['rolling_5d_fii_sum']} Cr")
    except Exception as e:
        print(f"  ❌ DB check failed (Is postgres running?): {e}")

if __name__ == "__main__":
    asyncio.run(main())
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.news_ingestor import fetch_news_for_stocks, embed_and_upsert, get_qdrant, ensure_collection

TEST_STOCKS = ["RELIANCE", "TCS"]

async def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — RAG Ingestor Test")
    print(f"{'='*60}\n")
    
    print("1. Initializing Qdrant Collection...")
    try:
        ensure_collection()
        print("  ✅ Qdrant connection successful.")
    except Exception as e:
        print(f"  ❌ Qdrant failed (Is docker container running?): {e}")
        return

    print("\n2. Fetching News from yfinance...")
    news_map = fetch_news_for_stocks(TEST_STOCKS)
    for stock, articles in news_map.items():
        print(f"  ✅ {stock}: Fetched {len(articles)} articles.")
        if articles:
            print(f"     Headline: {articles[0]['headline']}")

    print("\n3. Embedding and Upserting to Qdrant...")
    for stock, articles in news_map.items():
        if articles:
            count = embed_and_upsert(stock, articles)
            print(f"  ✅ {stock}: {count} vectors embedded and stored.")

    print("\n4. Verifying Search Retrieval...")
    try:
        client = get_qdrant()
        from app.data.news_ingestor import get_model
        
        # Create a mock search vector
        test_vector = get_model().encode("RELIANCE business updates").tolist()
        
        # UPDATED: Using the modern query_points method instead of search
        search_result = client.query_points(
            collection_name="stock_knowledge",
            query=test_vector,
            limit=1
        ).points
        
        if search_result:
            print(f"  ✅ Search works! Top match score: {search_result[0].score:.3f}")
            print(f"     Matched Headline: {search_result[0].payload.get('headline')}")
        else:
            print("  ⚠️ Search returned no results.")
    except Exception as e:
         print(f"  ❌ Search verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
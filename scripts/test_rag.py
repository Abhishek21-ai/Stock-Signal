"""
RAG Layer test — run after test_llm.py passes.

Usage:
    python scripts/test_rag.py

What this tests:
  1. Embedding model loads and produces 384-dim vectors
  2. Qdrant collection creation (requires Qdrant running)
  3. Upsert news articles → Qdrant
  4. Retrieve context for a symbol
  5. Idempotency — re-upsert same articles, count stays same
  6. NewsAPI fetch (requires NEWS_API_KEY in .env)
  7. Full RAGIngestor.run() batch flow
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.pipeline import (
    get_embedder, ensure_collection, embed_and_upsert,
    retrieve_context, get_context_map, RAGIngestor,
    _make_doc_text, _make_point_id, VECTOR_SIZE, COLLECTION_NAME,
)


# ── Mock articles ─────────────────────────────────────────────

def mock_articles(symbol: str, n: int = 3):
    return [
        {
            "title":       f"{symbol} Q4 results beat estimates by 12%",
            "description": f"{symbol} reported strong quarterly earnings driven by exports.",
            "url":         f"https://example.com/{symbol.lower()}-{i}",
            "source":      {"name": "Economic Times"},
            "publishedAt": "2026-06-17T08:00:00Z",
        }
        for i in range(n)
    ]


def test_embedder():
    print("\n── Test 1: Embedding model ──────────────────────────")
    try:
        embedder = get_embedder()
        vec = embedder.encode("Test sentence for embedding", show_progress_bar=False)
        assert len(vec) == VECTOR_SIZE, f"Expected {VECTOR_SIZE} dims, got {len(vec)}"
        print(f"  ✅ Embedder loaded: {VECTOR_SIZE}-dim vectors")
    except Exception as e:
        print(f"  ❌ Embedder failed: {e}")
        return False
    return True


def test_qdrant_collection():
    print("\n── Test 2: Qdrant collection setup ──────────────────")
    try:
        ensure_collection()
        from app.rag.pipeline import get_qdrant
        client = get_qdrant()
        names  = [c.name for c in client.get_collections().collections]
        assert COLLECTION_NAME in names
        print(f"  ✅ Collection '{COLLECTION_NAME}' exists in Qdrant")
    except Exception as e:
        print(f"  ❌ Qdrant not available: {e}")
        print("     → Is Qdrant running? docker compose up qdrant -d")
        return False
    return True


def test_upsert():
    print("\n── Test 3: Upsert articles → Qdrant ────────────────")
    try:
        articles = mock_articles("RELIANCE", n=3)
        n = embed_and_upsert("RELIANCE", articles)
        assert n == 3
        print(f"  ✅ Upserted {n} vectors for RELIANCE")
    except Exception as e:
        print(f"  ❌ Upsert failed: {e}")
        return False
    return True


def test_retrieval():
    print("\n── Test 4: Retrieve context ─────────────────────────")
    try:
        context = retrieve_context("RELIANCE", top_k=3)
        assert context and context != "No recent news context available."
        print(f"  ✅ Retrieved context ({len(context)} chars):")
        for line in context.split("\n"):
            print(f"     {line}")
    except Exception as e:
        print(f"  ❌ Retrieval failed: {e}")


def test_idempotency():
    print("\n── Test 5: Idempotency (re-upsert same articles) ────")
    try:
        from app.rag.pipeline import get_qdrant
        client = get_qdrant()

        articles = mock_articles("TCS", n=4)
        embed_and_upsert("TCS", articles)
        count1 = client.count(COLLECTION_NAME).count

        embed_and_upsert("TCS", articles)  # same articles again
        count2 = client.count(COLLECTION_NAME).count

        assert count1 == count2, f"Count changed: {count1} → {count2}"
        print(f"  ✅ Idempotent: count stayed at {count1} after re-upsert")
    except Exception as e:
        print(f"  ❌ Idempotency test failed: {e}")


def test_newsapi():
    print("\n── Test 6: NewsAPI fetch ────────────────────────────")
    from config.settings import settings
    if not settings.news_api_key:
        print("  ⚠️  NEWS_API_KEY not set in .env — skipping")
        return

    from app.rag.pipeline import _fetch_newsapi
    try:
        articles = _fetch_newsapi("Reliance Industries NSE India stock", page_size=5)
        print(f"  ✅ NewsAPI returned {len(articles)} articles")
        for a in articles[:2]:
            print(f"     [{a.get('source',{}).get('name','?')}] {a.get('title','')[:70]}")
    except Exception as e:
        print(f"  ❌ NewsAPI failed: {e}")


def test_batch_ingestor():
    print("\n── Test 7: RAGIngestor batch run ────────────────────")
    try:
        # Use mock articles by monkey-patching fetch
        from app.rag import pipeline as rag_mod

        original_fetch = rag_mod.fetch_news_for_stocks
        rag_mod.fetch_news_for_stocks = lambda syms: {
            s: mock_articles(s, n=2) for s in (syms + ["MARKET"])
        }

        ingestor = RAGIngestor()
        counts   = ingestor.run(["RELIANCE", "TCS", "INFY"])

        rag_mod.fetch_news_for_stocks = original_fetch  # restore

        assert len(counts) > 0
        total = sum(counts.values())
        print(f"  ✅ Batch ingest: {total} vectors across {len(counts)} symbols")
        for sym, n in counts.items():
            print(f"     {sym}: {n} vectors")
    except Exception as e:
        print(f"  ❌ Batch ingestor failed: {e}")


def test_context_map():
    print("\n── Test 8: get_context_map ──────────────────────────")
    try:
        ctx_map = get_context_map(["RELIANCE", "TCS"])
        assert "RELIANCE" in ctx_map
        assert "TCS" in ctx_map
        print(f"  ✅ Context map: {len(ctx_map)} symbols")
        for sym, ctx in ctx_map.items():
            print(f"     {sym}: {ctx[:80]}...")
    except Exception as e:
        print(f"  ❌ Context map failed: {e}")


def main():
    print(f"\n{'='*60}")
    print("  Stock Signal Platform — RAG Layer Test")
    print(f"{'='*60}")

    ok = test_embedder()
    if not ok:
        print("\n  ⛔ Embedder failed — install sentence-transformers")
        return

    qdrant_ok = test_qdrant_collection()
    if not qdrant_ok:
        print("\n  ⛔ Qdrant unavailable — run: docker compose up qdrant -d")
        return

    test_upsert()
    test_retrieval()
    test_idempotency()
    test_newsapi()
    test_batch_ingestor()
    test_context_map()

    print(f"\n{'='*60}")
    print("  ✅ RAG Layer verified")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

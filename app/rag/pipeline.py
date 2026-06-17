"""
RAG Layer — Section 11
News ingestion → embedding → Qdrant storage → retrieval per stock signal.

Pipeline (Section 11.1):
  1. Fetch headlines from NewsAPI for each stock + broad market
  2. Chunk: each headline + description = one document
  3. Embed with sentence-transformers all-MiniLM-L6-v2 (384-dim)
  4. Upsert to Qdrant collection 'stock_knowledge' with payload metadata
  5. At signal time: retrieve top-K relevant docs per symbol
  6. Return formatted context string for LLM prompt enrichment

Qdrant payload schema per point:
  {
    "symbol":      "RELIANCE",        # stock ticker or "MARKET"
    "headline":    "...",
    "description": "...",
    "url":         "...",
    "source":      "...",
    "published_at":"2026-06-17T10:00:00Z",
    "date":        "2026-06-17"
  }
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from app.logger import get_logger
from config.settings import settings

logger = get_logger("rag")

# ── Constants ─────────────────────────────────────────────────
EMBED_MODEL      = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
TOP_K            = 5                      # docs retrieved per symbol
MAX_NEWS_PER_STOCK = 10                  # headlines fetched per stock from NewsAPI
COLLECTION_NAME  = settings.qdrant_collection_name
VECTOR_SIZE      = settings.qdrant_vector_size   # 384


# ── Lazy singletons ───────────────────────────────────────────

_embedder  = None
_qdrant    = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
        )
    return _qdrant


# ── Collection setup ──────────────────────────────────────────

def ensure_collection() -> None:
    """Create Qdrant collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    else:
        logger.debug(f"Qdrant collection exists: {COLLECTION_NAME}")


# ── News fetching ─────────────────────────────────────────────

def _fetch_newsapi(query: str, page_size: int = 10) -> List[Dict]:
    """Fetch articles from NewsAPI. Returns list of article dicts."""
    if not settings.news_api_key:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI fetch")
        return []
    try:
        from newsapi import NewsApiClient
        client   = NewsApiClient(api_key=settings.news_api_key)
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=page_size,
        )
        return response.get("articles", [])
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed for query={query!r}: {e}")
        return []


def fetch_news_for_stocks(symbols: List[str]) -> Dict[str, List[Dict]]:
    """
    Fetch news for each symbol + broad market.
    Returns { symbol → [article, ...] }
    """
    results: Dict[str, List[Dict]] = {}

    # Broad market news
    market_articles = _fetch_newsapi("NSE India stock market Nifty", page_size=15)
    results["MARKET"] = market_articles
    logger.info(f"Fetched {len(market_articles)} market-level articles")

    # Per-stock news
    for symbol in symbols:
        # Strip .NS suffix for cleaner query
        clean = symbol.replace(".NS", "").replace(".BO", "")
        articles = _fetch_newsapi(f"{clean} NSE India stock", page_size=MAX_NEWS_PER_STOCK)
        results[symbol] = articles
        if articles:
            logger.debug(f"{symbol}: {len(articles)} articles fetched")

    return results


# ── Embedding ─────────────────────────────────────────────────

def _make_doc_text(article: Dict) -> str:
    """Combine headline + description into single embeddable string."""
    headline = article.get("title", "") or ""
    desc     = article.get("description", "") or ""
    # Truncate to avoid very long inputs
    return f"{headline.strip()} {desc.strip()[:200]}".strip()


def _make_point_id(symbol: str, url: str) -> str:
    """Stable UUID from symbol + URL so re-runs are idempotent."""
    h = hashlib.md5(f"{symbol}:{url}".encode()).hexdigest()
    return str(uuid.UUID(h))


def embed_and_upsert(symbol: str, articles: List[Dict]) -> int:
    """
    Embed articles for one symbol and upsert to Qdrant.
    Returns number of points upserted.
    """
    if not articles:
        return 0

    from qdrant_client.models import PointStruct

    embedder = get_embedder()
    client   = get_qdrant()
    today    = str(date.today())

    texts  = [_make_doc_text(a) for a in articles]
    # Filter empty texts
    valid  = [(t, a) for t, a in zip(texts, articles) if t.strip()]
    if not valid:
        return 0

    texts_clean, articles_clean = zip(*valid)
    vectors = embedder.encode(list(texts_clean), show_progress_bar=False).tolist()

    points = []
    for vec, article, text in zip(vectors, articles_clean, texts_clean):
        url = article.get("url", "") or ""
        points.append(PointStruct(
            id=_make_point_id(symbol, url),
            vector=vec,
            payload={
                "symbol":       symbol,
                "headline":     article.get("title", ""),
                "description":  article.get("description", ""),
                "url":          url,
                "source":       (article.get("source") or {}).get("name", ""),
                "published_at": article.get("publishedAt", ""),
                "date":         today,
                "text":         text,
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.debug(f"{symbol}: upserted {len(points)} news vectors to Qdrant")
    return len(points)


# ── Retrieval ─────────────────────────────────────────────────

def retrieve_context(symbol: str, top_k: int = TOP_K) -> str:
    """
    Retrieve top-K most relevant news docs for a symbol.
    Returns formatted string for LLM prompt injection.
    """
    try:
        embedder = get_embedder()
        client   = get_qdrant()

        # Query vector = embedding of the symbol name (simple but effective)
        query_vec = embedder.encode(
            f"{symbol} NSE India stock news", show_progress_bar=False
        ).tolist()

        from qdrant_client.models import Filter, FieldCondition, MatchAny
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="symbol", match=MatchAny(any=[symbol, "MARKET"]))]
            ),
        )
        hits = response.points

        if not hits:
            return "No recent news context available."

        lines = []
        for hit in hits:
            p    = hit.payload
            src  = p.get("source", "Unknown")
            pub  = p.get("published_at", "")[:10]
            head = p.get("headline", "")
            lines.append(f"[{src} | {pub}] {head}")

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"RAG retrieval failed for {symbol}: {e}")
        return "News context unavailable."


# ── Batch ingest entry point ──────────────────────────────────

class RAGIngestor:
    """
    Called by pipeline.py before LLM stage.
    Fetches news for all stocks, embeds, upserts to Qdrant.
    """

    def run(self, symbols: List[str]) -> Dict[str, int]:
        """
        Returns { symbol → points_upserted }
        """
        ensure_collection()

        news_map   = fetch_news_for_stocks(symbols)
        upsert_counts: Dict[str, int] = {}

        for symbol, articles in news_map.items():
            try:
                n = embed_and_upsert(symbol, articles)
                upsert_counts[symbol] = n
            except Exception as e:
                logger.error(f"RAG ingest failed for {symbol}: {e}")
                upsert_counts[symbol] = 0

        total = sum(upsert_counts.values())
        logger.info(
            f"RAG ingest complete: {total} vectors upserted "
            f"across {len(upsert_counts)} symbols"
        )
        return upsert_counts


def get_context_map(symbols: List[str]) -> Dict[str, str]:
    """
    Retrieve RAG context for all symbols.
    Returns { symbol → context_string } for LLM prompt injection.
    """
    return {symbol: retrieve_context(symbol) for symbol in symbols}
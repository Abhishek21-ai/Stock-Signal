"""
News ingestion for RAG layer — Section 11.1
Fetches headlines, embeds with all-MiniLM-L6-v2, and upserts to Qdrant.
"""
from __future__ import annotations

import uuid
import os 
from datetime import datetime, timezone
from typing import Dict, List
import yfinance as yf

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from app.logger import get_logger
from config.settings import settings
from config.watchlist import WATCHLIST_WITH_SECTORS

logger = get_logger("news_ingestor")

COLLECTION_NAME = "stock_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # Specific to MiniLM

# Lazy load model & Qdrant to avoid slowing down pipeline imports
_model = None
_qdrant = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"), 
            port=6333,
            check_compatibility=False  # Suppresses the version warning
        )
    return _qdrant

def ensure_collection():
    """Creates the Qdrant collection if it doesn't exist."""
    client = get_qdrant()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")

def fetch_news_for_stocks(symbols: List[str]) -> Dict[str, List[dict]]:
    """Fetches latest news from yfinance for given symbols."""
    news_map = {}
    for symbol in symbols:
        ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news[:5] # Get top 5 recent articles
            
            articles = []
            for item in raw_news:
                # yfinance news schema changed recently; handle both flat and nested schemas
                content = item.get("content", item)
                
                title = content.get("title", item.get("title", ""))
                summary = content.get("summary", item.get("summary", ""))
                
                # Extract URL securely from nested dicts if present
                url = content.get("clickThroughUrl", {}).get("url") if isinstance(content.get("clickThroughUrl"), dict) else None
                url = url or (content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else None)
                url = url or content.get("url", item.get("link", ""))
                
                pub_time = content.get("pubDate", item.get("providerPublishTime", 0))
                try:
                    pub_str = pub_time if isinstance(pub_time, str) else datetime.fromtimestamp(pub_time, tz=timezone.utc).isoformat()
                except Exception:
                    pub_str = datetime.now(timezone.utc).isoformat()

                if title:  # Only append if we actually found a headline
                    articles.append({
                        "symbol": symbol,
                        "headline": title,
                        "description": summary,
                        "url": url,
                        "published_at": pub_str,
                    })  
        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            news_map[symbol] = []
            
    return news_map

def embed_and_upsert(symbol: str, articles: List[dict]) -> int:
    """Embeds article chunks and pushes them to Qdrant."""
    if not articles:
        return 0

    client = get_qdrant()
    model = get_model()
    points = []

    for art in articles:
        # Create chunk text combining title and summary
        chunk_text = f"[{symbol}] {art['headline']}. {art['description']}"
        vector = model.encode(chunk_text).tolist()
        
        # Use UUID5 based on URL to prevent duplicating same article runs
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, art['url']))
        
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=art
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    return len(points)

async def ingest_news() -> None:
    """Orchestrator called by scheduler.py at 18:00 IST"""
    logger.info("Starting evening RAG news ingestion...")
    ensure_collection()
    
    symbols = list(WATCHLIST_WITH_SECTORS.keys()) if WATCHLIST_WITH_SECTORS else ["RELIANCE", "TCS", "HDFCBANK"]
    news_map = fetch_news_for_stocks(symbols)
    
    total_upserts = 0
    for symbol, articles in news_map.items():
        count = embed_and_upsert(symbol, articles)
        total_upserts += count
        
    logger.info(f"RAG ingestion complete. Embedded {total_upserts} news vectors into Qdrant.")
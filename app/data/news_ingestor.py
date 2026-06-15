"""
News ingestion for RAG layer.
Section 11.1: NewsAPI + RSS → chunk → embed → Qdrant
"""
from __future__ import annotations

from app.logger import get_logger

logger = get_logger("news_ingestor")


async def ingest_news() -> None:
    """
    Fetch news headlines, chunk, embed with all-MiniLM-L6-v2, upsert to Qdrant.
    Full implementation in app/rag/ layer.
    """
    logger.info("News ingestion stub — implementation in RAG layer")

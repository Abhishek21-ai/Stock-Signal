"""
Redis cache client.
Used for:
  - Feature vector caching (Section 6)
  - LLM response cache (Section 27.2) key: llm_override:{stock}:{date}  TTL: 24h
  - Signal dedup within a session
"""
from __future__ import annotations

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from config.settings import settings

_pool: Optional[aioredis.Redis] = None


def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    return _pool


class Cache:
    """Thin wrapper with JSON serialisation and namespaced keys."""

    def __init__(self, prefix: str = "ssp"):
        self._prefix = prefix
        self._r = get_redis()

    def _key(self, *parts: str) -> str:
        return f"{self._prefix}:{':'.join(parts)}"

    async def get(self, *key_parts: str) -> Optional[Any]:
        raw = await self._r.get(self._key(*key_parts))
        return json.loads(raw) if raw is not None else None

    async def set(self, *key_parts_and_value, ttl_seconds: int = 3600) -> None:
        *key_parts, value = key_parts_and_value
        await self._r.setex(
            self._key(*key_parts),
            ttl_seconds,
            json.dumps(value, default=str),
        )

    async def delete(self, *key_parts: str) -> None:
        await self._r.delete(self._key(*key_parts))

    async def exists(self, *key_parts: str) -> bool:
        return bool(await self._r.exists(self._key(*key_parts)))


# ── Named caches ─────────────────────────────────────────────
llm_cache = Cache(prefix="llm_override")     # TTL 24h per Section 27.2
feature_cache = Cache(prefix="features")      # TTL 1h
signal_cache = Cache(prefix="signal")         # TTL session

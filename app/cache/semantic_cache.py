"""Semantic cache using GPTCache and Redis."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config.settings import get_settings
from app.embeddings.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cached query-response pair with embedding."""

    query: str
    response: str
    citations: list[dict[str, Any]]
    embedding: list[float]
    metadata: dict[str, Any]
    created_at: float


@dataclass
class CacheResult:
    """Result of a cache lookup."""

    hit: bool
    response: str = ""
    citations: list[dict[str, Any]] | None = None
    similarity: float = 0.0
    metadata: dict[str, Any] | None = None


class SemanticCache:
    """Semantic similarity cache backed by Redis with in-memory fallback."""

    KEY_PREFIX = "instapark:semantic_cache:"

    def __init__(self) -> None:
        settings = get_settings()
        self.threshold = settings.cache_similarity_threshold
        self.enabled = settings.gptcache_enabled
        self.embedding_service = get_embedding_service()
        self._redis_client: Any = None
        self._memory_store: list[CacheEntry] = []
        self._stats = {"hits": 0, "misses": 0}
        self._init_redis()

    def _init_redis(self) -> None:
        if not self.enabled:
            logger.info("Semantic cache disabled")
            return
        try:
            import redis

            settings = get_settings()
            self._redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._redis_client.ping()
            logger.info("Connected to Redis semantic cache")
        except Exception as exc:
            logger.warning("Redis unavailable, using in-memory cache: %s", exc)
            self._redis_client = None

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _get_all_entries(self) -> list[CacheEntry]:
        if self._redis_client is None:
            return self._memory_store

        entries: list[CacheEntry] = []
        try:
            keys = self._redis_client.keys(f"{self.KEY_PREFIX}*")
            for key in keys:
                raw = self._redis_client.get(key)
                if raw:
                    data = json.loads(raw)
                    entries.append(
                        CacheEntry(
                            query=data["query"],
                            response=data["response"],
                            citations=data.get("citations", []),
                            embedding=data["embedding"],
                            metadata=data.get("metadata", {}),
                            created_at=data.get("created_at", 0),
                        )
                    )
        except Exception as exc:
            logger.warning("Failed to read cache from Redis: %s", exc)
        return entries

    def lookup(self, query: str) -> CacheResult:
        """Search cache for semantically similar query."""
        if not self.enabled:
            return CacheResult(hit=False)

        query_embedding = self.embedding_service.embed_query(query)
        best_similarity = 0.0
        best_entry: CacheEntry | None = None

        for entry in self._get_all_entries():
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        if best_entry and best_similarity >= self.threshold:
            self._stats["hits"] += 1
            logger.info("Cache HIT (similarity=%.4f)", best_similarity)
            return CacheResult(
                hit=True,
                response=best_entry.response,
                citations=best_entry.citations,
                similarity=best_similarity,
                metadata=best_entry.metadata,
            )

        self._stats["misses"] += 1
        logger.debug("Cache MISS (best=%.4f)", best_similarity)
        return CacheResult(hit=False, similarity=best_similarity)

    def store(
        self,
        query: str,
        response: str,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store query-response pair in semantic cache."""
        if not self.enabled:
            return

        embedding = self.embedding_service.embed_query(query)
        entry = CacheEntry(
            query=query,
            response=response,
            citations=citations or [],
            embedding=embedding,
            metadata=metadata or {},
            created_at=time.time(),
        )

        if self._redis_client is None:
            self._memory_store.append(entry)
            return

        try:
            import hashlib

            key = f"{self.KEY_PREFIX}{hashlib.sha256(query.encode()).hexdigest()[:16]}"
            payload = {
                "query": entry.query,
                "response": entry.response,
                "citations": entry.citations,
                "embedding": entry.embedding,
                "metadata": entry.metadata,
                "created_at": entry.created_at,
            }
            self._redis_client.set(key, json.dumps(payload))
            logger.debug("Stored response in semantic cache")
        except Exception as exc:
            logger.warning("Failed to store in Redis, using memory: %s", exc)
            self._memory_store.append(entry)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory_store.clear()
        if self._redis_client:
            try:
                keys = self._redis_client.keys(f"{self.KEY_PREFIX}*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception as exc:
                logger.warning("Failed to clear Redis cache: %s", exc)
        self._stats = {"hits": 0, "misses": 0}

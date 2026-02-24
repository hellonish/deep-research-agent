"""
Redis-backed working memory for active chat sessions and real-time state.

All keys are user-scoped: user:{user_id}:*
"""
import json
import time
from typing import Any

import redis.asyncio as redis

from app.config import settings


class MemoryService:
    """Redis-backed working memory with user-scoped key isolation."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    # ── Intra-chat context (sliding window) ──────────────────────────

    async def push_message(self, user_id: str, session_id: str, role: str, content: str):
        """Append a message to an active session's working memory."""
        msg = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time(),
        })
        key = f"user:{user_id}:chat:{session_id}:messages"
        await self.redis.rpush(key, msg)
        await self.redis.ltrim(key, -50, -1)   # Keep last 50 messages
        await self.redis.expire(key, 86400)     # TTL: 24 hours

    async def get_context(self, user_id: str, session_id: str, last_n: int = 20) -> list[dict]:
        """Retrieve the last N messages for LLM context injection."""
        key = f"user:{user_id}:chat:{session_id}:messages"
        raw = await self.redis.lrange(key, -last_n, -1)
        return [json.loads(m) for m in raw]

    # ── Research job progress (Pub/Sub) ──────────────────────────────

    async def publish_progress(self, job_id: str, event: dict):
        """Publish a structured progress event for WebSocket streaming."""
        await self.redis.publish(
            f"research:{job_id}:progress",
            json.dumps(event),
        )

    # ── API Key caching ──────────────────────────────────────────────

    def _api_key_cache_key(self, user_id: str, suffix: str | None = None) -> str:
        """Redis key for API key cache. suffix=None for legacy single-key per user."""
        if suffix:
            return f"user:{user_id}:api_key:{suffix}"
        return f"user:{user_id}:api_key"

    async def cache_api_key(self, user_id: str, api_key: str, suffix: str | None = None):
        """Cache the user's decrypted API key in Redis (1 hour TTL). suffix = provider for multi-provider."""
        await self.redis.setex(
            self._api_key_cache_key(user_id, suffix),
            3600,
            api_key,
        )

    async def get_cached_api_key(self, user_id: str, suffix: str | None = None) -> str | None:
        """Retrieve the cached API key, or None if expired/missing. suffix = provider for multi-provider."""
        key = self._api_key_cache_key(user_id, suffix)
        result = await self.redis.get(key)
        if result is None:
            return None
        return result.decode() if isinstance(result, bytes) else result

    # ── User preferences cache ───────────────────────────────────────

    async def set_user_prefs(self, user_id: str, prefs: dict[str, str]):
        """Cache user preferences (selected model, etc.)."""
        await self.redis.hset(f"user:{user_id}:prefs", mapping=prefs)
        await self.redis.expire(f"user:{user_id}:prefs", 86400)

    async def get_user_prefs(self, user_id: str) -> dict:
        """Retrieve cached user preferences."""
        result = await self.redis.hgetall(f"user:{user_id}:prefs")
        return {k.decode(): v.decode() for k, v in result.items()} if result else {}


# ── Redis connection factory ─────────────────────────────────────────

_redis_pool: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """FastAPI dependency — returns a shared async Redis connection."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False,
        )
    return _redis_pool

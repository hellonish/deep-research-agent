"""
Redis-based per-user rate limiting middleware.
"""
from fastapi import HTTPException

import redis.asyncio as redis


async def check_rate_limit(
    user_id: str,
    redis_client: redis.Redis,
    endpoint: str,
    max_per_hour: int,
):
    """
    Enforces per-user rate limiting using Redis INCR + EXPIRE.

    Args:
        user_id: The authenticated user's ID.
        redis_client: Async Redis connection.
        endpoint: The endpoint name (e.g., "chat", "research").
        max_per_hour: Maximum allowed requests per hour.

    Raises:
        HTTPException(429): If the rate limit is exceeded.
    """
    key = f"ratelimit:{user_id}:{endpoint}"
    count = await redis_client.incr(key)

    if count == 1:
        # First request — set the expiry window
        await redis_client.expire(key, 3600)

    if count > max_per_hour:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {max_per_hour} {endpoint} requests per hour",
        )

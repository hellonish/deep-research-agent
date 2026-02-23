"""
Shared API key resolution: Redis cache → DB (decrypted) → server default.
Used by chat and research to get the Gemini API key for the current user.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import User
from app.services.crypto_service import decrypt_api_key
from app.services.memory_service import MemoryService


async def resolve_api_key(
    user_id: str,
    memory: MemoryService,
    db: AsyncSession,
) -> str:
    """
    Resolve the Gemini API key for the user: cache → DB → server default.

    Returns:
        The API key string to use for LLM calls.
    """
    api_key = await memory.get_cached_api_key(user_id)
    if api_key:
        return api_key

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user and user.gemini_api_key:
        api_key = decrypt_api_key(user.gemini_api_key)
        await memory.cache_api_key(user_id, api_key)
        return api_key

    return settings.GOOGLE_API_KEY or ""

"""
Shared API key resolution: Redis cache → DB (decrypted) → server default.
Supports multiple providers (gemini, deepseek, openai) via UserApiKey; legacy Gemini key on User.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings, model_id_to_provider
from app.db.models import User, UserApiKey
from app.services.crypto_service import decrypt_api_key
from app.services.memory_service import MemoryService

SUPPORTED_PROVIDERS = ("gemini", "deepseek", "openai")
PROVIDER_LABELS = {"gemini": "Gemini", "deepseek": "DeepSeek", "openai": "OpenAI"}


async def list_provider_keys(user_id: str, db: AsyncSession) -> list[dict]:
    """
    Return list of providers the user has an API key for (no key values).
    Includes legacy gemini_api_key on User.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return []

    keys = []
    # From UserApiKey table
    result = await db.execute(
        select(UserApiKey.provider).where(UserApiKey.user_id == user_id)
    )
    for (provider,) in result.all():
        keys.append({"provider": provider, "label": PROVIDER_LABELS.get(provider, provider)})

    # Legacy: if user has gemini_api_key but no gemini in UserApiKey, include it
    if user.gemini_api_key and not any(k["provider"] == "gemini" for k in keys):
        keys.append({"provider": "gemini", "label": "Gemini"})

    return sorted(keys, key=lambda x: x["provider"])


async def resolve_api_key(
    user_id: str,
    provider: str,
    memory: MemoryService,
    db: AsyncSession,
) -> str:
    """
    Resolve the API key for the user and provider: cache → UserApiKey / User.gemini_api_key → server default.

    Returns:
        The API key string to use for LLM calls for that provider.
    """
    api_key = await memory.get_cached_api_key(user_id, suffix=provider)
    if api_key:
        return api_key

    # Prefer UserApiKey
    result = await db.execute(
        select(UserApiKey).where(
            UserApiKey.user_id == user_id,
            UserApiKey.provider == provider,
        )
    )
    row = result.scalar_one_or_none()
    if row and row.encrypted_key:
        api_key = decrypt_api_key(row.encrypted_key)
        await memory.cache_api_key(user_id, api_key, suffix=provider)
        return api_key

    # Legacy: Gemini only
    if provider == "gemini":
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user and user.gemini_api_key:
            api_key = decrypt_api_key(user.gemini_api_key)
            await memory.cache_api_key(user_id, api_key, suffix="gemini")
            return api_key
        return settings.GOOGLE_API_KEY or ""

    if provider == "deepseek":
        return getattr(settings, "DEEPSEEK_API_KEY", None) or ""

    return ""


async def resolve_api_key_for_model(
    user_id: str,
    model_id: str,
    memory: MemoryService,
    db: AsyncSession,
) -> str:
    """Resolve the API key for the model the user selected (derives provider from model_id)."""
    provider = model_id_to_provider(model_id)
    return await resolve_api_key(user_id, provider, memory, db)

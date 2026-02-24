"""
Model management router — API keys per provider, model discovery, active model.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.db.models import User, UserApiKey
from app.core.dependencies import get_current_user
from app.schemas.models import (
    SetKeyRequest,
    SetKeyResponse,
    SetModelRequest,
    SetModelResponse,
    ModelsResponse,
    KeyStatusResponse,
    ProviderKeysResponse,
)
from app.services.crypto_service import encrypt_api_key, decrypt_api_key
from app.services.memory_service import MemoryService, get_redis
from app.services.model_service import list_models_for_provider, validate_api_key
from app.services.api_key_service import list_provider_keys, resolve_api_key, SUPPORTED_PROVIDERS

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/set-key", response_model=SetKeyResponse)
async def set_api_key(
    req: SetKeyRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    Validates and stores the user's API key for the given provider (gemini, deepseek, openai).
    Returns the list of available models for that provider.
    """
    provider = (req.provider or "gemini").lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Provider must be one of: {SUPPORTED_PROVIDERS}")

    is_valid = validate_api_key(req.api_key, provider=provider)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid {provider} API key")

    models = list_models_for_provider(provider, req.api_key)

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Upsert UserApiKey
    result = await db.execute(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    row = result.scalar_one_or_none()
    if row:
        row.encrypted_key = encrypt_api_key(req.api_key)
    else:
        db.add(UserApiKey(user_id=user_id, provider=provider, encrypted_key=encrypt_api_key(req.api_key)))

    if provider == "gemini":
        user.gemini_api_key = encrypt_api_key(req.api_key)

    await db.commit()

    memory = MemoryService(redis)
    await memory.cache_api_key(user_id, req.api_key, suffix=provider)

    return SetKeyResponse(valid=True, models=models, provider=provider)


@router.get("/keys", response_model=ProviderKeysResponse)
async def get_provider_keys(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns list of providers the user has API keys for (no key values)."""
    keys = await list_provider_keys(user_id, db)
    return ProviderKeysResponse(keys=keys)


@router.get("/key-status", response_model=KeyStatusResponse)
async def get_key_status(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns whether the user has at least one API key stored."""
    keys = await list_provider_keys(user_id, db)
    configured = len(keys) > 0
    return KeyStatusResponse(configured=configured)


@router.get("/available", response_model=ModelsResponse)
async def list_models(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """Returns models from all providers the user has API keys for."""
    memory = MemoryService(redis)
    keys = await list_provider_keys(user_id, db)
    if not keys:
        if settings.GOOGLE_API_KEY:
            models = list_models_for_provider("gemini", settings.GOOGLE_API_KEY)
            return ModelsResponse(models=models)
        raise HTTPException(
            status_code=401,
            detail="No API key configured. Add one in Settings (e.g. Gemini or DeepSeek).",
        )

    all_models = []
    for k in keys:
        provider = k["provider"]
        api_key = await resolve_api_key(user_id, provider, memory, db)
        if not api_key and provider == "gemini":
            api_key = settings.GOOGLE_API_KEY
        if api_key:
            all_models.extend(list_models_for_provider(provider, api_key))

    if not all_models:
        raise HTTPException(status_code=401, detail="No API key configured. Add one in Settings.")

    all_models.sort(key=lambda m: (m.get("provider", ""), str(m.get("display_name", m.get("id", "")))))
    return ModelsResponse(models=all_models)


@router.post("/set-model", response_model=SetModelResponse)
async def set_active_model(
    req: SetModelRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Saves the user's preferred active model (any supported model id, e.g. gemini-2.5-flash or deepseek-chat)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.selected_model = req.model_id
    await db.commit()
    return SetModelResponse(success=True, selected_model=req.model_id)

"""
Model management router — API key validation and model discovery.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db
from app.db.models import User
from app.core.dependencies import get_current_user
from app.schemas.models import (
    SetKeyRequest,
    SetKeyResponse,
    SetModelRequest,
    SetModelResponse,
    ModelsResponse,
    KeyStatusResponse,
)
from app.services.crypto_service import encrypt_api_key, decrypt_api_key
from app.services.memory_service import MemoryService, get_redis
from app.services.model_service import list_available_models, validate_api_key

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/set-key", response_model=SetKeyResponse)
async def set_api_key(
    req: SetKeyRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """
    Validates and stores the user's Gemini API key.
    Returns the list of available models for is key.
    """
    # 1. Validate the key
    is_valid = validate_api_key(req.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid Gemini API key")

    # 2. Get available models
    models = list_available_models(req.api_key)

    # 3. Store encrypted in PostgreSQL
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.gemini_api_key = encrypt_api_key(req.api_key)
    await db.commit()

    # 4. Cache in Redis
    memory = MemoryService(redis)
    await memory.cache_api_key(user_id, req.api_key)

    return SetKeyResponse(valid=True, models=models)


@router.get("/key-status", response_model=KeyStatusResponse)
async def get_key_status(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Returns whether the user has an API key stored (without revealing the key)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    configured = user is not None and bool(user.gemini_api_key)
    return KeyStatusResponse(configured=configured)


@router.get("/available", response_model=ModelsResponse)
async def list_models(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
):
    """Returns curated models that the user's API key can access (intersection of our list and their key)."""
    memory = MemoryService(redis)

    # Try Redis cache first
    api_key = await memory.get_cached_api_key(user_id)

    if not api_key:
        # Fall back to database
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if user and user.gemini_api_key:
            api_key = decrypt_api_key(user.gemini_api_key)
            # Re-cache for next time
            await memory.cache_api_key(user_id, api_key)

    if not api_key:
        # No user key — use server default to show free-tier models
        api_key = settings.GOOGLE_API_KEY

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="No API key configured. Please add one in Settings.",
        )

    models = list_available_models(api_key)
    return ModelsResponse(models=models)


@router.post("/set-model", response_model=SetModelResponse)
async def set_active_model(
    req: SetModelRequest,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Saves the user's preferred active model.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.selected_model = req.model_id
    await db.commit()

    return SetModelResponse(success=True, selected_model=req.model_id)

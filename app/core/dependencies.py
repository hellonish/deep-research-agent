"""
FastAPI dependencies for authentication and shared resources.
"""
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_db


def create_jwt(user_id: str, email: str) -> str:
    """Create a signed JWT token for the given user."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=settings.JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


async def get_current_user(authorization: str = Header(None)) -> str:
    """
    FastAPI dependency — extracts and validates user_id from the JWT
    in the Authorization header.

    Returns:
        str: The authenticated user's ID.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # Support "Bearer <token>" format
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def decode_token_user_id(token: str) -> str | None:
    """Decode JWT and return user_id, or None if invalid/expired."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("user_id")
    except Exception:
        return None

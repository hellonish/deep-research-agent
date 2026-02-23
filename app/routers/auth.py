"""
Google OAuth 2.0 authentication router.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.core.config import settings
from app.core.dependencies import create_jwt
from app.db.database import get_db
from app.db.models import User
from app.schemas.auth import AuthResponse, GoogleAuthRequest

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=AuthResponse)
async def google_login(req: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """
    Validate a Google OAuth id_token and return a Wort JWT.

    Flow:
    1. Verify Google token using google-auth library
    2. Extract user info (email, name, picture)
    3. Create or update user in database
    4. Issue our own JWT
    """
    # 1. Verify the Google id_token
    try:
        payload = id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

    # 2. Extract user info
    google_id = payload["sub"]
    email = payload.get("email", "")
    name = payload.get("name", "")
    picture = payload.get("picture", "")

    if not email:
        raise HTTPException(status_code=400, detail="Email not found in Google token")

    # 3. Upsert user
    result = await db.execute(select(User).where(User.google_id == google_id))
    user = result.scalar_one_or_none()

    if user:
        # Update existing user
        user.name = name
        user.picture = picture
    else:
        # Create new user
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture,
        )
        db.add(user)

    await db.commit()
    await db.refresh(user)

    # 4. Issue JWT
    token = create_jwt(user_id=user.id, email=user.email)

    return AuthResponse(
        access_token=token,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
            "selected_model": user.selected_model,
        },
    )

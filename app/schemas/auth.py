"""Request/response schemas for the auth API."""
from pydantic import BaseModel


class GoogleAuthRequest(BaseModel):
    """Request body for Google OAuth login."""
    id_token: str


class AuthResponse(BaseModel):
    """Response after successful authentication."""
    access_token: str
    user: dict

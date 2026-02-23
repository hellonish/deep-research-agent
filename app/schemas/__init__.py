"""
Pydantic request/response schemas for API routes.
Routers can import from here to keep route modules thin.
"""
from app.schemas.auth import GoogleAuthRequest, AuthResponse
from app.schemas.chat import ChatCreateResponse, ChatRequest
from app.schemas.history import RenameRequest
from app.schemas.ingest import IngestResponse
from app.schemas.research import ResearchRequest, ResearchResponse
from app.schemas.models import (
    SetKeyRequest,
    SetKeyResponse,
    SetModelRequest,
    SetModelResponse,
    ModelsResponse,
)

__all__ = [
    "GoogleAuthRequest",
    "AuthResponse",
    "ChatRequest",
    "ChatCreateResponse",
    "RenameRequest",
    "IngestResponse",
    "ResearchRequest",
    "ResearchResponse",
    "SetKeyRequest",
    "SetKeyResponse",
    "SetModelRequest",
    "SetModelResponse",
    "ModelsResponse",
]

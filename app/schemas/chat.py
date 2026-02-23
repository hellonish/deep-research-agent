"""Request/response schemas for the chat API."""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    mode: str = "chat"  # "chat" | "web"
    model_id: str | None = None


class ChatCreateResponse(BaseModel):
    session_id: str

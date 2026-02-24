"""Request/response schemas for the chat API."""
from typing import Optional
from pydantic import BaseModel, Field


class WebSearchDecision(BaseModel):
    """LLM output: whether to run web search and what query to use."""
    use_web: bool = Field(description="True if a web search would help answer the user's message.")
    search_query: Optional[str] = Field(
        default=None,
        description="If use_web is True, the exact search query to run (concise, 2-8 words). If use_web is False, leave null.",
    )


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    mode: str = "chat"  # "chat" | "web"
    model_id: str | None = None


class ChatCreateResponse(BaseModel):
    session_id: str

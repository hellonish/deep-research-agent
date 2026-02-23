"""Request/response schemas for the history API."""
from pydantic import BaseModel


class RenameRequest(BaseModel):
    """Request body for renaming a chat session or research job."""
    title: str

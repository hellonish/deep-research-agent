"""Request/response schemas for the research API."""
from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    model_id: str | None = None
    config: dict | None = None


class ResearchResponse(BaseModel):
    job_id: str
    session_id: str
    status: str

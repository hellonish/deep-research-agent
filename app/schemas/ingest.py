"""Request/response schemas for the ingest API."""
from pydantic import BaseModel


class IngestResponse(BaseModel):
    """Response after a successful file upload and ingest."""
    file_name: str
    file_size: int
    chunks_created: int
    collection: str

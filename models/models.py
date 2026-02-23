import json
from enum import Enum
from typing import Any, List, Optional, Literal
from pydantic import BaseModel, Field

# Query Types for Specialized PLan
class QueryType(str, Enum):
    """A comprehensive set of research domains."""
    FINANCE = "Finance_and_Economics"
    MATHEMATICS = "Mathematics_and_Statistics"
    SOFTWARE = "Software_and_Computer_Science"
    NATURAL_SCIENCES = "Natural_Sciences"  # Physics, Chemistry, Biology
    SOCIAL_SCIENCES = "Social_Sciences"    # Psychology, Sociology, Anthropology
    HUMANITIES = "Humanities_and_Arts"     # History, Literature, Philosophy
    ENGINEERING = "Engineering_and_Technology"
    MEDICINE = "Medicine_and_Healthcare"
    LAW = "Law_and_Policy"
    GENERAL = "General_Research"           # Fallback for broad or multi-disciplinary queries

class PlanStep(BaseModel):
    step_number: int = Field(description="The sequential number of the step.")
    action: str = Field(description="A short, simple summary of the step for the user.")
    description: str = Field(description="Detailed instructions for the agent to execute this step.")

class ResearchPlan(BaseModel):
    query_type: QueryType = Field(description="The categorized domain of the research query.")
    plan: List[PlanStep] = Field(description="The sequential list of steps required to complete the research.")


# ── Vector Store Models ────────────────────────────────────────────────

class Document(BaseModel):
    """A document to be stored in the vector store."""

    id: str = Field(description="Unique identifier for the document.")
    content: str = Field(description="The text content of the document.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata: source, title, url, chunk_index, timestamp, etc.",
    )


class SearchQuery(BaseModel):
    """Parameters for a search request."""

    text: str = Field(description="The query text to search for.")
    top_k: int = Field(default=10, description="Number of results to return.")
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional key-value filters applied to metadata fields.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold — results below this are excluded.",
    )


class SearchResult(BaseModel):
    """A single search result returned by the vector store."""

    id: str = Field(description="ID of the matched document.")
    content: str = Field(description="Text content of the matched document.")
    score: float = Field(description="Relevance score (higher is better).")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata from the matched document.",
    )


class CollectionInfo(BaseModel):
    """Stats and status for a vector store collection."""

    name: str = Field(description="Collection name.")
    points_count: int = Field(description="Number of points (documents) stored.")
    vectors_count: int = Field(default=0, description="Total number of vectors stored.")
    status: str = Field(default="unknown", description="Collection status.")

# --- V2 Structured Output Models ---

class ToolPair(BaseModel):
    """A single tool call instruction from the LLM."""
    tool_name: Literal[
        "tavily_search", "duckduckgo_search", "serpapi_search",
        "firecrawl_scrape", "bs4_scrape", "arxiv_loader", "pdf_loader"
    ] = Field(description="The tool to use.")
    query: Optional[str] = Field(None, description="Search query string (for search/loader tools).")
    url: Optional[str] = Field(None, description="URL to scrape (for scrape tools).")

class ToolMap(BaseModel):
    """LLM-generated plan of which tools to call for a topic."""
    tool_pairs: List[ToolPair] = Field(description="List of tool calls to execute.")

class Gap(BaseModel):
    """A knowledge gap identified during research."""
    query: str = Field(description="The follow-up research question to resolve this gap.")
    severity: float = Field(description="Importance of this gap on a scale of 1-10.")

class GapAnalysis(BaseModel):
    """Result of gap analysis with severity-scored gaps."""
    gaps: List[Gap] = Field(default_factory=list, description="List of knowledge gaps with severity scores.")
    is_complete: bool = Field(description="True if the original goal is fully covered by current findings.")

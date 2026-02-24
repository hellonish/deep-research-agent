"""Request/response schemas for the research API."""
from typing import Any

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    model_id: str | None = None
    config: dict | None = None
    # Scoping: optional refined plan and user context (answers to clarifying questions, edits).
    refined_plan: list[dict[str, Any]] | None = None  # List of {step_number, action, description}; if set, orchestrator skips planner
    user_context: str | None = None  # User's answers to clarifying questions / refinements; injected into scope


class ResearchResponse(BaseModel):
    job_id: str
    session_id: str
    status: str


class ResearchScopeRequest(BaseModel):
    """Request body for POST /research/scope: get plan + clarifying questions only."""
    query: str
    model_id: str | None = None
    num_plan_steps: int = 5


class ResearchScopeResponse(BaseModel):
    """Response: plan + clarifying questions. User can refine then POST /research with refined_plan + user_context."""
    query_type: str
    plan: list[dict[str, Any]]  # List of {step_number, action, description}
    clarifying_questions: list[str]

"""Research job endpoint tests (research is part of chat: /api/chat/research)."""
from unittest.mock import patch, AsyncMock

import pytest
from app.db.database import async_session
from app.db.models import ResearchJob


@pytest.mark.asyncio
async def test_start_research_job(client, test_user, auth_headers):
    """POST /api/chat/research should create a job and return job_id."""
    with patch("app.services.research_service.run_research_job", new_callable=AsyncMock):
        response = await client.post(
            "/api/chat/research",
            json={"query": "Transformer architectures", "config": {"num_plan_steps": 3}},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_get_research_result_not_found(client, test_user, auth_headers):
    """GET /api/chat/research/result/{id} should 404 for non-existent job."""
    response = await client.get("/api/chat/research/result/no-such-id", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_research_result(client, test_user, auth_headers):
    """GET /api/chat/research/result/{id} should return completed job."""
    async with async_session() as db:
        job = ResearchJob(
            user_id=test_user["user_id"],
            query="Test query",
            status="complete",
            model_id="gemini-2.0-flash",
            report_json={"title": "Report", "summary": "Done.", "blocks": []},
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_id = job.id

    response = await client.get(f"/api/chat/research/result/{job_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert response.json()["report"]["title"] == "Report"

"""Health check and basic app tests."""
import pytest


@pytest.mark.asyncio
async def test_health_check(client):
    """GET /health should return 200 with status ok."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "wort-api"

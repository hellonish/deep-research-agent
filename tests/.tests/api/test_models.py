"""Model management and API key tests."""
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_set_api_key_valid(client, test_user, auth_headers):
    """Valid key should be accepted and return models."""
    models = [{"id": "models/gemini-2.0-flash", "display_name": "Gemini 2.0 Flash",
               "description": "Fast", "input_token_limit": 1000000, "output_token_limit": 8192}]

    with patch("app.routers.models.validate_api_key", return_value=True), \
         patch("app.routers.models.list_available_models", return_value=models):
        response = await client.post(
            "/api/models/set-key", json={"api_key": "valid-key"}, headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["models"]) == 1


@pytest.mark.asyncio
async def test_set_api_key_invalid(client, test_user, auth_headers):
    """Invalid key should return 400."""
    with patch("app.routers.models.validate_api_key", return_value=False):
        response = await client.post(
            "/api/models/set-key", json={"api_key": "bad"}, headers=auth_headers,
        )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_available_models(client, test_user, auth_headers):
    """GET /api/models/available should return models."""
    models = [{"id": "models/gemini-2.0-flash", "display_name": "Flash",
               "description": "", "input_token_limit": None, "output_token_limit": None}]

    with patch("app.routers.models.list_available_models", return_value=models):
        response = await client.get("/api/models/available", headers=auth_headers)

    assert response.status_code == 200
    assert "models" in response.json()

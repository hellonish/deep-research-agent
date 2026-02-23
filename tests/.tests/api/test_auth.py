"""Google OAuth and JWT authentication tests."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
import jwt


@pytest.mark.asyncio
async def test_google_auth_success(client):
    """POST /api/auth/google with valid token should return JWT."""
    mock_payload = {
        "sub": "google-oauth-user-456",
        "email": "newuser@example.com",
        "name": "New User",
        "picture": "https://example.com/new.jpg",
    }
    with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=mock_payload):
        response = await client.post("/api/auth/google", json={"id_token": "fake-token"})

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "newuser@example.com"

    decoded = jwt.decode(data["access_token"], "test-secret-key-for-testing-32ch", algorithms=["HS256"])
    assert decoded["email"] == "newuser@example.com"
    assert "user_id" in decoded


@pytest.mark.asyncio
async def test_google_auth_invalid_token(client):
    """POST /api/auth/google with invalid token should return 401."""
    with patch("app.routers.auth.id_token.verify_oauth2_token", side_effect=ValueError("Invalid")):
        response = await client.post("/api/auth/google", json={"id_token": "bad"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_google_auth_idempotent(client):
    """Logging in twice with the same Google ID returns the same user."""
    payload = {"sub": "google-idem-789", "email": "repeat@example.com", "name": "R", "picture": ""}
    with patch("app.routers.auth.id_token.verify_oauth2_token", return_value=payload):
        r1 = await client.post("/api/auth/google", json={"id_token": "t1"})
        r2 = await client.post("/api/auth/google", json={"id_token": "t2"})
    assert r1.json()["user"]["id"] == r2.json()["user"]["id"]


@pytest.mark.asyncio
async def test_no_auth_header_rejected(client):
    """Endpoints requiring auth should return 401 without header."""
    response = await client.get("/api/history/chats")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalid_jwt_rejected(client):
    """Endpoints should reject invalid JWTs."""
    response = await client.get("/api/history/chats", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_expired_jwt_rejected(client):
    """Endpoints should reject expired JWTs."""
    expired = jwt.encode(
        {"user_id": "x", "email": "x@x.com", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
        "test-secret-key-for-testing-32ch", algorithm="HS256",
    )
    response = await client.get("/api/history/chats", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401

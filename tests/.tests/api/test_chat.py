"""Chat SSE streaming endpoint tests."""
import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.asyncio
async def test_chat_stream_no_web(client, test_user, auth_headers):
    """Chat in no-web mode should stream tokens via SSE."""
    tokens = ["Hello", " there", "!"]
    with patch("app.routers.chat.get_llm_client") as factory:
        llm = MagicMock()
        llm.generate_text_stream.return_value = iter(tokens)
        factory.return_value = llm
        response = await client.post(
            "/api/chat/stream", json={"message": "Hi", "mode": "chat"}, headers=auth_headers,
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    body = response.text
    assert "event: status" in body
    assert "event: token" in body
    assert "event: done" in body
    assert "Hello" in body


@pytest.mark.asyncio
async def test_chat_stream_creates_session(client, test_user, auth_headers):
    """Chat without session_id should create a new session."""
    with patch("app.routers.chat.get_llm_client") as factory:
        llm = MagicMock()
        llm.generate_text_stream.return_value = iter(["OK"])
        factory.return_value = llm
        response = await client.post(
            "/api/chat/stream", json={"message": "New chat", "mode": "chat"}, headers=auth_headers,
        )

    assert response.status_code == 200
    for line in response.text.split("\n"):
        if line.startswith("data:") and "session_id" in line:
            data = json.loads(line[5:].strip())
            assert "session_id" in data
            break


@pytest.mark.asyncio
async def test_chat_stream_web_mode(client, test_user, auth_headers):
    """Chat in web mode should emit searching_web status."""
    with patch("app.routers.chat.get_llm_client") as factory:
        llm = MagicMock()
        llm.generate_text_stream.return_value = iter(["Web result."])
        factory.return_value = llm
        response = await client.post(
            "/api/chat/stream", json={"message": "Search", "mode": "web"}, headers=auth_headers,
        )

    assert response.status_code == 200
    assert "searching_web" in response.text

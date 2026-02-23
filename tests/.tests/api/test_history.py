"""Chat session and message history tests."""
import pytest
from app.db.database import async_session
from app.db.models import ChatSession


@pytest.mark.asyncio
async def test_list_chats_empty(client, auth_headers):
    """GET /api/history/chats should return empty list for new user."""
    response = await client.get("/api/history/chats", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_research_empty(client, auth_headers):
    """GET /api/history/research should return empty list."""
    response = await client.get("/api/history/research", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_get_messages_nonexistent(client, auth_headers):
    """GET /api/history/chats/{id}/messages should 404 for non-existent session."""
    response = await client.get("/api/history/chats/no-such-id/messages", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_chats_with_data(client, test_user, auth_headers):
    """After creating a session, it should appear in the chat list."""
    async with async_session() as db:
        session = ChatSession(user_id=test_user["user_id"], title="Test Chat")
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session_id = session.id

    response = await client.get("/api/history/chats", headers=auth_headers)
    assert response.status_code == 200
    assert any(c["id"] == session_id for c in response.json())

    # Messages should be empty
    response = await client.get(f"/api/history/chats/{session_id}/messages", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []

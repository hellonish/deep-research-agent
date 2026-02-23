"""Security tests: user isolation, rate limiting, crypto."""
import pytest
from unittest.mock import AsyncMock

from app.db.database import async_session
from app.db.models import User, ChatSession, ResearchJob


@pytest.mark.asyncio
async def test_user_cannot_access_other_users_session(client, test_user, auth_headers):
    """User A cannot read User B's chat messages."""
    async with async_session() as db:
        other = User(google_id="other-1", email="other1@example.com", name="Other")
        db.add(other)
        await db.commit()
        await db.refresh(other)
        session = ChatSession(user_id=other.id, title="Secret")
        db.add(session)
        await db.commit()
        await db.refresh(session)
        sid = session.id

    response = await client.get(f"/api/history/chats/{sid}/messages", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_user_cannot_access_other_users_research(client, test_user, auth_headers):
    """User A cannot read User B's research job."""
    async with async_session() as db:
        other = User(google_id="other-2", email="other2@example.com", name="Other")
        db.add(other)
        await db.commit()
        await db.refresh(other)
        job = ResearchJob(user_id=other.id, query="Secret", status="complete",
                          report_json={"title": "Classified"})
        db.add(job)
        await db.commit()
        await db.refresh(job)
        jid = job.id

    response = await client.get(f"/api/chat/research/result/{jid}", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_rate_limiting():
    """Rate limiter should reject after threshold."""
    from app.middleware.rate_limit import check_rate_limit
    from fastapi import HTTPException

    storage = {}
    mock = AsyncMock()

    async def mock_incr(key):
        storage[key] = storage.get(key, 0) + 1
        return storage[key]

    mock.incr = mock_incr
    mock.expire = AsyncMock()

    for _ in range(3):
        await check_rate_limit("user", mock, "ep", max_per_hour=3)

    with pytest.raises(HTTPException) as exc:
        await check_rate_limit("user", mock, "ep", max_per_hour=3)
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_crypto_round_trip():
    """Encrypt → decrypt should recover the original key."""
    from app.services.crypto_service import encrypt_api_key, decrypt_api_key

    original = "AIzaSyD-test-key-12345"
    encrypted = encrypt_api_key(original)
    assert encrypted != original
    assert decrypt_api_key(encrypted) == original


@pytest.mark.asyncio
async def test_crypto_unique_ciphertexts():
    """Same plaintext should produce different ciphertexts (random IV)."""
    from app.services.crypto_service import encrypt_api_key

    e1 = encrypt_api_key("same-key")
    e2 = encrypt_api_key("same-key")
    assert e1 != e2

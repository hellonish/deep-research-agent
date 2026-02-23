"""
Shared fixtures for all API tests.

Provides: app, client, test_user, auth_headers, mock_redis.
"""
import asyncio
import os
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# ── Patch settings BEFORE importing the app ──────────────────────────
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_wort.db"
os.environ["JWT_SECRET"] = "test-secret-key-for-testing-32ch"
os.environ["GOOGLE_CLIENT_ID"] = "test-google-client-id"
os.environ["ENCRYPTION_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = "test-server-api-key"

from app.main import create_app
from app.db.database import engine, Base, async_session
from app.dependencies import create_jwt
from app.db.models import User
from app.services.memory_service import get_redis


# ── In-memory mock Redis ─────────────────────────────────────────────

def create_mock_redis():
    """Creates a fresh in-memory mock Redis."""
    storage = {}
    lists = {}
    mock = AsyncMock()

    async def mock_get(key):
        return storage.get(key)

    async def mock_setex(key, ttl, value):
        storage[key] = value if isinstance(value, bytes) else value.encode()

    async def mock_rpush(key, value):
        lists.setdefault(key, []).append(value)

    async def mock_lrange(key, start, end):
        items = lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

    async def mock_ltrim(key, start, end):
        if key in lists:
            lists[key] = lists[key][start:] if end == -1 else lists[key][start:end + 1]

    async def mock_expire(key, ttl):
        pass

    async def mock_incr(key):
        val = int(storage.get(key, b"0"))
        val += 1
        storage[key] = str(val).encode()
        return val

    async def mock_hset(key, mapping=None):
        storage[key] = mapping or {}

    async def mock_hgetall(key):
        return storage.get(key, {})

    mock.get = mock_get
    mock.setex = mock_setex
    mock.rpush = mock_rpush
    mock.lrange = mock_lrange
    mock.ltrim = mock_ltrim
    mock.expire = mock_expire
    mock.incr = mock_incr
    mock.hset = mock_hset
    mock.hgetall = mock_hgetall
    return mock


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def app():
    """Create app with mock Redis injected as a dependency override."""
    test_app = create_app()

    # Override Redis dependency globally for all tests
    mock = create_mock_redis()

    async def override_redis():
        return mock

    test_app.dependency_overrides[get_redis] = override_redis

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield test_app

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    try:
        os.remove("./test_wort.db")
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user():
    """Create a unique test user for each test."""
    unique = uuid.uuid4().hex[:8]
    async with async_session() as db:
        user = User(
            google_id=f"google-test-{unique}",
            email=f"test-{unique}@example.com",
            name="Test User",
            picture="https://example.com/pic.jpg",
            selected_model="gemini-2.0-flash",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        token = create_jwt(user_id=user.id, email=user.email)
        yield {"user": user, "token": token, "user_id": user.id}


@pytest.fixture
def auth_headers(test_user):
    return {"Authorization": f"Bearer {test_user['token']}"}


@pytest_asyncio.fixture
async def mock_redis():
    """Standalone mock Redis for unit tests that don't go through FastAPI."""
    return create_mock_redis()

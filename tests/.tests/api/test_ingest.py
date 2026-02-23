"""Ingest endpoint tests."""
import io
from unittest.mock import patch, AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_upload_txt_file(client, test_user, auth_headers):
    """POST /api/ingest/upload with a .txt file should return chunk count."""
    mock_store = AsyncMock()
    mock_store.collection_exists = AsyncMock(return_value=False)
    mock_store.create_collection = AsyncMock()
    mock_store.upsert = AsyncMock()

    with patch("app.routers.ingest.QdrantStore", return_value=mock_store):
        content = b"Hello world. This is a test document with enough text to be meaningful."
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "test.txt"
    assert data["chunks_created"] >= 1
    assert "collection" in data


@pytest.mark.asyncio
async def test_upload_unsupported_extension(client, test_user, auth_headers):
    """POST /api/ingest/upload with unsupported file type should return 400."""
    response = await client.post(
        "/api/ingest/upload",
        files={"file": ("test.exe", io.BytesIO(b"binary data"), "application/octet-stream")},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_oversized_file(client, test_user, auth_headers):
    """POST /api/ingest/upload with file > 20MB should return 413."""
    # Create a fake large payload (just over 20MB)
    large_content = b"x" * (20 * 1024 * 1024 + 1)
    response = await client.post(
        "/api/ingest/upload",
        files={"file": ("big.txt", io.BytesIO(large_content), "text/plain")},
        headers=auth_headers,
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_html_file(client, test_user, auth_headers):
    """POST /api/ingest/upload with an .html file should strip tags and chunk."""
    mock_store = AsyncMock()
    mock_store.collection_exists = AsyncMock(return_value=True)
    mock_store.upsert = AsyncMock()

    html_content = b"<html><body><h1>Title</h1><p>Some paragraph content.</p></body></html>"
    with patch("app.routers.ingest.QdrantStore", return_value=mock_store):
        response = await client.post(
            "/api/ingest/upload",
            files={"file": ("page.html", io.BytesIO(html_content), "text/html")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "page.html"
    assert data["chunks_created"] >= 1


@pytest.mark.asyncio
async def test_upload_custom_collection(client, test_user, auth_headers):
    """POST /api/ingest/upload should respect the collection query param."""
    mock_store = AsyncMock()
    mock_store.collection_exists = AsyncMock(return_value=False)
    mock_store.create_collection = AsyncMock()
    mock_store.upsert = AsyncMock()

    with patch("app.routers.ingest.QdrantStore", return_value=mock_store):
        response = await client.post(
            "/api/ingest/upload?collection=my_custom_collection",
            files={"file": ("doc.txt", io.BytesIO(b"Custom collection test."), "text/plain")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert response.json()["collection"] == "my_custom_collection"


@pytest.mark.asyncio
async def test_upload_empty_file(client, test_user, auth_headers):
    """POST /api/ingest/upload with an empty file should return 422."""
    response = await client.post(
        "/api/ingest/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        headers=auth_headers,
    )

    assert response.status_code == 422

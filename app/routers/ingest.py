"""
Ingest router — upload files (PDF, TXT, MD, HTML, DOCX) to be parsed,
chunked, and upserted into the user's Qdrant vector store collection.
"""
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.schemas.ingest import IngestResponse
from tools.loaders import (
    PDFLoader,
    TextLoader,
    HTMLLoader,
    DocxLoader,
    chunk_documents,
)
from vector_store.qdrant_store import QdrantStore

router = APIRouter(prefix="/ingest", tags=["ingest"])

# ── Constants ────────────────────────────────────────────────────────
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".docx"}

LOADER_MAP = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".docx": DocxLoader,
}


@router.post("/upload", response_model=IngestResponse)
async def upload_file(
    file: UploadFile = File(...),
    collection: str | None = None,
    user_id: str = Depends(get_current_user),
):
    """
    Upload a document file, parse its text content, chunk it, and upsert
    the chunks into the user's Qdrant collection.

    Supported formats: PDF, TXT, MD, HTML, DOCX.
    Max file size: 20 MB.
    """
    # ── Validate extension ───────────────────────────────────────────
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # ── Read and validate size ───────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {MAX_FILE_SIZE} bytes ({MAX_FILE_SIZE // (1024*1024)} MB).",
        )

    # ── Write to temp file for loaders ───────────────────────────────
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(content)
        tmp.close()

        # ── Parse with the right loader ──────────────────────────────
        loader_cls = LOADER_MAP[ext]
        loader = loader_cls()
        documents = await loader.load(source=tmp.name)

        if not documents:
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the uploaded file.",
            )

        # ── Chunk ────────────────────────────────────────────────────
        chunked_docs = chunk_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200,
            source_filename=filename,
        )

        if not chunked_docs:
            raise HTTPException(
                status_code=422,
                detail="File parsed but produced no chunks. The file may be empty.",
            )

        # ── Upsert to Qdrant ─────────────────────────────────────────
        target_collection = collection or f"user_{user_id}_docs"

        location = (settings.QDRANT_LOCATION or "").strip()
        use_in_memory = location in ("", ":memory:") or location.startswith("path:")
        if use_in_memory:
            store = QdrantStore(in_memory=True)
        else:
            store = QdrantStore(url=location, api_key=settings.QDRANT_API_KEY or None)

        # Ensure collection exists
        if not await store.collection_exists(target_collection):
            await store.create_collection(target_collection)

        await store.upsert(target_collection, chunked_docs)

        return IngestResponse(
            file_name=filename,
            file_size=len(content),
            chunks_created=len(chunked_docs),
            collection=target_collection,
        )

    finally:
        # Clean up temp file
        os.unlink(tmp.name)

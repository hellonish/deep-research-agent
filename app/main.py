"""
FastAPI application factory — the main entry point for the Wort backend.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.database import init_db
from app.routers import auth, chat, history, models, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # ── Startup ──────────────────────────────────────────────────────
    print("[Wort] Initializing database...")
    await init_db()
    print("[Wort] Database ready.")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────
    print("[Wort] Shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Wort — Deep Research Engine",
        description="AI-powered research platform with Chat and Research modes",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS (restricted to frontend origin) ─────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routers ─────────────────────────────────────────────
    app.include_router(auth.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")  # chat + research (research is part of chat)
    app.include_router(history.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(ingest.router, prefix="/api")

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "wort-api"}

    # Vector store connection check (for verifying cloud vs in-memory)
    @app.get("/health/vector")
    async def health_vector():
        from vector_store.qdrant_store import QdrantStore

        location = (settings.QDRANT_LOCATION or "").strip()
        use_in_memory = location in ("", ":memory:") or location.startswith("path:")
        if use_in_memory:
            store = QdrantStore(in_memory=True)
            return {
                "status": "ok",
                "vector_store": "in_memory",
                "location": ":memory:",
                "connected": True,
            }
        store = QdrantStore(url=location, api_key=settings.QDRANT_API_KEY or None)
        connected = await store.check_connection()
        return {
            "status": "ok" if connected else "degraded",
            "vector_store": "remote",
            "location": store.location,
            "connected": connected,
        }

    return app


app = create_app()

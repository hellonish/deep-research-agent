"""
Application configuration — loads all settings from environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./wort.db")

    # ── Redis ────────────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ── Google OAuth ─────────────────────────────────────────────────
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    # ── JWT ───────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_DAYS: int = 7

    # ── Encryption (Fernet key for API key storage) ──────────────────
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")

    # ── Gemini (server default for free-tier users) ──────────────────
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

    # ── CORS ─────────────────────────────────────────────────────────
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # ── Qdrant (vector store for research and ingest) ─────────────────
    QDRANT_LOCATION: str = os.getenv("QDRANT_LOCATION", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # ── Curated models (intersection with API key) ────────────────────
    # Only these model IDs are shown in the app; user sees (this list ∩ their key's models).
    CURATED_MODEL_IDS: tuple = (
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview-05-06",
        "gemini-exp-1206",
    )


settings = Settings()

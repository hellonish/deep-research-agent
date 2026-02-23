"""
Async SQLAlchemy engine and session factory.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _add_missing_user_columns(conn):
    """Add columns to users table if they were added to the model after the table was created."""
    try:
        conn.execute(text("ALTER TABLE users ADD COLUMN metadata_ TEXT"))
    except Exception as e:
        if "duplicate column name" in str(e).lower():
            pass  # Column already exists
        else:
            raise


async def init_db():
    """Create all tables. Called once on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_user_columns)


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields an async database session."""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

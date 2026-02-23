"""
SQLAlchemy ORM models for the Wort application.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, func
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import relationship

from app.db.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Authenticated user (via Google OAuth)."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    google_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    picture = Column(String, nullable=True)
    gemini_api_key = Column(String, nullable=True)  # Fernet-encrypted
    selected_model = Column(String, default="gemini-2.0-flash")
    metadata_ = Column(JSON, nullable=True)  # Optional: {"preferences": {}, "facts": []}
    created_at = Column(DateTime, default=func.now())

    # Relationships
    sessions = relationship("ChatSession", back_populates="user")
    research_jobs = relationship("ResearchJob", back_populates="user")


class ChatSession(Base):
    """A single conversation thread (can include chat + research)."""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=True)  # Auto-generated from first message
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", order_by="Message.created_at")
    research_jobs = relationship("ResearchJob", back_populates="session")


class Message(Base):
    """A single message within a chat session."""
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    mode = Column(String, default="chat")  # "chat" | "web" | "research"
    sources = Column(JSON, nullable=True)  # URLs used in web mode
    created_at = Column(DateTime, default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class ResearchJob(Base):
    """A deep research job linked to a chat session."""
    __tablename__ = "research_jobs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=True, index=True)
    query = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending | running | complete | failed
    model_id = Column(String, nullable=True)
    config_json = Column(JSON, nullable=True)
    report_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="research_jobs")
    session = relationship("ChatSession", back_populates="research_jobs")

"""
Core application config and dependencies.
"""
from app.core.config import settings
from app.core.dependencies import create_jwt, get_current_user

__all__ = ["settings", "create_jwt", "get_current_user"]

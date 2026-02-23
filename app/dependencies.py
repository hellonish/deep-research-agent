"""Re-export for backward compatibility. Prefer: from app.core.dependencies import ..."""
from app.core.dependencies import create_jwt, get_current_user

__all__ = ["create_jwt", "get_current_user"]

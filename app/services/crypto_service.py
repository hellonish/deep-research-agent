"""
Fernet-based symmetric encryption for storing API keys at rest.
"""
import os
from cryptography.fernet import Fernet
from app.config import settings

_fernet = None

def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = settings.ENCRYPTION_KEY
        if not key:
            key = Fernet.generate_key().decode()
            print(f"[WARN] No ENCRYPTION_KEY set. Generated ephemeral key: {key}")
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    return _fernet


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for database storage."""
    return get_fernet().encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key retrieved from the database."""
    return get_fernet().decrypt(encrypted_key.encode()).decode()

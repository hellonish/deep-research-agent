"""
Model discovery service — lists available Gemini models for a given API key.

Returns only the intersection of: (1) Wort's curated model list, (2) models
the user's API key can access.
"""
from google import genai

from app.core.config import settings


def _normalize_model_id(name: str) -> str:
    """Strip 'models/' prefix for consistent comparison."""
    if not name:
        return ""
    return name.replace("models/", "", 1).strip()


def list_available_models(api_key: str) -> list[dict]:
    """
    Returns models that are both in our curated list and accessible with this API key.

    Args:
        api_key: The user's Gemini API key.

    Returns:
        List of model info dicts: id, display_name, description, token limits.
    """
    client = genai.Client(api_key=api_key)
    all_models = client.models.list()
    curated_ids = {_normalize_model_id(mid) for mid in settings.CURATED_MODEL_IDS}

    available = []
    for model in all_models:
        name = (model.name or "").strip()
        normalized = _normalize_model_id(name)
        if normalized not in curated_ids:
            continue
        if "embedding" in name.lower():
            continue
        display_name = getattr(model, "display_name", name)
        available.append({
            "id": normalized,
            "name": display_name,
            "display_name": display_name,
            "description": getattr(model, "description", ""),
            "input_token_limit": getattr(model, "input_token_limit", None),
            "output_token_limit": getattr(model, "output_token_limit", None),
        })

    available.sort(key=lambda m: str(m.get("display_name", m["id"])))
    return available


def validate_api_key(api_key: str) -> bool:
    """
    Quick validation: tries to list models. If it fails, key is invalid.

    Args:
        api_key: The Gemini API key to validate.

    Returns:
        True if the key is valid, False otherwise.
    """
    try:
        client = genai.Client(api_key=api_key)
        # Force evaluation by consuming at least one item
        models = list(client.models.list())
        return len(models) > 0
    except Exception:
        return False

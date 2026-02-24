"""
Model discovery and key validation per provider (Gemini, DeepSeek, etc.).
"""
from google import genai

from app.core.config import settings


def _normalize_model_id(name: str) -> str:
    """Strip 'models/' prefix for consistent comparison."""
    if not name:
        return ""
    return name.replace("models/", "", 1).strip()


# ── Gemini ───────────────────────────────────────────────────────────

def list_gemini_models(api_key: str) -> list[dict]:
    """Models from curated list that this Gemini API key can access."""
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
            "provider": "gemini",
        })

    available.sort(key=lambda m: str(m.get("display_name", m["id"])))
    return available


def validate_gemini_key(api_key: str) -> bool:
    """Validate Gemini API key by listing models."""
    try:
        client = genai.Client(api_key=api_key)
        models = list(client.models.list())
        return len(models) > 0
    except Exception:
        return False


# ── DeepSeek (static list; no list API) ───────────────────────────────

def list_deepseek_models(_api_key: str) -> list[dict]:
    """Return curated DeepSeek models. Key is not used for listing."""
    return [
        {"id": mid, "name": mid, "display_name": mid, "description": "", "provider": "deepseek"}
        for mid in settings.DEEPSEEK_MODEL_IDS
    ]


def validate_deepseek_key(api_key: str) -> bool:
    """Validate DeepSeek API key (non-empty; optional: minimal API call)."""
    if not (api_key and api_key.strip()):
        return False
    try:
        from deepseek import DeepSeekAPI
        client = DeepSeekAPI(api_key=api_key.strip())
        # Minimal completion to verify key
        out = client.chat_completion(prompt="Hi", prompt_sys="You say OK.", model="deepseek-chat", stream=False)
        return bool(out and out.strip())
    except Exception:
        return False


# ── Generic ───────────────────────────────────────────────────────────

def list_available_models(api_key: str) -> list[dict]:
    """Legacy: Gemini-only. Prefer list_models_for_provider."""
    return list_gemini_models(api_key)


def list_models_for_provider(provider: str, api_key: str) -> list[dict]:
    """List models for a given provider and key."""
    if provider == "gemini":
        return list_gemini_models(api_key)
    if provider == "deepseek":
        return list_deepseek_models(api_key)
    return []


def validate_api_key(api_key: str, provider: str = "gemini") -> bool:
    """Validate API key for the given provider."""
    if provider == "gemini":
        return validate_gemini_key(api_key)
    if provider == "deepseek":
        return validate_deepseek_key(api_key)
    return False

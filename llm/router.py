"""
Gemini-only LLM router — creates a GeminiClient with the user's API key and model.
"""
from llm.gemini import GeminiClient


def get_llm_client(model_id: str, api_key: str | None = None) -> GeminiClient:
    """
    Factory: returns a GeminiClient initialized with the user's API key
    and their selected model.

    Args:
        model_id: The Gemini model name (e.g., "gemini-2.0-flash" or "models/gemini-2.5-pro").
        api_key: The user's Gemini API key. If None, falls back to env var.

    Returns:
        GeminiClient configured for the specified model.
    """
    # Normalize: SDK expects "models/..." format for some operations,
    # but generate_content works with both formats
    if model_id.startswith("models/"):
        model_id = model_id.replace("models/", "", 1)

    return GeminiClient(model_name=model_id, api_key=api_key)

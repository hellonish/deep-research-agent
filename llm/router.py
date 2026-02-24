"""
LLM router — returns the appropriate client (Gemini or DeepSeek) for the given model_id.
"""
from llm.base import BaseLLMClient
from llm.gemini import GeminiClient
from llm.deepseek import DeepSeekClient


def get_llm_client(model_id: str, api_key: str | None = None) -> BaseLLMClient:
    """
    Factory: returns GeminiClient or DeepSeekClient based on model_id.

    Args:
        model_id: Model identifier (e.g. "gemini-2.5-flash", "deepseek-chat").
        api_key: The user's API key for that provider. If None, client may use env default.

    Returns:
        Configured LLM client for the model's provider.
    """
    if model_id.startswith("models/"):
        model_id = model_id.replace("models/", "", 1)
    n = model_id.lower().strip()
    if n.startswith("deepseek-"):
        return DeepSeekClient(model_name=model_id, api_key=api_key)
    return GeminiClient(model_name=model_id, api_key=api_key)

"""Request/response schemas for the models API."""
from pydantic import BaseModel


class SetKeyRequest(BaseModel):
    api_key: str
    provider: str = "gemini"  # gemini | deepseek | openai


class SetKeyResponse(BaseModel):
    valid: bool
    models: list[dict]
    provider: str | None = None


class ProviderKeyItem(BaseModel):
    provider: str
    label: str


class ProviderKeysResponse(BaseModel):
    keys: list[ProviderKeyItem]


class ModelsResponse(BaseModel):
    models: list[dict]


class KeyStatusResponse(BaseModel):
    configured: bool


class SetModelRequest(BaseModel):
    model_id: str


class SetModelResponse(BaseModel):
    success: bool
    selected_model: str

import json
import os
from typing import Generator

from deepseek import DeepSeekAPI
from pydantic import BaseModel

from .base import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    def __init__(self, model_name: str = "deepseek-chat", api_key: str | None = None):
        """
        Initialize the DeepSeek client (uses the deepseek package's DeepSeekAPI).

        Args:
            model_name: Model ID (e.g. deepseek-chat, deepseek-reasoner).
            api_key: Optional API key; falls back to DEEPSEEK_API_KEY env var.
        """
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.client = DeepSeekAPI(api_key=key)
        self.model_name = model_name

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Requests JSON from DeepSeek and parses the response into the given Pydantic schema.
        The required output shape is derived from the schema and injected into the instruction
        so the model always returns the exact field names and types.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"{(system_prompt or '').strip()}\n\n"
            f"You MUST respond with a single valid JSON object that exactly matches this schema:\n"
            f"{schema_json}\n\n"
            f"Use the exact field names from the schema. Output raw JSON only — no markdown, no code fences."
        ).strip()

        completion_kw: dict = {
            "prompt": prompt,
            "prompt_sys": instruction,
            "model": self.model_name,
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
        }
        if max_tokens is not None:
            # DeepSeek API valid range is [1, 8192]; clamp to avoid 400
            completion_kw["max_tokens"] = min(max(1, max_tokens), 8192)
        content = self.client.chat_completion(**completion_kw)
        if not content or not content.strip():
            raise ValueError("DeepSeek returned empty content for structured output")
        data = json.loads(content)
        return schema.model_validate(data)

    def generate_text(
        self, prompt: str, system_prompt: str, temperature: float = 0.5
    ) -> str:
        """
        Generates a text response from the LLM.
        """
        out = self.client.chat_completion(
            prompt=prompt,
            prompt_sys=system_prompt or "",
            model=self.model_name,
            stream=False,
            temperature=temperature,
        )
        return out or ""

    def generate_text_stream(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.5,
    ) -> Generator[str, None, None]:
        """
        Streams a text response token-by-token.
        """
        stream = self.client.chat_completion(
            prompt=prompt,
            prompt_sys=system_prompt or "",
            model=self.model_name,
            stream=True,
            temperature=temperature,
        )
        yield from stream
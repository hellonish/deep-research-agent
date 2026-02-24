from typing import Generator

from google import genai
from google.genai import types
from .base import BaseLLMClient
from pydantic import BaseModel


class GeminiClient(BaseLLMClient):
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None):
        """
        Initialize the Gemini client.

        Args:
            model_name: The Gemini model to use.
            api_key: Optional API key. If None, falls back to GOOGLE_API_KEY env var.
        """
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
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
        Calls the Gemini API and returns a parsed Pydantic object matching the schema.
        """
        config_kw: dict = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }
        if max_tokens is not None:
            config_kw["max_output_tokens"] = max_tokens
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kw),
        )
        return response.parsed

    def generate_text(self, prompt: str, system_prompt: str, temperature: float = 0.5) -> str:
        """
        Generates a text response from the LLM.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        return response.text

    def generate_text_stream(self, prompt: str, system_prompt: str, temperature: float = 0.5) -> Generator[str, None, None]:
        """
        Streams a text response token-by-token for real-time chat SSE.

        Yields:
            str: Individual text chunks as they arrive from the LLM.
        """
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
            ),
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
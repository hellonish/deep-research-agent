import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from the .env file
load_dotenv()

class BaseLLMClient:

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
        temperature: float = 0.5,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Generates a structured output based on the provided Pydantic schema.

        Args:
            prompt: The user's prompt or query.
            system_prompt: The system instructions for the LLM.
            schema: The Pydantic model class to structure the output.
            temperature: The temperature to use for the LLM.
            max_tokens: Optional max tokens for the response (enables longer reports).

        Returns:
            An instance of the provided Pydantic schema populated with the LLM's response.
        """
        raise NotImplementedError("Subclasses must implement this method")
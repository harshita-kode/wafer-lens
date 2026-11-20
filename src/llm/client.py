"""One small interface over several LLM providers: generate(system, user) -> text."""
import os
from typing import Protocol

from src.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


def _require_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return key


class GeminiClient:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=_require_key("GEMINI_API_KEY"))
        self._types = types
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return (response.text or "").strip()


class AnthropicClient:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        import anthropic

        self._client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()


class OpenAIClient:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=_require_key("OPENAI_API_KEY"))
        self.model, self.max_tokens = model, max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()


_PROVIDERS = {"gemini": GeminiClient, "anthropic": AnthropicClient, "openai": OpenAIClient}


def get_llm(provider: str = LLM_PROVIDER, model: str = LLM_MODEL) -> LLMClient:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of {list(_PROVIDERS)}")
    return _PROVIDERS[provider](model, LLM_TEMPERATURE, LLM_MAX_TOKENS)


if __name__ == "__main__":
    llm = get_llm()
    print(f"Provider: {LLM_PROVIDER}   Model: {LLM_MODEL}")
    print(llm.generate("You answer in one short sentence.", "What is a silicon wafer?"))
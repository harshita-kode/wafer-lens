"""One small interface over several LLM providers: generate(system, user) -> text."""
import os
from typing import Protocol
import logging
import time
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, LLM_CACHE_ENABLED, LLM_CACHE_DIR


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


def _require_key(name: str) -> str:
    key = (os.getenv(name) or "").strip()
    if not key:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return key

class TruncatedResponseError(RuntimeError):
    """The model hit its output limit before finishing the answer."""

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
        candidate = (response.candidates or [None])[0]
        if candidate is not None and "MAX_TOKENS" in str(candidate.finish_reason):
            raise TruncatedResponseError(f"response cut off at {self.max_tokens} output tokens")
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
        if response.stop_reason == "max_tokens":
            raise TruncatedResponseError(f"response cut off at {self.max_tokens} output tokens")
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
        if response.choices[0].finish_reason == "length":
            raise TruncatedResponseError(f"response cut off at {self.max_tokens} output tokens")
        return (response.choices[0].message.content or "").strip()

logger = logging.getLogger(__name__)

RETRYABLE_MARKERS = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED",
                     "UNAVAILABLE", "overloaded", "timeout", "timed out")


def _is_retryable(error: Exception) -> bool:
    message = str(error)
    return any(marker.lower() in message.lower() for marker in RETRYABLE_MARKERS)


class RetryingLLM:
    """Wraps any LLM client and retries temporary failures with exponential backoff."""

    def __init__(self, inner: LLMClient, retries: int = 3, base_delay: float = 2.0,
                 sleep=time.sleep):
        self.inner, self.retries, self.base_delay, self.sleep = inner, retries, base_delay, sleep

    def generate(self, system: str, user: str) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self.inner.generate(system, user)
            except Exception as e:
                if attempt == self.retries or not _is_retryable(e):
                    raise
                delay = self.base_delay * 2 ** attempt
                logger.warning("LLM call failed (%s). Retrying in %.0fs...", type(e).__name__, delay)
                self.sleep(delay)


import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class CachedLLM:
    """Wraps any LLM client and stores responses on disk, keyed by a hash of the full prompt."""

    def __init__(self, inner: LLMClient, cache_dir: Path, model: str):
        self.inner, self.cache_dir, self.model = inner, Path(cache_dir), model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.model}\n{system}\n{user}".encode("utf-8")).hexdigest()

    def generate(self, system: str, user: str) -> str:
        path = self.cache_dir / f"{self._key(system, user)}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        self.misses += 1
        response = self.inner.generate(system, user)
        if response:                                  # never cache empty answers
            path.write_text(json.dumps({
                "model": self.model,
                "created": datetime.now(timezone.utc).isoformat(),
                "response": response,
            }, indent=2), encoding="utf-8")
        return response

_PROVIDERS = {"gemini": GeminiClient, "anthropic": AnthropicClient, "openai": OpenAIClient}


def get_llm(provider: str = LLM_PROVIDER, model: str = LLM_MODEL,
            use_cache: bool = LLM_CACHE_ENABLED) -> LLMClient:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of {list(_PROVIDERS)}")
    llm: LLMClient = RetryingLLM(_PROVIDERS[provider](model, LLM_TEMPERATURE, LLM_MAX_TOKENS))
    if use_cache:
        llm = CachedLLM(llm, LLM_CACHE_DIR, f"{provider}:{model}")
    return llm


if __name__ == "__main__":
    llm = get_llm()
    print(f"Provider: {LLM_PROVIDER}   Model: {LLM_MODEL}")
    print(llm.generate("You answer in one short sentence.", "What is a silicon wafer?"))
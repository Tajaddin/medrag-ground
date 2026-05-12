"""LLM adapters. Same three-tier pattern used across the portfolio."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CompletionResult:
    text: str
    model: str = ""


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600) -> CompletionResult: ...


class MockLLMClient:
    """Deterministic LLM for tests."""

    def __init__(self, model: str = "mock", handler=None, canned: dict[str, str] | None = None) -> None:
        self.model = model
        self.handler = handler
        self.canned = canned or {}
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600) -> CompletionResult:
        self.calls.append((prompt, system))
        if self.handler is not None:
            return CompletionResult(text=self.handler(prompt, system), model=self.model)
        for needle, ans in self.canned.items():
            if needle in prompt:
                return CompletionResult(text=ans, model=self.model)
        return CompletionResult(text="", model=self.model)


class GroqClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, min_interval_secs: float = 2.1) -> None:
        from groq import Groq  # lazy

        self.model = model or os.environ.get("MEDRAG_GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.min_interval_secs = min_interval_secs
        self._last_t = 0.0

    def _wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last_t
        if delta < self.min_interval_secs:
            time.sleep(self.min_interval_secs - delta)
        self._last_t = time.monotonic()

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600) -> CompletionResult:
        from groq import RateLimitError

        backoff = 4.0
        for attempt in range(5):
            self._wait()
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens, temperature=0.2
                )
                return CompletionResult(text=resp.choices[0].message.content or "", model=self.model)
            except RateLimitError:
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
        raise RuntimeError("unreachable")


class AnthropicClient:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic  # lazy

        self.model = model or os.environ.get("MEDRAG_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 600) -> CompletionResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return CompletionResult(text=text, model=self.model)

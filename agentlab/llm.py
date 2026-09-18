"""Thin LLM client with retries, timeouts and token accounting.

Kept deliberately small: the agent loops depend on `LLMClient.complete`
returning a `Completion`, nothing else. Swap the backend by writing another
class with the same `complete` signature.
"""

from __future__ import annotations

import os
import time
import random
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Rough USD per million tokens. Update when pricing changes.
PRICING = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1e6


@dataclass
class Completion:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Anthropic-backed client. Retries on transient errors with jittered backoff."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 4,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._anthropic = anthropic
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.total_usage = Usage()

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**kwargs)
                break
            except (
                self._anthropic.RateLimitError,
                self._anthropic.APIConnectionError,
                self._anthropic.InternalServerError,
            ) as exc:
                last_exc = exc
                sleep = min(2**attempt + random.random(), 30.0)
                logger.warning(
                    "LLM call failed (%s), retry %d/%d in %.1fs",
                    type(exc).__name__, attempt + 1, self.max_retries, sleep,
                )
                time.sleep(sleep)
        else:
            raise LLMError(f"exhausted {self.max_retries} retries") from last_exc

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        self.total_usage = self.total_usage + usage

        return Completion(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=resp.stop_reason,
        )

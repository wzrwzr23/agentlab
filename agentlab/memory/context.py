"""Context management.

Three strategies so you can ablate them against each other — that comparison
is the interesting result, not the code:

    none       keep everything (baseline; fails on long tasks)
    trim       drop the oldest tool results, keep all reasoning turns
    summarize  fold the oldest exchanges into a running summary via the LLM

`estimate_tokens` is a chars/4 heuristic. That is fine for budgeting decisions
and wrong for reporting — report real usage from the API instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Strategy = Literal["none", "trim", "summarize"]

SUMMARY_PROMPT = (
    "Condense the following agent transcript into a factual summary. Keep: "
    "findings, tool results that matter, dead ends already tried. Drop: "
    "restatements and filler. Be terse; this replaces the transcript."
)


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                total += len(str(block.get("text", "") or block.get("content", "")))
    return total // 4


@dataclass
class ContextStats:
    compactions: int = 0
    dropped_messages: int = 0
    summary_tokens_spent: int = 0


class ContextManager:
    def __init__(
        self,
        strategy: Strategy = "trim",
        max_tokens: int = 24_000,
        keep_recent: int = 6,
        llm: Any = None,
    ):
        if strategy == "summarize" and llm is None:
            raise ValueError("summarize strategy needs an llm client")
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.llm = llm
        self.summary: str | None = None
        self.stats = ContextStats()

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return messages that fit the budget. Never mutates the input."""
        if self.strategy == "none" or estimate_tokens(messages) <= self.max_tokens:
            return self._with_summary(messages)

        # Always keep the first user message: it carries the task.
        head, tail = messages[:1], messages[1:]
        recent = tail[-self.keep_recent:] if self.keep_recent else []
        older = tail[: len(tail) - len(recent)]
        if not older:
            return self._with_summary(messages)

        self.stats.compactions += 1
        self.stats.dropped_messages += len(older)

        if self.strategy == "summarize":
            self._fold(older)
        else:
            logger.info("context: dropped %d older messages", len(older))

        return self._with_summary(head + recent)

    # -- internals --------------------------------------------------------

    def _fold(self, older: list[dict[str, Any]]) -> None:
        transcript = []
        for msg in older:
            content = msg.get("content")
            text = content if isinstance(content, str) else str(content)
            transcript.append(f"{msg['role']}: {text[:1500]}")

        prior = f"Existing summary:\n{self.summary}\n\n" if self.summary else ""
        completion = self.llm.complete(
            messages=[{"role": "user", "content": prior + "\n".join(transcript)}],
            system=SUMMARY_PROMPT,
            max_tokens=1024,
        )
        self.summary = completion.text.strip()
        self.stats.summary_tokens_spent += (
            completion.usage.input_tokens + completion.usage.output_tokens
        )

    def _with_summary(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.summary:
            return list(messages)
        note = {"role": "user", "content": f"[Summary of earlier work]\n{self.summary}"}
        return [messages[0], note, *messages[1:]] if messages else [note]

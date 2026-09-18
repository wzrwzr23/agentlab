"""Agent base: step budget, structured trace, typed result.

Every agent records a trace of typed steps. The trace is what the eval harness
reads to compute tool-error rate and classify failures, so treat it as the
agent's real output — the final answer is only one field of it.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

from ..llm import LLMClient, Usage
from ..tools.base import ToolRegistry

StepKind = Literal["think", "tool_call", "tool_result", "plan", "delegate", "answer", "error"]

FailureKind = Literal[
    "step_budget_exhausted",
    "tool_error_loop",
    "no_tool_use",
    "llm_error",
    "malformed_output",
    "wrong_answer",
    "none",
]


@dataclass
class Step:
    index: int
    kind: StepKind
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    ok: bool = True
    latency_s: float = 0.0


@dataclass
class AgentResult:
    task_id: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    wall_s: float = 0.0
    failure: FailureKind = "none"
    agent: str = ""

    @property
    def n_steps(self) -> int:
        return sum(1 for s in self.steps if s.kind in ("tool_call", "delegate"))

    @property
    def n_tool_errors(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool_result" and not s.ok)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        d["n_steps"] = self.n_steps
        d["n_tool_errors"] = self.n_tool_errors
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")


class Agent(ABC):
    name: str = "agent"

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        max_steps: int = 12,
        context: Any = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.context = context

    @abstractmethod
    def _run(self, task: str, result: AgentResult) -> str:
        """Do the work; append to result.steps; return the final answer."""

    def run(self, task: str, task_id: str | None = None) -> AgentResult:
        result = AgentResult(
            task_id=task_id or uuid.uuid4().hex[:8],
            answer="",
            agent=self.name,
        )
        start = time.perf_counter()
        before = Usage(self.llm.total_usage.input_tokens, self.llm.total_usage.output_tokens)
        try:
            result.answer = self._run(task, result)
        except Exception as exc:  # noqa: BLE001
            result.failure = "llm_error"
            result.answer = ""
            result.steps.append(
                Step(len(result.steps), "error", f"{type(exc).__name__}: {exc}", ok=False)
            )
        result.wall_s = time.perf_counter() - start
        result.usage = Usage(
            self.llm.total_usage.input_tokens - before.input_tokens,
            self.llm.total_usage.output_tokens - before.output_tokens,
        )
        return result

    # -- shared helpers ---------------------------------------------------

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.context.prepare(messages) if self.context else messages

    def _consecutive_tool_errors(self, result: AgentResult, threshold: int = 3) -> bool:
        recent = [s for s in result.steps if s.kind == "tool_result"][-threshold:]
        return len(recent) == threshold and all(not s.ok for s in recent)

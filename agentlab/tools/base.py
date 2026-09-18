"""Tool abstraction with input *and* output validation.

The output side is the part most agent codebases skip. An unvalidated tool
result goes straight into the model's context, so a 400 KB HTML dump or a
stack trace silently becomes the agent's next prompt. `Tool.call` enforces a
size ceiling and normalises failures into a structured `ToolResult` the agent
can reason about instead of a raw exception.
"""

from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 8_000


@dataclass
class ToolResult:
    ok: bool
    content: str
    error_kind: str | None = None  # validation | execution | timeout | oversized
    latency_s: float = 0.0
    truncated: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_model(self) -> str:
        """Rendered form that enters the model's context."""
        if self.ok:
            return self.content + ("\n[truncated]" if self.truncated else "")
        return f"ERROR ({self.error_kind}): {self.content}"


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    #: Seconds before the call is abandoned.
    timeout_s: float = 30.0

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute. Raise on failure; `call` converts that into a ToolResult."""

    # -- plumbing ---------------------------------------------------------

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate_input(self, args: dict[str, Any]) -> str | None:
        """Return an error message, or None if valid."""
        try:
            import jsonschema
        except ImportError:
            return None  # validation is best-effort if jsonschema is absent
        try:
            jsonschema.validate(args, self.input_schema)
        except jsonschema.ValidationError as exc:
            return f"{exc.message} (at {'/'.join(str(p) for p in exc.absolute_path) or 'root'})"
        return None

    def call(self, args: dict[str, Any]) -> ToolResult:
        err = self.validate_input(args)
        if err:
            return ToolResult(False, err, error_kind="validation")

        start = time.perf_counter()
        try:
            raw = self.run(**args)
        except TimeoutError as exc:
            return ToolResult(False, str(exc), error_kind="timeout",
                              latency_s=time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - deliberate: surface to the agent
            logger.exception("tool %s raised", self.name)
            return ToolResult(False, f"{type(exc).__name__}: {exc}",
                              error_kind="execution",
                              latency_s=time.perf_counter() - start)

        latency = time.perf_counter() - start
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

        if len(text) > MAX_RESULT_CHARS:
            return ToolResult(
                True, text[:MAX_RESULT_CHARS], latency_s=latency, truncated=True,
                meta={"original_chars": len(text)},
            )
        return ToolResult(True, text, latency_s=latency)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                False,
                f"unknown tool {name!r}; available: {', '.join(self.names())}",
                error_kind="validation",
            )
        return tool.call(args)

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A registry containing only the named tools — used to scope subagents."""
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"not registered: {missing}")
        return ToolRegistry([self._tools[n] for n in names])

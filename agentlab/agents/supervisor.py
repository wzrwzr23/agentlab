"""Supervisor routing to scoped subagents.

Each subagent gets its own tool subset and its own context window. That
isolation is the actual argument for multi-agent architectures: a specialist
with four tools and a clean context outperforms one agent holding twelve tools
and a transcript of everything. State passes between them explicitly, as text,
which keeps handoffs inspectable in the trace.

Routing is a single LLM call with a constrained output. Keep it that way —
a router that reasons at length is a planner, and you already have one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .base import Agent, AgentResult, Step

ROUTER_SYSTEM = """Route a task to the most suitable specialist.

Specialists:
{roster}

Respond with JSON only, no prose and no code fences:
{{"specialist": "<name>", "reason": "<one sentence>", "subtask": "<task restated for them>"}}"""


@dataclass
class Specialist:
    name: str
    description: str
    agent: Agent


def _parse_route(text: str, valid: list[str]) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"router returned no JSON: {text[:200]}")
    data = json.loads(match.group(0))
    if data.get("specialist") not in valid:
        raise ValueError(f"router chose unknown specialist {data.get('specialist')!r}")
    return data


class SupervisorAgent(Agent):
    name = "supervisor"

    def __init__(self, llm: Any, specialists: list[Specialist], max_steps: int = 3, **kwargs: Any):
        # The supervisor holds no tools of its own; it only delegates.
        from ..tools.base import ToolRegistry
        super().__init__(llm=llm, tools=ToolRegistry([]), max_steps=max_steps, **kwargs)
        if not specialists:
            raise ValueError("supervisor needs at least one specialist")
        self.specialists = {s.name: s for s in specialists}

    def _roster(self) -> str:
        return "\n".join(
            f"- {s.name}: {s.description} (tools: {', '.join(s.agent.tools.names()) or 'none'})"
            for s in self.specialists.values()
        )

    def _run(self, task: str, result: AgentResult) -> str:
        completion = self.llm.complete(
            messages=[{"role": "user", "content": task}],
            system=ROUTER_SYSTEM.format(roster=self._roster()),
            max_tokens=512,
        )
        try:
            route = _parse_route(completion.text, list(self.specialists))
        except ValueError as exc:
            result.failure = "malformed_output"
            result.steps.append(Step(len(result.steps), "error", str(exc), ok=False))
            # Fall back to the first specialist rather than failing outright.
            route = {"specialist": next(iter(self.specialists)),
                     "reason": "router fallback", "subtask": task}

        chosen = self.specialists[route["specialist"]]
        result.steps.append(
            Step(len(result.steps), "delegate",
                 f"{route['specialist']}: {route.get('reason', '')}",
                 tool_name=route["specialist"])
        )

        sub = chosen.agent.run(route.get("subtask") or task, task_id=result.task_id)

        # Flatten the subagent's trace so the harness sees one continuous record.
        offset = len(result.steps)
        for step in sub.steps:
            step.index += offset
            result.steps.append(step)
        if sub.failure != "none":
            result.failure = sub.failure

        return sub.answer

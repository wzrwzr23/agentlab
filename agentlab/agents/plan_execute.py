"""Plan-and-execute agent with bounded replanning.

Contrast with ReAct: the plan is committed up front, so the agent cannot drift
task-to-task, but it also cannot adapt cheaply. Replanning is capped — an
uncapped replan loop is how these agents burn a budget without progressing.

Worth measuring: on the multi-hop tasks this should beat ReAct on step count
and lose on recoverability. Whether that holds on your task set is the point
of the ablation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import Agent, AgentResult, Step

PLANNER_SYSTEM = """You plan research over a corpus of machine learning papers.

Break the task into 2-5 concrete steps. Each step must be answerable with the \
available tools: {tool_names}.

Respond with JSON only, no prose and no code fences:
{{"steps": ["...", "..."]}}"""

EXECUTOR_SYSTEM = """You execute one step of a research plan using tools.

Overall task: {task}
Current step: {step}

Findings so far:
{findings}

Use tools to complete this step, then state what you found in two or three \
sentences. Ground claims in tool output. If the step cannot be completed, say \
why plainly."""

SYNTH_SYSTEM = """Answer the task using only the findings below. Cite paper IDs. \
If the findings do not support an answer, say so."""


def _parse_plan(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in planner output: {text[:200]}")
    steps = json.loads(match.group(0)).get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("planner returned no steps")
    return [str(s) for s in steps][:5]


class PlanExecuteAgent(Agent):
    name = "plan_execute"

    def __init__(self, *args: Any, max_replans: int = 1, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.max_replans = max_replans

    def _plan(self, task: str, result: AgentResult, failed: str | None = None) -> list[str]:
        prompt = task if not failed else (
            f"{task}\n\nA previous plan failed at: {failed}\nProduce a different plan."
        )
        completion = self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=PLANNER_SYSTEM.format(tool_names=", ".join(self.tools.names())),
            max_tokens=1024,
        )
        steps = _parse_plan(completion.text)
        result.steps.append(
            Step(len(result.steps), "plan", json.dumps(steps, ensure_ascii=False))
        )
        return steps

    def _execute(self, task: str, step: str, findings: list[str],
                 result: AgentResult, budget: int) -> tuple[str, bool]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"Execute: {step}"}]
        system = EXECUTOR_SYSTEM.format(
            task=task, step=step,
            findings="\n".join(f"- {f}" for f in findings) or "(none yet)",
        )

        for _ in range(budget):
            completion = self.llm.complete(
                messages=self._messages(messages), system=system, tools=self.tools.specs()
            )
            if not completion.tool_calls:
                return completion.text.strip(), True

            assistant_content: list[dict[str, Any]] = []
            if completion.text.strip():
                assistant_content.append({"type": "text", "text": completion.text})
            for call in completion.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": call["id"],
                     "name": call["name"], "input": call["input"]}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            blocks = []
            for call in completion.tool_calls:
                result.steps.append(
                    Step(len(result.steps), "tool_call", "",
                         tool_name=call["name"], tool_args=call["input"])
                )
                tr = self.tools.call(call["name"], call["input"])
                result.steps.append(
                    Step(len(result.steps), "tool_result", tr.to_model(),
                         tool_name=call["name"], ok=tr.ok, latency_s=tr.latency_s)
                )
                blocks.append({
                    "type": "tool_result", "tool_use_id": call["id"],
                    "content": tr.to_model(), "is_error": not tr.ok,
                })
            messages.append({"role": "user", "content": blocks})

            if self._consecutive_tool_errors(result):
                return "step failed: repeated tool errors", False

        return "step failed: budget exhausted", False

    def _run(self, task: str, result: AgentResult) -> str:
        findings: list[str] = []
        replans = 0
        plan = self._plan(task, result)

        while True:
            per_step = max(2, self.max_steps // max(len(plan), 1))
            failed_at = None
            for step in plan:
                finding, ok = self._execute(task, step, findings, result, per_step)
                findings.append(f"{step} -> {finding}")
                if not ok:
                    failed_at = step
                    break

            if failed_at is None or replans >= self.max_replans:
                break
            replans += 1
            plan = self._plan(task, result, failed=failed_at)

        if failed_at is not None:
            result.failure = "step_budget_exhausted"

        synth = self.llm.complete(
            messages=[{"role": "user", "content":
                       f"Task: {task}\n\nFindings:\n" + "\n".join(f"- {f}" for f in findings)}],
            system=SYNTH_SYSTEM,
            max_tokens=1024,
        )
        result.steps.append(Step(len(result.steps), "answer", synth.text.strip()))
        return synth.text.strip()

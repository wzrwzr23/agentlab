"""ReAct agent: interleaved reasoning and tool use, native tool calling.

Uses the API's tool-calling interface rather than parsing "Action:" strings
out of free text. The prompt-parsing variant of ReAct exists because early
models had no tool API; reimplementing it now just adds a parser to debug.
"""

from __future__ import annotations

from typing import Any

from .base import Agent, AgentResult, Step

SYSTEM = """You are a research agent working over a corpus of machine learning papers.

Work in small steps. Before each tool call, state briefly what you are looking \
for and why. After each result, say what you learned and what remains unknown.

Rules:
- Ground every factual claim in something a tool returned. Never rely on prior knowledge \
about specific papers.
- If a tool errors, read the error and change your approach. Do not repeat the same call.
- If the corpus genuinely does not contain the answer, say so explicitly rather than guessing.
- When you have the answer, state it directly with the paper IDs you used as evidence.

You have at most {max_steps} tool calls."""


class ReActAgent(Agent):
    name = "react"

    def _run(self, task: str, result: AgentResult) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        system = SYSTEM.format(max_steps=self.max_steps)

        for _ in range(self.max_steps):
            completion = self.llm.complete(
                messages=self._messages(messages),
                system=system,
                tools=self.tools.specs(),
            )

            if completion.text.strip():
                result.steps.append(Step(len(result.steps), "think", completion.text.strip()))

            if not completion.tool_calls:
                result.steps.append(
                    Step(len(result.steps), "answer", completion.text.strip())
                )
                return completion.text.strip()

            assistant_content: list[dict[str, Any]] = []
            if completion.text.strip():
                assistant_content.append({"type": "text", "text": completion.text})
            for call in completion.tool_calls:
                assistant_content.append(
                    {"type": "tool_use", "id": call["id"],
                     "name": call["name"], "input": call["input"]}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_blocks = []
            for call in completion.tool_calls:
                result.steps.append(
                    Step(len(result.steps), "tool_call", "",
                         tool_name=call["name"], tool_args=call["input"])
                )
                tool_result = self.tools.call(call["name"], call["input"])
                result.steps.append(
                    Step(len(result.steps), "tool_result", tool_result.to_model(),
                         tool_name=call["name"], ok=tool_result.ok,
                         latency_s=tool_result.latency_s)
                )
                tool_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": tool_result.to_model(),
                    "is_error": not tool_result.ok,
                })
            messages.append({"role": "user", "content": tool_blocks})

            if self._consecutive_tool_errors(result):
                result.failure = "tool_error_loop"
                return ""

        result.failure = "step_budget_exhausted"
        # One forced attempt with what it has, rather than returning nothing.
        final = self.llm.complete(
            messages=self._messages(messages) + [
                {"role": "user", "content":
                 "Step budget reached. Give your best answer from what you have, "
                 "or say the corpus does not support an answer."}
            ],
            system=system,
        )
        result.steps.append(Step(len(result.steps), "answer", final.text.strip()))
        return final.text.strip()

"""Eval harness.

This is the part of the repo worth showing an interviewer. It turns "the agent
seems to work" into numbers you can put in a table and defend:

    success rate         overall and per task type
    steps                mean tool calls on solved tasks
    tool error rate      share of tool results that errored
    citation recall      did it cite the sources it should have
    cost                 USD per task, and per solved task
    failure taxonomy     why the failures failed

Run two configurations and diff them — that diff is the ablation.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from ..agents.base import AgentResult
from .tasks import Task, citation_recall


@dataclass
class TaskOutcome:
    task_id: str
    task_type: str
    correct: bool
    reason: str
    n_steps: int
    n_tool_errors: int
    n_tool_calls: int
    citation_recall: float
    cost_usd: float
    wall_s: float
    failure: str
    answer: str = ""


@dataclass
class EvalReport:
    config: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return sum(o.correct for o in self.outcomes) / max(len(self.outcomes), 1)

    def by_type(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for tt in sorted({o.task_type for o in self.outcomes}):
            subset = [o for o in self.outcomes if o.task_type == tt]
            out[tt] = {
                "n": len(subset),
                "success_rate": sum(o.correct for o in subset) / len(subset),
                "mean_steps": statistics.fmean(o.n_steps for o in subset),
            }
        return out

    def tool_error_rate(self) -> float:
        calls = sum(o.n_tool_calls for o in self.outcomes)
        errors = sum(o.n_tool_errors for o in self.outcomes)
        return errors / calls if calls else 0.0

    def failure_taxonomy(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.outcomes:
            if o.correct:
                continue
            key = o.failure if o.failure != "none" else "wrong_answer"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict[str, Any]:
        solved = [o for o in self.outcomes if o.correct]
        total_cost = sum(o.cost_usd for o in self.outcomes)
        n = max(len(self.outcomes), 1)
        budget_exhausted = sum(
            1 for o in self.outcomes if o.failure == "step_budget_exhausted"
        )
        unanswerable = [o for o in self.outcomes if o.task_type == "unanswerable"]
        return {
            "config": self.config,
            "n_tasks": len(self.outcomes),
            "success_rate": round(self.success_rate, 3),
            "by_type": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                        for k, v in self.by_type().items()},
            "mean_steps_solved": round(
                statistics.fmean([o.n_steps for o in solved]), 2) if solved else None,
            "tool_error_rate": round(self.tool_error_rate(), 3),
            "budget_exhausted_rate": round(budget_exhausted / n, 3),
            "mean_citation_recall": round(statistics.fmean(
                [o.citation_recall for o in self.outcomes
                 if o.citation_recall == o.citation_recall]), 3)
            if any(o.citation_recall == o.citation_recall for o in self.outcomes) else None,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_solved_usd": round(total_cost / len(solved), 4) if solved else None,
            "mean_unanswerable_cost_usd": round(
                statistics.fmean(o.cost_usd for o in unanswerable), 5
            ) if unanswerable else None,
            "mean_wall_s": round(statistics.fmean(o.wall_s for o in self.outcomes), 2),
            "failures": self.failure_taxonomy(),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n=== {s['config']} ===")
        print(f"tasks              {s['n_tasks']}")
        print(f"success rate       {s['success_rate']:.1%}")
        for tt, stats in s["by_type"].items():
            print(f"  {tt:<14} {stats['success_rate']:.1%}  "
                  f"(n={int(stats['n'])}, steps={stats['mean_steps']:.1f})")
        print(f"tool error rate    {s['tool_error_rate']:.1%}")
        print(f"budget exhausted   {s['budget_exhausted_rate']:.1%}")
        print(f"mean steps         {s['mean_steps_solved']}")
        print(f"citation recall    {s['mean_citation_recall']}")
        print(f"cost / solved      ${s['cost_per_solved_usd']}")
        if s.get("mean_unanswerable_cost_usd") is not None:
            print(f"cost / unanswerable ${s['mean_unanswerable_cost_usd']:.5f}")
        print(f"failures           {s['failures']}")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"summary": self.summary(), "outcomes": [asdict(o) for o in self.outcomes]},
            indent=2, ensure_ascii=False,
        ))


def run_eval(
    agent_factory: Callable[[], Any],
    tasks: list[Task],
    config_name: str,
    model: str = "claude-sonnet-4-6",
    max_workers: int = 4,
    trace_path: str | Path | None = None,
) -> EvalReport:
    """Run every task. `agent_factory` is called per task so agents never share state."""
    report = EvalReport(config=config_name)

    def _one(task: Task) -> TaskOutcome:
        agent = agent_factory()
        result: AgentResult = agent.run(task.prompt, task_id=task.id)
        correct, reason = task.grade(result.answer)
        if trace_path:
            result.save(Path(trace_path))
        return TaskOutcome(
            task_id=task.id,
            task_type=task.type,
            correct=correct,
            reason=reason,
            n_steps=result.n_steps,
            n_tool_errors=result.n_tool_errors,
            n_tool_calls=sum(1 for s in result.steps if s.kind == "tool_call"),
            citation_recall=citation_recall(result.answer, task.expected_sources),
            cost_usd=result.usage.cost_usd(model),
            wall_s=result.wall_s,
            failure=result.failure,
            answer=result.answer[:500],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, t): t for t in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                report.outcomes.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                report.outcomes.append(TaskOutcome(
                    task.id, task.type, False, f"harness error: {exc}",
                    0, 0, 0, float("nan"), 0.0, 0.0, "llm_error",
                ))

    report.outcomes.sort(key=lambda o: o.task_id)
    return report

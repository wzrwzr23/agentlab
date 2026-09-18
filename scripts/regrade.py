#!/usr/bin/env python3
"""Re-grade a traces.jsonl file with the current Task.grade implementation.

Usage:
    python scripts/regrade.py results/react__ctx-trim.traces.jsonl
    python scripts/regrade.py results/react__ctx-trim.traces.jsonl --tasks tasks/tasks.yaml

Re-reads every answer from the trace and scores it against the live grader,
so grader changes (normalisation fixes, new refusal patterns) can be re-scored
without paying for another model run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/regrade.py` from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentlab.evaluation.tasks import Task, citation_recall, load_tasks
from agentlab.evaluation.harness import EvalReport, TaskOutcome
from agentlab.llm import Usage


def _count_tool_calls(steps: list[dict]) -> int:
    return sum(1 for s in steps if s.get("kind") == "tool_call")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-grade a traces.jsonl with the current grader."
    )
    parser.add_argument("traces", help="Path to *.traces.jsonl")
    parser.add_argument(
        "--tasks", default="tasks/tasks.yaml", help="Task definitions (default: tasks/tasks.yaml)"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model id used for cost recalculation (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    traces_path = Path(args.traces)
    if not traces_path.exists():
        sys.exit(f"error: file not found: {traces_path}")

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        # try relative to the traces file's grandparent (project root)
        tasks_path = traces_path.parent.parent / args.tasks
    if not tasks_path.exists():
        sys.exit(f"error: tasks file not found: {args.tasks}")

    task_index: dict[str, Task] = {t.id: t for t in load_tasks(tasks_path)}

    # Config name: "react__ctx-trim.traces.jsonl" → "react__ctx-trim"
    config = traces_path.name
    for suffix in (".traces.jsonl", ".jsonl"):
        config = config.removesuffix(suffix)

    report = EvalReport(config=f"{config} [regraded]")
    skipped = 0

    for lineno, raw in enumerate(traces_path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue

        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"warning: line {lineno}: invalid JSON ({exc}), skipping", file=sys.stderr)
            skipped += 1
            continue

        task_id: str = rec.get("task_id", "")
        task = task_index.get(task_id)
        if task is None:
            print(
                f"warning: task {task_id!r} not found in {tasks_path}, skipping",
                file=sys.stderr,
            )
            skipped += 1
            continue

        answer: str = rec.get("answer", "")
        correct, reason = task.grade(answer)

        steps: list[dict] = rec.get("steps", [])
        usage_raw = rec.get("usage", {})
        cost = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
        ).cost_usd(args.model)

        report.outcomes.append(
            TaskOutcome(
                task_id=task_id,
                task_type=task.type,
                correct=correct,
                reason=reason,
                n_steps=rec.get("n_steps", 0),
                n_tool_errors=rec.get("n_tool_errors", 0),
                n_tool_calls=_count_tool_calls(steps),
                citation_recall=citation_recall(answer, task.expected_sources),
                cost_usd=cost,
                wall_s=rec.get("wall_s", 0.0),
                failure=rec.get("failure", "none"),
                answer=answer[:500],
            )
        )

    if skipped:
        print(f"warning: skipped {skipped} line(s)", file=sys.stderr)

    report.outcomes.sort(key=lambda o: o.task_id)
    report.print_summary()


if __name__ == "__main__":
    main()

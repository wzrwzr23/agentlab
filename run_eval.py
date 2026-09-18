#!/usr/bin/env python3
"""Run an evaluation configuration, or an ablation across several.

    python run_eval.py --agent react
    python run_eval.py --agent plan_execute --context summarize
    python run_eval.py --ablation            # runs the preset grid

Every run writes results/<config>.json and results/<config>.traces.jsonl.
The traces are how you debug a failure: open the JSONL, find the task, read
what the agent actually did. Do that before changing any prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from agentlab.llm import LLMClient
from agentlab.tools import default_registry
from agentlab.memory import ContextManager
from agentlab.agents import ReActAgent, PlanExecuteAgent, SupervisorAgent, Specialist
from agentlab.evaluation import load_tasks, run_eval

RESULTS = Path("results")


def build_agent(kind: str, model: str, context_strategy: str, max_steps: int):
    """Factory-of-factories: each task gets a fresh agent with fresh state."""

    def factory():
        llm = LLMClient(model=model)
        tools = default_registry()
        ctx = ContextManager(
            strategy=context_strategy,
            llm=llm if context_strategy == "summarize" else None,
        )

        if kind == "react":
            return ReActAgent(llm, tools, max_steps=max_steps, context=ctx)
        if kind == "plan_execute":
            return PlanExecuteAgent(llm, tools, max_steps=max_steps, context=ctx)
        if kind == "supervisor":
            searcher = ReActAgent(
                llm, tools.subset(["search_papers", "fetch_paper"]),
                max_steps=max_steps, context=ctx,
            )
            analyst = PlanExecuteAgent(
                llm, tools.subset(["search_papers", "fetch_paper", "calculate"]),
                max_steps=max_steps, context=ctx,
            )
            return SupervisorAgent(llm, [
                Specialist("searcher", "Finds and retrieves specific papers. "
                                       "Best for single-fact lookups.", searcher),
                Specialist("analyst", "Compares and synthesises across several papers. "
                                      "Best for multi-hop and comparative questions.", analyst),
            ])
        raise ValueError(f"unknown agent kind: {kind}")

    return factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="react",
                    choices=["react", "plan_execute", "supervisor"])
    ap.add_argument("--context", default="trim", choices=["none", "trim", "summarize"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--tasks", default="tasks/tasks.yaml")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ablation", action="store_true",
                    help="run the preset configuration grid instead of one config")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tasks = load_tasks(args.tasks)
    print(f"loaded {len(tasks)} tasks from {args.tasks}")

    configs = (
        [("react", "none"), ("react", "trim"), ("react", "summarize"),
         ("plan_execute", "trim"), ("supervisor", "trim")]
        if args.ablation
        else [(args.agent, args.context)]
    )

    reports = []
    for kind, ctx in configs:
        name = f"{kind}__ctx-{ctx}"
        print(f"\nrunning {name} ...")
        report = run_eval(
            agent_factory=build_agent(kind, args.model, ctx, args.max_steps),
            tasks=tasks,
            config_name=name,
            model=args.model,
            max_workers=args.workers,
            trace_path=RESULTS / f"{name}.traces.jsonl",
        )
        report.print_summary()
        report.save(RESULTS / f"{name}.json")
        reports.append(report)

    if len(reports) > 1:
        print("\n=== ablation ===")
        print(f"{'config':<28} {'success':>8} {'steps':>7} {'toolerr':>8} {'$/solved':>9}")
        for r in reports:
            s = r.summary()
            print(f"{s['config']:<28} {s['success_rate']:>7.1%} "
                  f"{str(s['mean_steps_solved']):>7} {s['tool_error_rate']:>7.1%} "
                  f"{str(s['cost_per_solved_usd']):>9}")
        (RESULTS / "ablation.json").write_text(
            json.dumps([r.summary() for r in reports], indent=2)
        )


if __name__ == "__main__":
    main()

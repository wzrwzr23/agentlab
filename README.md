# agentlab

A multi-agent research system over a corpus of ML papers, built to compare agent
architectures against each other with real numbers rather than demos.

The interesting part is not the agents — ReAct and plan-execute are well known.
It is the evaluation harness: a task set with deterministic grading, a failure
taxonomy, and an ablation runner, so claims like "summarisation beats trimming"
can be checked instead of asserted.

## Status

Working: tool layer, context management, all three agent architectures, eval
harness, 24 passing unit tests, end-to-end run against a stubbed model.

Not done yet: no results have been produced against a real model. The tables
below are empty on purpose — see [Running it](#running-it).

## Architecture

```
                    ┌─────────────┐
       task ───────▶│ Supervisor  │  routes once, holds no tools
                    └──────┬──────┘
                 ┌─────────┴─────────┐
                 ▼                   ▼
          ┌────────────┐      ┌──────────────┐
          │  searcher  │      │   analyst    │   scoped tool subsets,
          │  (ReAct)   │      │(plan-execute)│   separate contexts
          └─────┬──────┘      └──────┬───────┘
                └──────────┬─────────┘
                           ▼
                  ┌─────────────────┐
                  │  ToolRegistry   │  schema validation in,
                  └────────┬────────┘  size + error handling out
                           ▼
              search_papers · fetch_paper · calculate
```

Three design choices worth arguing about:

**Tool output is validated, not just tool input.** An unvalidated result goes
straight into the model's context, so a 400 KB dump or a stack trace silently
becomes the next prompt. `Tool.call` enforces a size ceiling and converts
exceptions into structured results the agent can reason about.

**Subagents get scoped tool subsets and separate contexts.** That isolation is
the real argument for multi-agent designs — a specialist with two tools and a
clean window beats one agent holding twelve tools and a full transcript.

**Context strategy is a swappable parameter, not a hardcoded policy.** `none`,
`trim`, and `summarize` exist so the comparison can be measured.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

python -m pytest tests/ -q          # offline, no key needed
python run_eval.py --agent react    # one configuration
python run_eval.py --ablation       # the full grid
```

Results land in `results/`. Each run also writes `*.traces.jsonl` — the
step-by-step record of what the agent did. Read the trace before changing a
prompt; most apparent reasoning failures turn out to be tool or context
failures.

## Results

Run `--ablation` and fill this in.

| Config | Success | Lookup | Multi-hop | Comparative | Unanswerable | Steps | Tool err | $/solved |
|---|---|---|---|---|---|---|---|---|
| react + none | | | | | | | | |
| react + trim | | | | | | | | |
| react + summarize | | | | | | | | |
| plan_execute + trim | | | | | | | | |
| supervisor + trim | | | | | | | | |

Questions the numbers should answer:

1. Does plan-execute use fewer steps than ReAct on multi-hop tasks, and does it
   lose on recoverability when a step fails?
2. Does summarisation beat trimming enough to justify its extra token cost?
3. Does the supervisor's routing overhead pay for itself, or does routing
   error exceed the specialisation gain?
4. Which configuration hallucinates least on the unanswerable set?

## Evaluation design

16 seed tasks across four types. The `unanswerable` type is the one usually
omitted and the one that matters most: an agent scoring well on the other three
while answering unanswerable questions confidently is not good, just assertive.

Grading is deterministic substring and citation matching. An LLM judge has its
own error rate, and using one means reporting two systems' errors compounded.

**The seed set is too small to report from.** Below about 40 tasks the
confidence interval on success rate is wide enough to swamp ablation
differences. Grow it before publishing numbers.

## Known limitations

- Retrieval is BM25-lite over 14 papers. Swapping in a real retriever is itself
  a worthwhile ablation.
- The supervisor routes once and does not re-route on subagent failure.
- No MCP server yet.
- Costs assume the pricing table in `llm.py`, which goes stale.

## Layout

```
agentlab/
  llm.py                 client: retries, timeouts, token accounting
  tools/base.py          Tool ABC, input+output validation, registry
  tools/builtin.py       corpus search, fetch, calculator
  memory/context.py      none | trim | summarize
  agents/base.py         step budget, structured trace, failure kinds
  agents/react.py        interleaved reasoning and acting
  agents/plan_execute.py plan up front, bounded replanning
  agents/supervisor.py   routing to scoped specialists
  evaluation/tasks.py    task schema and deterministic graders
  evaluation/harness.py  runner, metrics, failure taxonomy
tasks/tasks.yaml         the evaluation set
tests/test_core.py       24 offline tests
run_eval.py              CLI
```

## What I'd do differently

Written after the first real ablation run — leave this section until then, then
be specific. This is the section interviewers read most closely, because it
shows judgment rather than tutorial completion.

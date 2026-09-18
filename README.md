# agentlab

A multi-agent research system over a corpus of ML papers, built to compare agent
architectures against each other with real numbers rather than demos.

The interesting part is not the agents — ReAct and plan-execute are well known.
It is the evaluation harness: a task set with deterministic grading, a failure
taxonomy, and an ablation runner, so claims like "summarisation beats trimming"
can be checked instead of asserted.

## Status

Working: tool layer, context management, all three agent architectures, eval
harness, 42 passing unit tests, end-to-end ablation against `claude-haiku-4-5`
over 62 tasks. Results are in the table below.

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
    search_papers · fetch_paper · list_papers · calculate
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
cp .env.example .env    # then put your key in it

python -m pytest tests/ -q          # offline, no key needed
python run_eval.py --agent react --model claude-haiku-4-5-20251001   # one configuration
python run_eval.py --ablation --model claude-haiku-4-5-20251001     # the full grid
```

Results land in `results/`. Each run also writes `*.traces.jsonl` — the
step-by-step record of what the agent did. Read the trace before changing a
prompt; most apparent reasoning failures turn out to be tool or context
failures.

The `--model` flag matters. Runs here use Haiku deliberately: on a stronger
model this task set saturates and every configuration scores near the ceiling,
leaving nothing to compare. A weaker model keeps the set discriminative.

## Results

Single-run ablation, 62 tasks, `claude-haiku-4-5`, n=1 for three configs and
n=3 for the two headline configs. Per-type scores shown for one representative config; the per-run variance section below is the more honest read.

| Config | Success | Lookup | Multi-hop | Comparative | Unanswerable | Steps | Budget&nbsp;exhausted | $/solved |
|---|---|---|---|---|---|---|---|---|
| react + none | 91.9% | — | — | — | — | 4.42 | 0.0% | $0.0112 |
| react + trim (run 1) | 93.5% | 87.5% | 100% | 85.7% | 100% | 4.57 | 0.0% | $0.0105 |
| react + summarize | 90.3% | — | — | — | — | 4.64 | 0.0% | $0.0119 |
| plan_execute + trim | 90.3% | — | — | — | — | 6.64 | 14.5% | $0.0261 |
| supervisor + trim | 87.1% | — | — | — | — | 7.41 | 14.5% | $0.0257 |

**Run-to-run variance (n=3), headline configs:**

| Config | Run 1 | Run 2 | Run 3 | Mean | Range |
|---|---|---|---|---|---|
| react + trim success | 93.5% | 88.7% | 87.1% | 89.8% | 6.4 pp |
| plan_execute + trim success | 90.3% | 85.5% | 85.5% | 87.1% | 4.8 pp |
| react + trim $/solved | $0.0105 | $0.0118 | $0.0120 | $0.0114 | — |
| plan_execute + trim $/solved | $0.0261 | $0.0274 | $0.0276 | $0.0270 | — |

### What the numbers say

**Accuracy differences are not distinguishable; cost differences are.**
The gap between the best and worst single-run success rates is 2.7 pp
(87.1%–93.5% between react+none and supervisor+trim). The within-config
run-to-run variance for the two repeated configs is 4.8–6.4 pp — larger than
any between-config difference. At n=1 or n=3 on 62 tasks, no architecture is
reliably more accurate than another. Do not draw accuracy rankings from this
table. Cost is a different story: plan_execute costs 2.4× more per solved task
($0.027 vs $0.011), and the cost variance across runs is about 5%, making the
2.4× difference reliable.

**Budget exhaustion is the clearest separator.**
ReAct exhausts its step budget on 0.0% of tasks across all three runs.
Plan-execute and supervisor run at 9.7–14.5% across their runs. Planning agents
commit to a plan before searching; when one step misfires the plan keeps
running until the budget is gone rather than pivoting immediately as ReAct
would. The budget-exhaustion rate is a direct measure of this cost.

**All five configs score 100% on unanswerable tasks, but planning
architectures pay roughly twice as much to get there.**
Mean cost on unanswerable tasks is approximately $0.018 for the planning
configs versus $0.010 for ReAct. A plan-execute agent generates a multi-step
search plan and works through it even when the answer does not exist in the
corpus; a ReAct agent typically checks one or two sources, confirms they lack
the answer, and stops.

**Context strategy (none / trim / summarize) has no measurable effect.**
ReAct averages 4.4–4.6 steps across all three strategies. Tasks in this set
rarely accumulate enough context to trigger compaction. The three numbers are
within the run-to-run variance band and should not be interpreted as
evidence that any strategy is better. A longer task set or tasks that require
iterating over all 14 papers would be needed to stress context management.

## Evaluation design

62 tasks across four types (16 lookup, 19 multi-hop, 14 comparative, 13
unanswerable). The `unanswerable` type is the one usually omitted and the one
that matters most: an agent scoring well on the other three while answering
unanswerable questions confidently is not good, just assertive.

Grading is deterministic substring and citation matching. An LLM judge has its
own error rate, and using one means reporting two systems' errors compounded.
See Known limitations for what deterministic grading misses.

## Known limitations

- **Sample size.** Three of five configs were run once. At n=1 on 62 tasks, the
  95% confidence interval on a 90% success rate is roughly ±7 pp, which is
  wider than every gap in the table. The n=3 runs narrow this but do not
  eliminate it.
- **Deterministic grading misses hedged hallucination.** A grader based on
  substring matching passes an answer that says "the corpus doesn't specify,
  but the value is likely 8" because it sees "does not specify". Hallucination
  that is correctly hedged looks correct to this grader.
- **Refusal regex can misfire.** The bidirectional pattern that detects correct
  refusals on unanswerable tasks will match "the corpus contains X, not Y" as
  a refusal, because it sees a corpus word near a negation. The 100% refusal
  rate may be slightly overstated.
- **Retrieval is BM25-lite over 14 papers.** Results will not transfer to a
  real retriever over a larger corpus.
- **Single model.** All runs used `claude-haiku-4-5`. Architecture differences
  may look different with a stronger or weaker model, and cost comparisons are
  model-specific.
- **Costs assume the pricing table in `llm.py`,** which goes stale.

## Layout

```
agentlab/
  llm.py                 client: retries, timeouts, token accounting
  tools/base.py          Tool ABC, input+output validation, registry
  tools/builtin.py       corpus search, fetch, list, calculator
  memory/context.py      none | trim | summarize
  agents/base.py         step budget, structured trace, failure kinds
  agents/react.py        interleaved reasoning and acting
  agents/plan_execute.py plan up front, bounded replanning
  agents/supervisor.py   routing to scoped specialists
  evaluation/tasks.py    task schema and deterministic graders
  evaluation/harness.py  runner, metrics, failure taxonomy
tasks/tasks.yaml         the evaluation set
tests/test_core.py       42 offline tests
run_eval.py              CLI
```

## What I'd do differently

**Audit the grader against real outputs before trusting any number.** The first
real run scored 69.6% overall with unanswerable at 30.8% — 9 of 13 marked
wrong. Reading those 9 traces showed the agent had declined correctly every
single time, using phrasings like "does not appear to contain", "cannot find",
and "is not included in the corpus". The grader used a fixed `REFUSAL_MARKERS`
phrase list that simply didn't cover those forms. Replacing the list with a
bidirectional negation-near-corpus-word regex took unanswerable to 13/13 and
overall to something meaningful. The lesson: read the failing outputs before
changing anything else. A number that looks wrong almost always is, and the
cause is usually in the grader.

**Pass retrieval context between plan steps.** The first plan-execute
implementation gave each executor an independent context with no memory of what
prior steps had found. The executor's system prompt said nothing about which
paper IDs were valid. On multi-hop tasks the model started pulling IDs from
pretraining memory — it called `fetch_paper` with `1706.03762` (the Transformer
arXiv ID), `2104.14881`, and the bare string `"CLIP"` — producing a stream of
tool errors. The fix — tracking IDs seen in tool results across steps and
injecting them into the executor prompt — was straightforward but should have
been obvious from the design: a plan-execute agent is only useful if steps
share what they find.

**Check discriminative power before running the full grid.** The first 46-task
set scored 69.6% for ReAct on the first run. After the grader and tool fixes,
ReAct reached 100% on the same 46 tasks. A benchmark where one configuration
maxes out gives no signal about any other configuration. The right order is:
build the harness, fix the grader, run one agent, verify it fails on at least
15–20% of tasks, adjust difficulty, then run the grid. Instead 16 harder tasks
had to be added after the fact to bring the set to 62 and ReAct back to ~90%,
which meant re-running everything.

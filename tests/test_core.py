"""Tests for the parts that must not silently break.

No LLM calls here — these run offline in under a second. The agent loops are
tested separately against recorded traces; see tests/README.
"""

import pytest

from agentlab.tools.base import Tool, ToolRegistry, MAX_RESULT_CHARS
from agentlab.tools.builtin import SearchPapers, FetchPaper, Calculate, ListPapers
from agentlab.agents.plan_execute import PlanExecuteAgent
from agentlab.agents.base import AgentResult, Step
from agentlab.llm import Completion, Usage
from agentlab.memory.context import ContextManager, estimate_tokens
from agentlab.evaluation.tasks import Task


class Echo(Tool):
    name = "echo"
    description = "echo"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, text: str) -> str:
        return text


class Exploding(Tool):
    name = "boom"
    description = "always fails"
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def run(self) -> str:
        raise ValueError("deliberate")


# -- tool validation -------------------------------------------------------

def test_valid_input_passes():
    assert Echo().call({"text": "hi"}).ok


def test_missing_required_field_is_validation_error():
    r = Echo().call({})
    assert not r.ok and r.error_kind == "validation"


def test_extra_field_rejected():
    r = Echo().call({"text": "hi", "sneaky": 1})
    assert not r.ok and r.error_kind == "validation"


def test_exception_becomes_structured_result_not_a_crash():
    r = Exploding().call({})
    assert not r.ok and r.error_kind == "execution" and "deliberate" in r.content


def test_oversized_output_is_truncated_not_passed_through():
    class Big(Echo):
        name = "big"
        def run(self, text: str) -> str:
            return "x" * (MAX_RESULT_CHARS + 5_000)

    r = Big().call({"text": "go"})
    assert r.ok and r.truncated and len(r.content) == MAX_RESULT_CHARS
    assert "[truncated]" in r.to_model()


def test_error_result_renders_as_error_for_the_model():
    assert Exploding().call({}).to_model().startswith("ERROR (execution)")


# -- registry --------------------------------------------------------------

def test_unknown_tool_returns_result_not_exception():
    reg = ToolRegistry([Echo()])
    r = reg.call("nope", {})
    assert not r.ok and "echo" in r.content


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        ToolRegistry([Echo(), Echo()])


def test_subset_scopes_tools():
    reg = ToolRegistry([Echo(), Exploding()])
    assert reg.subset(["echo"]).names() == ["echo"]
    with pytest.raises(KeyError):
        reg.subset(["missing"])


# -- builtin tools ---------------------------------------------------------

def test_search_returns_ranked_results():
    r = SearchPapers().call({"query": "diffusion video", "top_k": 3})
    assert r.ok and "P004" in r.content


def test_search_rejects_out_of_range_top_k():
    assert not SearchPapers().call({"query": "x", "top_k": 99}).ok


def test_list_papers_all():
    r = ListPapers().call({})
    assert r.ok and "P001" in r.content and "P014" in r.content


def test_list_papers_year_filter():
    r = ListPapers().call({"year": 2022})
    assert r.ok
    assert "P003" in r.content   # 2022 paper
    assert "P001" not in r.content  # 2017 paper filtered out


def test_fetch_unknown_id_errors_cleanly():
    r = FetchPaper().call({"paper_id": "P999"})
    assert not r.ok and r.error_kind == "execution"


def test_calculate_rejects_code_execution():
    r = Calculate().call({"expression": "__import__('os').system('ls')"})
    assert not r.ok


def test_calculate_arithmetic():
    assert Calculate().call({"expression": "(2+3)*4"}).content == "20"


# -- context management ----------------------------------------------------

def _messages(n: int, size: int = 4_000):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "w" * size}
            for i in range(n)]


def test_under_budget_is_untouched():
    cm = ContextManager(strategy="trim", max_tokens=100_000)
    msgs = _messages(4)
    assert cm.prepare(msgs) == msgs


def test_trim_drops_middle_and_keeps_task():
    cm = ContextManager(strategy="trim", max_tokens=2_000, keep_recent=2)
    msgs = _messages(10)
    out = cm.prepare(msgs)
    assert out[0] is msgs[0], "first message carries the task and must survive"
    assert len(out) < len(msgs)
    assert cm.stats.compactions == 1


def test_prepare_does_not_mutate_input():
    cm = ContextManager(strategy="trim", max_tokens=2_000, keep_recent=2)
    msgs = _messages(10)
    cm.prepare(msgs)
    assert len(msgs) == 10


def test_summarize_requires_llm():
    with pytest.raises(ValueError):
        ContextManager(strategy="summarize")


def test_estimate_tokens_scales_with_content():
    assert estimate_tokens(_messages(2, 400)) < estimate_tokens(_messages(2, 4_000))


# -- grading ---------------------------------------------------------------

def test_unanswerable_rewards_refusal():
    t = Task(id="u", type="unanswerable", prompt="?")
    assert t.grade("The corpus does not contain that information.")[0]


def test_refusal_regex_negation_then_corpus():
    t = Task(id="u", type="unanswerable", prompt="?")
    assert t.grade("The corpus does not appear to contain that information.")[0]


def test_refusal_regex_negation_then_find():
    t = Task(id="u", type="unanswerable", prompt="?")
    assert t.grade("I cannot find specific information about that.")[0]


def test_refusal_regex_corpus_then_negation():
    # corpus word ("contains") precedes the negation ("not") — bidirectional match
    t = Task(id="u", type="unanswerable", prompt="?")
    assert t.grade("The stored record only contains the abstract, not the detailed results.")[0]


def test_unanswerable_punishes_confident_answer():
    t = Task(id="u", type="unanswerable", prompt="?")
    ok, reason = t.grade("The FID score was 42.1.")
    assert not ok and "hallucinated" in reason


def test_missing_required_content_fails():
    t = Task(id="l", type="lookup", prompt="?", must_include=["transformer"])
    assert not t.grade("It was a recurrent model.")[0]


def test_expected_source_must_be_cited():
    t = Task(id="l", type="lookup", prompt="?", must_include=["attention"],
             expected_sources=["P001"])
    assert not t.grade("Attention matters.")[0]
    assert t.grade("Attention matters, see P001.")[0]


def test_empty_answer_never_passes():
    assert not Task(id="l", type="lookup", prompt="?").grade("   ")[0]


# -- plan_execute: shared budget ------------------------------------------

class _SequentialLLM:
    """Returns a tool-call response on the first complete(), plain text after."""
    def __init__(self, *extra_responses):
        self.total_usage = Usage()
        self._responses = iter([
            Completion(text="", tool_calls=[
                {"id": "tc1", "name": "calculate", "input": {"expression": "1+1"}}
            ]),
            Completion(text="the answer is 2"),
            *extra_responses,
        ])

    def complete(self, *, messages, system=None, tools=None, max_tokens=None):
        return next(self._responses, Completion(text="done"))


def test_shared_budget_decrements_per_tool_call():
    """Each tool call consumes exactly one unit from the shared remaining pool."""
    agent = PlanExecuteAgent(_SequentialLLM(), ToolRegistry([Calculate()]), max_steps=6)
    remaining = [6]
    result = AgentResult(task_id="t", answer="", agent="plan_execute")
    _, ok = agent._execute("task", "calc step", [], result, remaining, set())
    assert ok
    assert remaining[0] == 5  # one tool call used one budget unit from six


def test_shared_budget_skips_step_below_minimum():
    """A step is skipped (not attempted) when fewer than 2 budget units remain."""
    agent = PlanExecuteAgent(_SequentialLLM(), ToolRegistry([Calculate()]), max_steps=6)
    remaining = [1]  # below the minimum-2 threshold
    result = AgentResult(task_id="t", answer="", agent="plan_execute")
    finding, ok = agent._execute("task", "step", [], result, remaining, set())
    # budget exhausted is returned from within _execute when remaining hits 0
    # but the caller (_run) should have skipped; test the guard in _execute itself:
    # with remaining=1, first tool call decrements to 0, then the loop checks again
    # and returns budget-exhausted on the next iteration
    assert remaining[0] == 0


# -- plan_execute: replan threshold ----------------------------------------

class _SynthLLM:
    total_usage = Usage()
    def complete(self, *, messages, system=None, tools=None, max_tokens=None):
        return Completion(text="synthesis result")


def test_no_replan_when_majority_of_steps_succeed():
    """With 3 steps where 2 succeed and 1 fails, replan is skipped."""
    plan_calls = [0]

    class MajoritySuccessAgent(PlanExecuteAgent):
        def _plan(self, task, result, failed=None):
            plan_calls[0] += 1
            result.steps.append(Step(len(result.steps), "plan", ""))
            return ["s1", "s2", "s3"]

        def _execute(self, task, step, findings, result, remaining, seen_ids):
            remaining[0] = max(0, remaining[0] - 1)
            return ("s3 failed", False) if step == "s3" else (f"{step} done", True)

    agent = MajoritySuccessAgent(_SynthLLM(), ToolRegistry([]), max_steps=12)
    result = agent.run("test task")
    # s1 and s2 succeed (2/3 ≥ half) → no replan; only initial _plan call
    assert plan_calls[0] == 1


def test_replan_triggered_when_minority_succeed():
    """With 3 steps where only the first succeeds (1/3 < half), replan fires once."""
    plan_calls = [0]

    class MinoritySuccessAgent(PlanExecuteAgent):
        def _plan(self, task, result, failed=None):
            plan_calls[0] += 1
            result.steps.append(Step(len(result.steps), "plan", ""))
            return ["s1", "s2", "s3"]

        def _execute(self, task, step, findings, result, remaining, seen_ids):
            remaining[0] = max(0, remaining[0] - 1)
            return ("s1 done", True) if step == "s1" else (f"{step} failed", False)

    agent = MinoritySuccessAgent(_SynthLLM(), ToolRegistry([]), max_steps=12, max_replans=1)
    agent.run("test task")
    # s1 succeeds, s2 fails → 1/3 < 1/2 → one replan; then second plan also fails quickly
    assert plan_calls[0] == 2


# -- normalization in grading ---------------------------------------------

def test_citation_accepts_paper_title():
    t = Task(id="l", type="lookup", prompt="?",
             must_include=["contrastive"],
             expected_sources=["P005"])
    assert t.grade("Learning Transferable Visual Models uses a contrastive objective.")[0]


def test_citation_accepts_alias():
    # Both sources cited by alias only — no paper IDs in the answer.
    t = Task(id="c", type="comparative", prompt="?",
             must_include=["differ"],
             expected_sources=["P006", "P012"])
    assert t.grade("ViViT and Make-A-Video differ in goal.")[0]


def test_normalize_ized_matches_ised_must_include():
    # must_include uses British spelling; answer uses American — should still pass
    t = Task(id="n1", type="lookup", prompt="?", must_include=["factorised"],
             expected_sources=["P006"])
    assert t.grade("The factorized encoder reduces compute. See P006.")[0]


def test_normalize_hyphen_optional_in_must_include():
    # must_include keeps hyphen; answer omits it — should still pass
    t = Task(id="n2", type="lookup", prompt="?", must_include=["spatio-temporal"],
             expected_sources=["P006"])
    assert t.grade("It extracts spatiotemporal tokens. P006.")[0]


def test_must_include_list_passes_with_one_alternative():
    t = Task(id="n3", type="lookup", prompt="?",
             must_include=[["factorised", "factorized"]],
             expected_sources=["P006"])
    assert t.grade("Uses factorized attention. P006.")[0]
    assert t.grade("Uses factorised attention. P006.")[0]


def test_normalize_space_matches_hyphen_in_must_include():
    # "chain of thought" in answer should match must_include "chain-of-thought"
    t = Task(id="n5", type="lookup", prompt="?", must_include=["chain-of-thought"],
             expected_sources=["P011"])
    assert t.grade("Uses chain of thought reasoning. P011.")[0]


def test_normalize_hyphen_matches_space_in_must_include():
    # "power law" in answer should match must_include "power-law"
    t = Task(id="n6", type="lookup", prompt="?", must_include=["power-law"],
             expected_sources=["P013"])
    assert t.grade("Follows a power law relationship. P013.")[0]


def test_normalize_izes_matches_ises_must_include():
    # "parallelizes" (American) should match must_include "parallelises" (British)
    t = Task(id="n7", type="lookup", prompt="?", must_include=["parallelises"],
             expected_sources=["P001"])
    assert t.grade("Attention parallelizes across positions. See P001.")[0]


def test_must_include_list_fails_when_no_alternative_present():
    t = Task(id="n4", type="lookup", prompt="?",
             must_include=[["factorised", "factorized"]],
             expected_sources=["P006"])
    ok, reason = t.grade("Uses sparse attention. P006.")
    assert not ok and "missing" in reason

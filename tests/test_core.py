"""Tests for the parts that must not silently break.

No LLM calls here — these run offline in under a second. The agent loops are
tested separately against recorded traces; see tests/README.
"""

import pytest

from agentlab.tools.base import Tool, ToolRegistry, MAX_RESULT_CHARS
from agentlab.tools.builtin import SearchPapers, FetchPaper, Calculate
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


# -- normalization in grading ---------------------------------------------

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


def test_must_include_list_fails_when_no_alternative_present():
    t = Task(id="n4", type="lookup", prompt="?",
             must_include=[["factorised", "factorized"]],
             expected_sources=["P006"])
    ok, reason = t.grade("Uses sparse attention. P006.")
    assert not ok and "missing" in reason

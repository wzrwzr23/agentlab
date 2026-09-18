"""Task definitions and graders.

Four task types, on purpose:

    lookup       single hop, one paper
    multi_hop    requires combining two or more papers
    comparative  requires contrasting approaches
    unanswerable the corpus cannot answer it

The unanswerable set is the one people leave out, and it is the one that
catches hallucination. An agent scoring 90% on the first three and 0% on
unanswerable is not a good agent; it is a confident one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

TaskType = Literal["lookup", "multi_hop", "comparative", "unanswerable"]

# A must_include entry is either a plain string or a list of alternatives;
# a list means any one alternative satisfies the requirement.
MustIncludeEntry = Union[str, list[str]]

REFUSAL_MARKERS = (
    "does not contain", "not in the corpus", "no paper", "cannot answer",
    "does not support", "not available", "no information", "unable to find",
    "not present",
)


def _normalize(text: str) -> str:
    """Lowercase, harmonise British/American spellings, strip hyphens, collapse whitespace.

    Ensures 'factorised'/'factorized' and 'spatio-temporal'/'spatiotemporal'
    compare equal regardless of which spelling appears in the answer.
    """
    t = text.lower()
    t = t.replace("isation", "ization").replace("ised", "ized")
    t = re.sub(r"[\s-]+", "", t)
    return t


@dataclass
class Task:
    id: str
    type: TaskType
    prompt: str
    #: Each entry is a string or a list of alternatives (any one satisfies it).
    must_include: list[MustIncludeEntry] = field(default_factory=list)
    #: Substrings that must not appear — used to catch known wrong answers.
    must_exclude: list[str] = field(default_factory=list)
    #: Paper IDs a correct answer should cite.
    expected_sources: list[str] = field(default_factory=list)
    notes: str = ""

    def grade(self, answer: str) -> tuple[bool, str]:
        """Return (correct, reason). Deterministic — no LLM judge."""
        text = answer.strip()
        if not text:
            return False, "empty answer"

        if self.type == "unanswerable":
            lower = text.lower()
            if any(m in lower for m in REFUSAL_MARKERS):
                return True, "correctly declined"
            return False, "hallucinated an answer to an unanswerable question"

        norm = _normalize(text)

        missing = []
        for entry in self.must_include:
            if isinstance(entry, list):
                if not any(_normalize(alt) in norm for alt in entry):
                    missing.append(entry)
            else:
                if _normalize(entry) not in norm:
                    missing.append(entry)
        if missing:
            return False, f"missing required content: {missing}"

        present = [s for s in self.must_exclude if _normalize(s) in norm]
        if present:
            return False, f"contains excluded content: {present}"

        if self.expected_sources:
            cited = [pid for pid in self.expected_sources if _normalize(pid) in norm]
            if not cited:
                return False, f"cited none of the expected sources {self.expected_sources}"

        return True, "ok"


def load_tasks(path: str | Path) -> list[Task]:
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    tasks = [Task(**item) for item in raw["tasks"]]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate task ids: {dupes}")
    return tasks


def citation_recall(answer: str, expected: list[str]) -> float:
    if not expected:
        return float("nan")
    found = sum(1 for pid in expected if re.search(re.escape(pid), answer, re.I))
    return found / len(expected)

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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

TaskType = Literal["lookup", "multi_hop", "comparative", "unanswerable"]

# Corpus metadata loaded once so grade() can match on id, title, and aliases.
_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.json"
_corpus_raw: list[dict] = json.loads(_CORPUS_PATH.read_text()) if _CORPUS_PATH.exists() else []
_CORPUS_TITLES: dict[str, str] = {p["id"]: p["title"] for p in _corpus_raw}
_CORPUS_ALIASES: dict[str, list[str]] = {p["id"]: p.get("aliases", []) for p in _corpus_raw}
del _corpus_raw


_TITLE_SPLIT_PREPS = frozenset(
    {"from", "with", "for", "via", "in", "on", "of", "at", "by", "to", "without", "using"}
)


def _citation_keys(pid: str) -> list[str]:
    """All normalized strings that count as a citation of paper `pid`.

    Checks: id, full title, title before the first colon ('LoRA: …' → 'LoRA'),
    title before the first major preposition after ≥3 words ('Learning
    Transferable Visual Models From …' → 'Learning Transferable Visual Models'),
    and every alias.
    """
    keys = [_normalize(pid)]
    title = _CORPUS_TITLES.get(pid, "")
    if title:
        keys.append(_normalize(title))
        # Short name for colon-titled papers: "LoRA: Low-Rank …" → "LoRA"
        short = title.split(":")[0].strip()
        if short != title:
            keys.append(_normalize(short))
        # Main clause before first major preposition (≥3 words in):
        # "Learning Transferable Visual Models From …" → "Learning Transferable Visual Models"
        words = title.split()
        for i, w in enumerate(words):
            if i >= 3 and w.lower() in _TITLE_SPLIT_PREPS:
                prefix = _normalize(" ".join(words[:i]))
                if prefix not in keys:
                    keys.append(prefix)
                break
    for alias in _CORPUS_ALIASES.get(pid, []):
        keys.append(_normalize(alias))
    return keys

# A must_include entry is either a plain string or a list of alternatives;
# a list means any one alternative satisfies the requirement.
MustIncludeEntry = Union[str, list[str]]

REFUSAL_MARKERS = (
    "does not contain", "not in the corpus", "no paper", "cannot answer",
    "does not support", "not available", "no information", "unable to find",
    "not present",
)

# Negation word within 60 chars of a corpus/availability word, in either
# direction.  Either order is accepted so phrases like "contains … not" (corpus
# word first, negation second) are caught alongside the more common "cannot
# find" order.
_NEG = r"(?:\bnot\b|\bno\b|\bcannot\b|can't|doesn't|does\s+not|don't|isn't|\bunable\b)"
_CRP = r"(?:\bcontain|\bfind\b|\bfound\b|\binclude|\bpresent\b|\bavailable\b|\bprovide|\bspecify|\bmention|\bappear|\brecord\b|\bsupport)"
_REFUSAL_RE = re.compile(
    rf"(?:{_NEG}.{{0,60}}{_CRP}|{_CRP}.{{0,60}}{_NEG})",
    re.IGNORECASE | re.DOTALL,
)


def _normalize(text: str) -> str:
    """Lowercase, harmonise British/American spellings, strip hyphens, collapse whitespace.

    Ensures 'factorised'/'factorized' and 'spatio-temporal'/'spatiotemporal'
    compare equal regardless of which spelling appears in the answer.
    """
    t = text.lower()
    t = t.replace("isation", "ization").replace("ised", "ized").replace("ises", "izes")
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
            if _REFUSAL_RE.search(lower) or any(m in lower for m in REFUSAL_MARKERS):
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
            cited = [
                pid for pid in self.expected_sources
                if any(k in norm for k in _citation_keys(pid))
            ]
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
    """Fraction of expected paper IDs explicitly cited by ID in the answer.

    Intentionally strict: only bare IDs (e.g. 'P005') count, not titles or
    aliases.  grade() measures correctness; citation_recall measures the
    separate discipline of ID-citing so the two metrics can move independently.
    """
    if not expected:
        return float("nan")
    found = sum(1 for pid in expected if re.search(re.escape(pid), answer, re.I))
    return found / len(expected)

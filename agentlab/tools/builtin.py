"""Tools over a local paper corpus.

Deliberately local: the eval harness must be deterministic and runnable
without network access, otherwise your success-rate numbers move because
arXiv changed, not because your agent did. Swap `search_papers` for a real
retriever once the harness is stable — that swap is itself a good ablation.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .base import Tool

CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.json"


def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        return []
    return json.loads(CORPUS_PATH.read_text())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SearchPapers(Tool):
    name = "search_papers"
    description = (
        "Search the local paper corpus by keyword. Returns ranked paper IDs with "
        "titles and a short snippet. Use this to find papers before fetching them."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.corpus = _load_corpus()
        self._df = Counter()
        for paper in self.corpus:
            aliases_text = " ".join(paper.get("aliases", []))
            for term in set(_tokenize(
                paper["title"] + " " + paper["abstract"] + " "
                + str(paper["year"]) + " " + aliases_text
            )):
                self._df[term] += 1

    def run(self, query: str, top_k: int = 5) -> str:
        if not self.corpus:
            raise RuntimeError("corpus is empty; run scripts/build_corpus.py")

        q_terms = _tokenize(query)
        n = len(self.corpus)
        scored = []
        for paper in self.corpus:
            aliases_text = " ".join(paper.get("aliases", []))
            terms = _tokenize(
                paper["title"] + " " + paper["abstract"] + " "
                + str(paper["year"]) + " " + aliases_text
            )
            tf = Counter(terms)
            # BM25-lite: idf-weighted term frequency, length-normalised.
            score = sum(
                tf[t] / (tf[t] + 1.5) * math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                for t in q_terms if t in tf
            )
            # Title and alias matches carry more signal than abstract matches.
            title_alias_tokens = set(_tokenize(paper["title"] + " " + aliases_text))
            score += 0.5 * sum(1 for t in q_terms if t in title_alias_tokens)
            if score > 0:
                scored.append((score, paper))

        scored.sort(key=lambda x: -x[0])
        if not scored:
            return "No matching papers. Try broader or different terms."

        lines = []
        for score, paper in scored[:top_k]:
            lines.append(
                f"[{paper['id']}] {paper['title']} ({paper['year']}) "
                f"score={score:.2f}\n    {paper['abstract'][:180]}..."
            )
        return "\n".join(lines)


class FetchPaper(Tool):
    name = "fetch_paper"
    description = "Retrieve the full stored record for one paper by its corpus ID."
    input_schema = {
        "type": "object",
        "properties": {"paper_id": {"type": "string", "pattern": r"^[A-Za-z0-9._-]+$"}},
        "required": ["paper_id"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.by_id = {p["id"]: p for p in _load_corpus()}

    def run(self, paper_id: str) -> str:
        paper = self.by_id.get(paper_id)
        if paper is None:
            raise KeyError(f"no paper with id {paper_id!r}")
        return json.dumps(paper, indent=2, ensure_ascii=False)


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


class Calculate(Tool):
    name = "calculate"
    description = "Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses."
    input_schema = {
        "type": "object",
        "properties": {"expression": {"type": "string", "maxLength": 200}},
        "required": ["expression"],
        "additionalProperties": False,
    }

    def run(self, expression: str) -> str:
        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
                return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                val = _eval(node.operand)
                return val if isinstance(node.op, ast.UAdd) else -val
            raise ValueError(f"unsupported expression element: {type(node).__name__}")

        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))


class ListPapers(Tool):
    name = "list_papers"
    description = (
        "List papers in the corpus. Returns id, title, and year for every paper. "
        "Pass 'year' to restrict results to a specific publication year."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "year": {
                "type": "integer",
                "description": "Filter by publication year (optional).",
            },
        },
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.corpus = _load_corpus()

    def run(self, year: int | None = None) -> str:
        papers = self.corpus if year is None else [p for p in self.corpus if p["year"] == year]
        if not papers:
            suffix = f" for year {year}" if year is not None else ""
            return f"No papers found{suffix}."
        return "\n".join(f"[{p['id']}] {p['title']} ({p['year']})" for p in papers)


def default_registry():
    from .base import ToolRegistry
    return ToolRegistry([SearchPapers(), FetchPaper(), Calculate(), ListPapers()])

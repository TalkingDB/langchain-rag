"""Load the evaluation dataset (.csv, .jsonl or .json).

RAGAS testset columns are mapped automatically:
    question         <- question | user_input | query
    ground_truth     <- ground_truth | reference | reference_answer | answer
    expected_sources <- expected_sources | source | sources | source_document
    gold_context     <- gold_context | reference_contexts | context | contexts
Any other columns (persona_name, query_style, ...) are kept in `metadata`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ALIASES = {
    "id": ("id", "question_id"),
    "question": ("question", "user_input", "query"),
    "ground_truth": ("ground_truth", "reference", "reference_answer", "answer"),
    "expected_sources": ("expected_sources", "source", "sources", "source_document"),
    "gold_context": ("gold_context", "reference_contexts", "context", "contexts"),
}


@dataclass
class EvalItem:
    id: str
    question: str
    ground_truth: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    gold_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def split_jsonl(text: str) -> list[str]:
    """Split JSONL into records on "\\n" only. str.splitlines() would also split on characters
    such as \\x0c or \\u2028 that occur inside PDF-derived strings and corrupt records."""
    return [line for line in text.split("\n") if line.strip()]


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _as_list(value: Any) -> list[str]:
    value = _maybe_json(value)
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split("|") if s.strip()]


def normalize_row(row: dict[str, Any], index: int) -> EvalItem:
    used: set[str] = set()

    def pick(name: str) -> Any:
        for alias in _ALIASES[name]:
            if alias in row and row[alias] not in (None, ""):
                used.add(alias)
                return row[alias]
        return None

    question = pick("question")
    if not question:
        raise ValueError(f"Row {index + 1} has no question/user_input column")
    gold = pick("gold_context")
    reference = pick("ground_truth")
    return EvalItem(
        id=str(pick("id") or f"q{index + 1:04d}"),
        question=str(question).strip(),
        ground_truth=str(reference).strip() if reference is not None else None,
        expected_sources=_as_list(pick("expected_sources")),
        gold_context=("\n\n".join(_as_list(gold)) or None) if gold is not None else None,
        metadata={k: v for k, v in row.items() if k not in used and v not in (None, "")},
    )


def load_dataset(path: Path, limit: int | None = None) -> list[EvalItem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        text = path.read_text(encoding="utf-8-sig")
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in split_jsonl(text)]
        else:
            data = json.loads(text)
            rows = data if isinstance(data, list) else data.get("items", [])
    items = [normalize_row(r, i) for i, r in enumerate(rows)]
    return items[:limit] if limit else items

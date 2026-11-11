"""Deterministic metrics: answer vs reference, and retrieval (passage and document level).

Retrieval relevance
  passage level (`retrieval.*`): a retrieved chunk is relevant when it lies inside the
      question's gold passage -- overlap coefficient |chunk ∩ gold| / min(|chunk|, |gold|) over
      word sets >= 0.9. Chunk-size independent; calibrated so gold chunks score >= 0.9 and
      non-gold chunks stay below it.
  document level (`retrieval.doc_*`): a retrieved chunk is relevant when it comes from the
      question's expected source document.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

OVERLAP_THRESHOLD = 0.9
MIN_CHUNK_WORDS = 8  # a lone heading can't be judged reliably

_WORD = re.compile(r"\w+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


# ----------------------------------------------------------------- answer vs reference
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def contains_reference(answer: str, reference: str) -> float:
    ref = normalize(reference)
    return 1.0 if ref and ref in normalize(answer) else 0.0


def token_f1(answer: str, reference: str) -> float:
    a, r = normalize(answer).split(), normalize(reference).split()
    if not a or not r:
        return 0.0
    common = sum((Counter(a) & Counter(r)).values())
    if not common:
        return 0.0
    p, rc = common / len(a), common / len(r)
    return round(2 * p * rc / (p + rc), 4)


# ----------------------------------------------------------------- retrieval
def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def overlap_coefficient(chunk: str, gold: str) -> float:
    c, g = _words(chunk), _words(gold)
    if len(c) < MIN_CHUNK_WORDS or not g:
        return 0.0
    return len(c & g) / min(len(c), len(g))


def source_stem(source: str | None) -> str:
    """'data/books/x.pdf', 'x.pdf (x-12)' and 'x.docx' all become 'x'."""
    if not source:
        return ""
    name = re.sub(r"\s*\([^)]*\)\s*$", "", str(source)).replace("\\", "/").rsplit("/", 1)[-1]
    return Path(name).stem.lower()


def ranking_metrics(relevant: list[int]) -> dict[str, float]:
    """Binary relevance list (in rank order) -> hit rate, MRR, precision, recall, nDCG.
    Recall assumes one gold item per question (found or not); callers may override it."""
    first = next((i for i, r in enumerate(relevant, start=1) if r), None)
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(relevant, start=1))
    n_ideal = max(1, sum(relevant))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return {
        "hit_rate": 1.0 if any(relevant) else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "precision": sum(relevant) / len(relevant) if relevant else 0.0,
        "recall": 1.0 if any(relevant) else 0.0,
        "ndcg": dcg / idcg if any(relevant) else 0.0,
    }


def passage_retrieval(contexts: list[str], gold: str | None) -> dict[str, float] | None:
    if not gold:
        return None
    rel = [1 if overlap_coefficient(c, gold) >= OVERLAP_THRESHOLD else 0 for c in contexts]
    return ranking_metrics(rel)


def document_retrieval(sources: list[str], expected: list[str]) -> dict[str, float] | None:
    want = {source_stem(s) for s in expected if s}
    if not want:
        return None
    got = [source_stem(s) for s in sources]
    m = ranking_metrics([1 if g in want else 0 for g in got])
    m["recall"] = len(want & set(got)) / len(want)
    return m

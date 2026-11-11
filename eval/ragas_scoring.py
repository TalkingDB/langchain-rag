"""Optional RAGAS metrics (pip install -r eval/requirements.txt).

Uses the RAGAS 0.4 "collections" API (async metric.ascore) with an OpenAI client, on a seeded
sample of answered questions (the same questions for every mode, so scores are comparable).

  answer_correctness  question, answer, reference        (+ embeddings)
  faithfulness        question, answer, retrieved contexts
  context_precision   question, reference, retrieved contexts
  context_recall      question, reference, retrieved contexts
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import warnings
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]
NEEDS_CONTEXT = {"faithfulness", "context_precision", "context_recall"}
NEEDS_REFERENCE = {"answer_correctness", "context_precision", "context_recall"}


def select_sample(rows: list[dict[str, Any]], size: int | None, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample by question id (independent of row order)."""
    ok = [r for r in rows if "error" not in r]
    if size is None or size >= len(ok):
        return ok
    chosen = set(random.Random(seed).sample(sorted(r["id"] for r in ok), size))
    return [r for r in ok if r["id"] in chosen]


class RagasScorer:
    def __init__(self, metrics: list[str], model: str = "gpt-4.1-mini",
                 embedding_model: str = "text-embedding-3-small", concurrency: int = 4):
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import ragas.metrics.collections as mc
                from openai import AsyncOpenAI
                from ragas.embeddings import embedding_factory
                from ragas.llms import llm_factory
        except ImportError as e:
            raise ImportError("RAGAS is not installed: pip install -r eval/requirements.txt") from e
        unknown = set(metrics) - set(DEFAULT_METRICS)
        if unknown:
            raise ValueError(f"Unknown RAGAS metrics: {sorted(unknown)}")
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)
        llm = llm_factory(model, client=client)
        emb = embedding_factory("openai", model=embedding_model, client=client)
        factories = {
            "answer_correctness": lambda: mc.AnswerCorrectness(llm=llm, embeddings=emb),
            "faithfulness": lambda: mc.Faithfulness(llm=llm),
            "context_precision": lambda: mc.ContextPrecision(llm=llm),
            "context_recall": lambda: mc.ContextRecall(llm=llm),
        }
        self.metrics = {m: factories[m]() for m in metrics}
        self.concurrency = concurrency

    async def _one(self, name: str, metric: Any, row: dict[str, Any]) -> float | None:
        q, a, ref, ctx = row["question"], row["answer"], row.get("ground_truth"), row.get("contexts")
        if name in NEEDS_REFERENCE and not ref:
            return None
        if name in NEEDS_CONTEXT and not ctx:
            return None
        kwargs = {
            "answer_correctness": {"user_input": q, "response": a, "reference": ref},
            "faithfulness": {"user_input": q, "response": a, "retrieved_contexts": ctx},
            "context_precision": {"user_input": q, "reference": ref, "retrieved_contexts": ctx},
            "context_recall": {"user_input": q, "retrieved_contexts": ctx, "reference": ref},
        }[name]
        try:
            result = await metric.ascore(**kwargs)
        except Exception as e:  # one failed metric must not sink the run
            logger.warning("RAGAS %s failed for %s: %s", name, row["id"], e)
            return None
        try:
            return round(float(getattr(result, "value", result)), 4)
        except (TypeError, ValueError):
            return None

    async def _all(self, rows: list[dict[str, Any]]) -> None:
        sem = asyncio.Semaphore(self.concurrency)

        async def run(name: str, metric: Any, row: dict[str, Any]) -> None:
            async with sem:
                row[f"ragas.{name}"] = await self._one(name, metric, row)

        await asyncio.gather(*(run(n, m, r) for r in rows for n, m in self.metrics.items()))

    def score(self, rows: list[dict[str, Any]]) -> None:
        """Add ragas.<metric> to each row in place. Unanswered rows score 0."""
        answered = [r for r in rows if r.get("found")]
        asyncio.run(self._all(answered))
        for r in rows:
            if not r.get("found"):
                for m in self.metrics:
                    r[f"ragas.{m}"] = 0.0

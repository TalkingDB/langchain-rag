"""LLM-as-judge metrics, scored 1-5 by an OpenAI model and rescaled to 0-1 (1->0 ... 5->1).

correctness        same facts as the reference answer?          (needs a reference)
answer_relevance   does the answer address the question?
faithfulness       is every claim supported by the retrieved context?
context_relevance  does the retrieved context contain what is needed?

Uses the OpenAI SDK directly with JSON output, so it works with this repo's pinned
langchain/openai versions.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

RUBRICS = {
    "correctness": (
        "Judge CORRECTNESS: does the ANSWER convey the same facts as the REFERENCE answer? "
        "5 = equivalent, 3 = partially correct, 1 = wrong or missing key facts. "
        "Extra correct detail is fine."
    ),
    "answer_relevance": (
        "Judge ANSWER RELEVANCE: does the ANSWER directly and completely address the QUESTION? "
        "5 = fully addresses it, 1 = off-topic or evasive."
    ),
    "faithfulness": (
        "Judge FAITHFULNESS: are all factual claims in the ANSWER supported by the CONTEXT? "
        "5 = fully supported, 1 = mostly unsupported or contradicts the context. "
        "An answer that correctly says the context lacks the information is faithful."
    ),
    "context_relevance": (
        "Judge CONTEXT RELEVANCE: does the CONTEXT contain the information needed to answer "
        "the QUESTION? 5 = everything needed, 1 = nothing useful."
    ),
}
ALL_METRICS = list(RUBRICS)


class Judge:
    def __init__(self, model: str = "gpt-4.1-mini", metrics: list[str] | None = None):
        self.model = model
        self.metrics = metrics or ALL_METRICS
        unknown = set(self.metrics) - set(RUBRICS)
        if unknown:
            raise ValueError(f"Unknown judge metrics: {sorted(unknown)}")
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=5)

    def _ask(self, prompt: str) -> tuple[float, str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Reply with JSON: "
                 '{"reason": "<one or two sentences>", "score": <integer 1-5>}.'},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = min(5, max(1, int(data.get("score", 1))))
        return (score - 1) / 4, str(data.get("reason", ""))

    def score(self, question: str, answer: str, contexts: list[str] | None,
              reference: str | None) -> dict[str, dict]:
        context = "\n\n---\n\n".join(contexts or []) or "(empty)"
        out: dict[str, dict] = {}
        for metric in self.metrics:
            if metric == "correctness" and not reference:
                continue
            if metric in ("faithfulness", "context_relevance") and contexts is None:
                continue  # retrieved text not available
            parts = [RUBRICS[metric], f"QUESTION:\n{question}"]
            if metric in ("faithfulness", "context_relevance"):
                parts.append(f"CONTEXT:\n{context}")
            if metric != "context_relevance":
                parts.append(f"ANSWER:\n{answer}")
            if metric == "correctness":
                parts.append(f"REFERENCE:\n{reference}")
            value, reason = self._ask("\n\n".join(parts))
            out[metric] = {"score": value, "reason": reason}
        return out

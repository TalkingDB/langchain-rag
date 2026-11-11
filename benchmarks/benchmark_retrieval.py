"""Benchmark symbolic retrieval: exact synset match (range index) vs Personalized PageRank.

For a random sample of testset questions, every method gets the same WSD synsets for the
question, then retrieves chunks from Neo4j, answers with the pipeline's prompt and model,
and is graded on:

  hit@k        a retrieved chunk comes from the question's gold reference context
  MRR          1 / rank of the first such chunk (0 if none)
  ans_in_ctx   the reference answer appears verbatim (normalised) in the retrieved text
  correct      an LLM judge says the generated answer matches the reference answer
  chunks/chars how much context was sent to the LLM

Usage (from the langchain-rag directory):
    python benchmarks/benchmark_retrieval.py --csv ../testset_stripped.csv -n 20 --seed 42
"""
import argparse
import ast
import csv
import json
import os
import re
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

from neo4j_graphdb import GraphRepository  # noqa: E402
from query_data import PROMPT_TEMPLATE  # noqa: E402
from symbolic_link_data import link_entities, to_senses  # noqa: E402

TOP_K = 5
GOLD_OVERLAP = 0.5  # share of a chunk's word 4-grams that must fall inside the gold context

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer: {reference}
System answer: {answer}

Does the system answer convey the same fact as the reference answer? Extra correct detail is fine.
Saying it does not know, or giving a different fact, is wrong. Reply with exactly one word: CORRECT or INCORRECT."""


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list[str], n: int = 4) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def from_gold(chunk_text: str, gold_grams: set) -> bool:
    grams = ngrams(words(chunk_text))
    return bool(grams) and len(grams & gold_grams) / len(grams) >= GOLD_OVERLAP


def parse_contexts(raw: str) -> list[str]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def retrieve(graph: GraphRepository, method: str, synset_ids: list[str]) -> list[dict]:
    if not synset_ids:
        return []
    if method == "exact_all":
        return graph.find_chunks_by_synset_ids(synset_ids)
    if method == "exact_top5":
        return graph.find_chunks_by_synset_ids(synset_ids)[:TOP_K]
    if method == "ppr_top5":
        return graph.find_chunks_by_ppr(synset_ids, top_k=TOP_K)
    raise ValueError(method)


METHODS = ["exact_all", "exact_top5", "ppr_top5"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../testset_stripped.csv")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="benchmarks/results.json")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sample = random.Random(args.seed).sample(rows, args.n)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    graph = GraphRepository()
    results = []
    try:
        for qi, row in enumerate(sample, 1):
            question, reference = row["user_input"], row["reference"]
            gold_grams = set().union(*(ngrams(words(c)) for c in parse_contexts(row["reference_contexts"])))

            senses = to_senses(link_entities(question))
            synset_ids = list(dict.fromkeys(s["synset_id"] for s in senses))
            entry = {
                "question": question,
                "reference": reference,
                "source": row["source"],
                "synsets": [(s["surface"], s["synset_id"]) for s in senses],
                "methods": {},
            }

            for method in METHODS:
                t = time.perf_counter()
                chunks = retrieve(graph, method, synset_ids)
                retrieval_ms = (time.perf_counter() - t) * 1000

                ranks = [i for i, c in enumerate(chunks, 1) if from_gold(c["text"], gold_grams)]
                context = "\n\n---\n\n".join(c["text"] for c in chunks)
                ans_norm = " ".join(words(reference))
                ans_in_ctx = bool(ans_norm) and ans_norm in " ".join(words(context))

                if chunks:
                    answer = llm.invoke(prompt_template.format(context=context, question=question)).content
                    verdict = llm.invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)).content
                    correct = verdict.strip().upper().startswith("CORRECT")
                else:
                    answer, correct = "(no chunks retrieved)", False

                entry["methods"][method] = {
                    "retrieval_ms": round(retrieval_ms, 1),
                    "n_chunks": len(chunks),
                    "context_chars": len(context),
                    "hit": bool(ranks),
                    "first_gold_rank": ranks[0] if ranks else None,
                    "rr": 1 / ranks[0] if ranks else 0.0,
                    "ans_in_ctx": ans_in_ctx,
                    "correct": correct,
                    "answer": answer,
                    "chunk_ids": [c["chunk_id"] for c in chunks],
                }

            m = entry["methods"]
            print(
                f"[{qi:2}/{args.n}] {question[:60]:60} | "
                + " | ".join(f"{k}: hit={int(m[k]['hit'])} ok={int(m[k]['correct'])} n={m[k]['n_chunks']}" for k in METHODS),
                flush=True,
            )
            results.append(entry)
    finally:
        graph.close()

    summary = {}
    for method in METHODS:
        ms = [r["methods"][method] for r in results]
        summary[method] = {
            "hit_rate": statistics.mean(m["hit"] for m in ms),
            "mrr": statistics.mean(m["rr"] for m in ms),
            "ans_in_ctx": statistics.mean(m["ans_in_ctx"] for m in ms),
            "accuracy": statistics.mean(m["correct"] for m in ms),
            "avg_chunks": statistics.mean(m["n_chunks"] for m in ms),
            "avg_context_chars": statistics.mean(m["context_chars"] for m in ms),
            "median_retrieval_ms": statistics.median(m["retrieval_ms"] for m in ms),
        }

    print("\nmethod       hit   MRR   ans_in_ctx  accuracy  avg_chunks  avg_ctx_chars  med_retr_ms")
    for method, s in summary.items():
        print(
            f"{method:11} {s['hit_rate']:5.2f} {s['mrr']:5.2f} {s['ans_in_ctx']:10.2f} {s['accuracy']:9.2f}"
            f" {s['avg_chunks']:11.1f} {s['avg_context_chars']:14.0f} {s['median_retrieval_ms']:12.1f}"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

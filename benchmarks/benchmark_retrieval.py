"""Benchmark retrieval: exact synset match, Leiden community ranking, and vector (Chroma) search.

For a random sample of testset questions, every method gets the same WSD synsets for the
question, then retrieves chunks from Neo4j, answers with the pipeline's prompt and model,
and is graded on:

  hit@k        a retrieved chunk comes from the question's gold reference context
  MRR          1 / rank of the first such chunk (0 if none)
  ans_in_ctx   the reference answer appears verbatim (normalised) in the retrieved text
  correct      an LLM judge says the generated answer matches the reference answer
  chunks/chars how much context was sent to the LLM

Usage (from the langchain-rag directory):
    python benchmarks/benchmark_retrieval.py -n 20 --seed 42          # random sample
    python benchmarks/benchmark_retrieval.py -n 20 --first            # first 20 rows of the sheet
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
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)  # the repo's .env must beat a stale OS OPENAI_API_KEY

from neo4j_graphdb import GraphRepository  # noqa: E402
from query_data import PROMPT_TEMPLATE  # noqa: E402
from symbolic_link_data import link_entities, to_senses  # noqa: E402

TOP_K = 5
VECTOR_MIN_RELEVANCE = 0.7  # same cutoff as api/services/vector_service.py
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


def retrieve_vector(vectordb, question: str) -> list[dict]:
    """Top-k Chroma similarity search, exactly like the vector pipeline: if the best chunk's
    relevance is below the cutoff, nothing is retrieved (the pipeline then answers nothing)."""
    results = vectordb.similarity_search_with_relevance_scores(question, k=TOP_K)
    if not results or results[0][1] < VECTOR_MIN_RELEVANCE:
        return []
    return [
        {
            "chunk_id": f"{os.path.splitext(os.path.basename(doc.metadata.get('source', '')))[0]}"
                        f"@{doc.metadata.get('start_index', i)}",
            "source": doc.metadata.get("source"),
            "text": doc.page_content,
            "score": score,
        }
        for i, (doc, score) in enumerate(results)
    ]


def retrieve(graph: GraphRepository, method: str, synset_ids: list[str],
             question: str = "", vectordb=None) -> list[dict]:
    if method == "vector_top5":
        return retrieve_vector(vectordb, question)
    if not synset_ids:
        return []
    if method == "exact_all":
        return graph.find_chunks_by_synset_ids(synset_ids)
    if method == "exact_top5":
        return graph.find_chunks_by_synset_ids(synset_ids)[:TOP_K]
    if method == "leiden_top5":
        return graph.find_chunks_by_leiden(synset_ids, top_k=TOP_K)
    if method == "leiden_top3":
        return graph.find_chunks_by_leiden(synset_ids, top_k=3)
    raise ValueError(method)


METHODS = ["exact_all", "exact_top5", "leiden_top5", "vector_top5"]  # run by default
ALL_METHODS = METHODS + ["leiden_top3"]
SYMBOLIC_METHODS = {"exact_all", "exact_top5", "leiden_top5", "leiden_top3"}
NETWORK_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError)


def timed_wsd(question: str, attempts: int = 6) -> tuple[list[dict], float]:
    """WSD-link the question, retrying dropped connections with backoff. Returns the senses
    and the duration of the successful call only, so retries don't distort the timings."""
    for attempt in range(1, attempts + 1):
        t = time.perf_counter()
        try:
            senses = to_senses(link_entities(question))
            return senses, (time.perf_counter() - t) * 1000
        except NETWORK_ERRORS as e:
            if attempt == attempts:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1))
            print(f"  WSD call failed ({e}); retry {attempt}/{attempts - 1} in {wait}s", flush=True)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/eval/testset_stripped.csv")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first", action="store_true",
                        help="use the first N rows of the sheet instead of a random sample")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=ALL_METHODS)
    parser.add_argument("--out", default="benchmarks/results.json")
    parser.add_argument("--resume", action="store_true",
                        help="keep the questions already in --out (saved after each question) and run the rest")
    args = parser.parse_args()
    methods = args.methods

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    indexed = list(enumerate(rows, 1))
    sample = indexed[: args.n] if args.first else random.Random(args.seed).sample(indexed, args.n)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    graph = GraphRepository()
    needs_wsd = any(m in SYMBOLIC_METHODS for m in methods)
    vectordb = None
    if "vector_top5" in methods:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        from query_data import CHROMA_PATH

        vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = [r for r in json.load(f)["results"] if set(methods) <= set(r["methods"])]
        print(f"Resuming: {len(results)} questions already in {args.out}", flush=True)
    done = {r["row"] for r in results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save(summary=None):
        """Write after every question, so a crash loses at most the question in progress."""
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "summary": summary or {}, "results": results},
                      f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.out)

    try:
        for qi, (row_no, row) in enumerate(sample, 1):
            if row_no in done:
                continue
            question, reference = row["user_input"], row["reference"]
            gold_grams = set().union(*(ngrams(words(c)) for c in parse_contexts(row["reference_contexts"])))

            senses, wsd_ms = timed_wsd(question) if needs_wsd else ([], 0.0)
            synset_ids = list(dict.fromkeys(s["synset_id"] for s in senses))
            entry = {
                "row": row_no,
                "question": question,
                "reference": reference,
                "source": row["source"],
                "synsets": [(s["surface"], s["synset_id"]) for s in senses],
                "wsd_ms": round(wsd_ms, 1),
                "methods": {},
            }

            for method in methods:
                t = time.perf_counter()
                chunks = retrieve(graph, method, synset_ids, question, vectordb)
                retrieval_ms = (time.perf_counter() - t) * 1000

                ranks = [i for i, c in enumerate(chunks, 1) if from_gold(c["text"], gold_grams)]
                context = "\n\n---\n\n".join(c["text"] for c in chunks)
                ans_norm = " ".join(words(reference))
                ans_in_ctx = bool(ans_norm) and ans_norm in " ".join(words(context))

                answer_ms = 0.0
                if chunks:
                    t = time.perf_counter()
                    answer = llm.invoke(prompt_template.format(context=context, question=question)).content
                    answer_ms = (time.perf_counter() - t) * 1000
                    verdict = llm.invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)).content
                    correct = verdict.strip().upper().startswith("CORRECT")
                else:
                    answer, correct = "(no chunks retrieved)", False

                entry["methods"][method] = {
                    "retrieval_ms": round(retrieval_ms, 1),
                    "answer_ms": round(answer_ms, 1),
                    "pipeline_ms": round((wsd_ms if method in SYMBOLIC_METHODS else 0.0)
                                         + retrieval_ms + answer_ms, 1),
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
                + " | ".join(f"{k}: hit={int(m[k]['hit'])} ok={int(m[k]['correct'])} n={m[k]['n_chunks']}" for k in methods),
                flush=True,
            )
            results.append(entry)
            save()
    finally:
        graph.close()
    results.sort(key=lambda r: r["row"])

    summary = {}
    for method in methods:
        ms = [r["methods"][method] for r in results]
        summary[method] = {
            "hit_rate": statistics.mean(m["hit"] for m in ms),
            "mrr": statistics.mean(m["rr"] for m in ms),
            "ans_in_ctx": statistics.mean(m["ans_in_ctx"] for m in ms),
            "accuracy": statistics.mean(m["correct"] for m in ms),
            "avg_chunks": statistics.mean(m["n_chunks"] for m in ms),
            "avg_context_chars": statistics.mean(m["context_chars"] for m in ms),
            "median_retrieval_ms": statistics.median(m["retrieval_ms"] for m in ms),
            "mean_retrieval_ms": statistics.mean(m["retrieval_ms"] for m in ms),
            "max_retrieval_ms": max(m["retrieval_ms"] for m in ms),
            "median_pipeline_ms": statistics.median(m["pipeline_ms"] for m in ms),
        }

    print("\nmethod       hit   MRR   ans_in_ctx  accuracy  avg_chunks  avg_ctx_chars  med_retr_ms")
    for method, s in summary.items():
        print(
            f"{method:11} {s['hit_rate']:5.2f} {s['mrr']:5.2f} {s['ans_in_ctx']:10.2f} {s['accuracy']:9.2f}"
            f" {s['avg_chunks']:11.1f} {s['avg_context_chars']:14.0f} {s['median_retrieval_ms']:12.1f}"
        )

    save(summary)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

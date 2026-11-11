"""Evaluate the vector and/or symbolic RAG pipelines on a question set.

Calls api/services/{vector,symbolic}_service.query() in-process (no server needed), captures
the answer *and* the retrieved chunk text, then scores every question:

  retrieval   passage level vs the gold passage, plus document level vs the expected source
  answer      answer rate, contains-reference, token F1
  LLM judge   correctness, answer relevance, faithfulness, context relevance (gpt-4.1-mini, 0-1)
  RAGAS       answer correctness, faithfulness, context precision, context recall (optional)

Writes eval/results/<run_id>/<mode>/{results.jsonl,summary.json,config.json},
comparison.csv and report.html.

Usage (from the repo root, with Chroma + Neo4j already ingested):
    python -m eval.run                                   # both modes, service defaults
    python -m eval.run --modes vector symbolic --top-k 5
    python -m eval.run --modes vector --limit 10 --no-ragas   # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.dataset import EvalItem, load_dataset
from eval.judge import ALL_METRICS as JUDGE_METRICS
from eval.judge import Judge
from eval.metrics import (
    OVERLAP_THRESHOLD,
    contains_reference,
    document_retrieval,
    passage_retrieval,
    token_f1,
)

logger = logging.getLogger("eval")
METRIC_PREFIXES = ("retrieval.", "answer.", "judge.", "ragas.")
CORRECT_THRESHOLD = 0.75

# Fixed parameters of this repo's pipelines (create_database.py / query services), recorded in
# each run's config.json so the report shows what was evaluated.
PIPELINE = {
    "data.corpus_dir": "data/books",
    "chunking.splitter": "recursive",
    "chunking.chunk_size": 300,
    "chunking.chunk_overlap": 100,
    "embedding.model": "text-embedding-ada-002 (OpenAIEmbeddings default)",
    "generation.model": "gpt-4.1-mini",
    "generation.temperature": 0,
}


def get_service(mode: str):
    if mode == "vector":
        from api.services import vector_service as svc
    elif mode == "symbolic":
        from api.services import symbolic_service as svc
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return svc


# ----------------------------------------------------------------- per question
def evaluate_item(item: EvalItem, mode: str, top_k: int | None, judge: Judge | None) -> dict:
    svc = get_service(mode)
    t0 = time.perf_counter()
    try:
        res = svc.query(item.question, top_k=top_k, return_contexts=True)
    except Exception as e:  # keep going; record the failure
        logger.warning("[%s] %s failed: %s", mode, item.id, e)
        return {"id": item.id, "question": item.question, "error": f"{type(e).__name__}: {e}"}
    latency = time.perf_counter() - t0

    found = bool(res.get("found")) and bool((res.get("answer") or "").strip())
    answer = (res.get("answer") or "").strip()
    contexts = res.get("contexts") or []
    sources = [s for s in (res.get("sources") or []) if s]
    row: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answer": answer if found else "(no answer: nothing relevant retrieved)",
        "ground_truth": item.ground_truth,
        "found": found,
        "retrieved_sources": sources,
        "contexts": contexts,
        "num_contexts": len(contexts),
        "synsets": [s.get("synset_id") for s in res.get("synsets") or []],
        "retrieval_latency_s": 0.0,
        "generation_latency_s": round(latency, 4),  # end-to-end query time
        "input_tokens": 0,
        "output_tokens": 0,
        "answer.answered": 1.0 if found else 0.0,
    }
    for prefix, metrics in (
        ("retrieval.", passage_retrieval(contexts, item.gold_context)),
        ("retrieval.doc_", document_retrieval(sources, item.expected_sources)),
    ):
        if metrics:
            keep = metrics if prefix == "retrieval." else {
                k: metrics[k] for k in ("hit_rate", "mrr")}
            row.update({f"{prefix}{k}": v for k, v in keep.items()})
    if item.ground_truth:
        row["answer.contains_reference"] = contains_reference(answer, item.ground_truth) if found else 0.0
        row["answer.token_f1"] = token_f1(answer, item.ground_truth) if found else 0.0
    for key, value in item.metadata.items():
        row[f"group.{key}"] = value

    if judge:
        if not found:
            for m in judge.metrics:
                if m == "correctness" and not item.ground_truth:
                    continue
                row[f"judge.{m}"] = 0.0
                row[f"judge.{m}.reason"] = "No answer returned."
        else:
            try:
                for m, v in judge.score(item.question, answer, contexts, item.ground_truth).items():
                    row[f"judge.{m}"] = v["score"]
                    row[f"judge.{m}.reason"] = v["reason"]
            except Exception as e:
                logger.warning("[%s] judge failed for %s: %s", mode, item.id, e)
                row["judge_error"] = str(e)
    return row


# ----------------------------------------------------------------- aggregation
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 4)


def _metric_keys(rows: list[dict]) -> list[str]:
    return sorted({k for r in rows for k in r
                   if k.startswith(METRIC_PREFIXES) and not k.endswith(".reason")})


def aggregate(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    summary: dict[str, Any] = {"questions": len(rows), "errors": len(rows) - len(ok)}
    for k in _metric_keys(ok):
        summary[k] = _mean([r[k] for r in ok if r.get(k) is not None])
    lat = [r["generation_latency_s"] for r in ok]
    summary["latency.total_p50_s"] = _pct(lat, 0.5)
    summary["latency.total_p95_s"] = _pct(lat, 0.95)
    if ok:
        summary["context.sources_mean"] = round(statistics.fmean(r["num_contexts"] for r in ok), 2)
        if any(r["synsets"] for r in ok):
            summary["context.synsets_mean"] = round(
                statistics.fmean(len(r["synsets"]) for r in ok), 2)
        summary["correct"] = sum(
            1 for r in ok if (r.get("judge.correctness") or 0) >= CORRECT_THRESHOLD)
    return summary


def group_breakdown(rows: list[dict], fields: list[str]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    keys = _metric_keys(ok)
    out: dict[str, Any] = {}
    for field in fields:
        groups: dict[str, list[dict]] = {}
        for r in ok:
            groups.setdefault(str(r.get(f"group.{field}")), []).append(r)
        out[field] = {
            value: {"n": len(rs), **{k: _mean([r[k] for r in rs if r.get(k) is not None])
                                     for k in keys}}
            for value, rs in sorted(groups.items())
        }
    return out


# ----------------------------------------------------------------- saving
def save_experiment(run_dir: Path, name: str, run_id: str, config: dict, rows: list[dict],
                    summary: dict, groups: dict) -> None:
    safe = "".join(c if c.isalnum() or c in "-_=.," else "_" for c in name)
    d = run_dir / safe
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(
        {"experiment": name, "run_id": run_id, "summary": summary, "groups": groups}, indent=2),
        encoding="utf-8")
    with (d / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_comparison(run_dir: Path, results: list[tuple[str, dict]]) -> None:
    keys = sorted({k for _, s in results for k in s})
    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", *keys])
        for name, s in results:
            w.writerow([name, *(s.get(k) for k in keys)])


def print_table(results: list[tuple[str, dict]]) -> None:
    keys = [k for k in ("correct", "answer.answered", "retrieval.hit_rate", "retrieval.mrr",
                        "retrieval.doc_hit_rate", "answer.contains_reference",
                        "judge.correctness", "judge.faithfulness", "ragas.answer_correctness",
                        "ragas.faithfulness", "ragas.context_recall", "context.sources_mean",
                        "latency.total_p50_s")
            if any(s.get(k) is not None for _, s in results)]
    width = max(len(k) for k in keys) + 2
    print("\n" + "metric".ljust(width) + "".join(n.rjust(24) for n, _ in results))
    for k in keys:
        cells = []
        for _, s in results:
            v = s.get(k)
            cells.append(("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v))
                         .rjust(24))
        print(k.ljust(width) + "".join(cells))


# ----------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the vector / symbolic RAG pipelines.")
    ap.add_argument("--dataset", type=Path, default=Path("data/eval/testset_stripped.csv"))
    ap.add_argument("--modes", nargs="+", choices=["vector", "symbolic"],
                    default=["vector", "symbolic"])
    ap.add_argument("--top-k", type=int, default=None,
                    help="chunks per question for both modes (default: vector 3, symbolic all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--workers", type=int, default=1,
                    help="questions in parallel (symbolic calls the shared WSD service; keep low)")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--judge-metrics", nargs="+", choices=JUDGE_METRICS, default=JUDGE_METRICS)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--ragas-sample", type=int, default=None,
                    help="RAGAS on a seeded sample of N questions (default: all)")
    ap.add_argument("--ragas-model", default="gpt-4.1-mini")
    ap.add_argument("--group-by", nargs="*", default=["query_style", "query_length", "persona_name"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--title", default=None, help="report heading")
    ap.add_argument("--notes", type=Path, default=None, help="markdown findings for the report")
    args = ap.parse_args(argv)

    # The repo's .env must win over a stale OPENAI_API_KEY in the OS environment.
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "openai", "neo4j", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    items = load_dataset(args.dataset, args.limit)
    judge = None if args.no_judge else Judge(args.judge_model, args.judge_metrics)
    scorer = None
    if not args.no_ragas:
        from eval.ragas_scoring import DEFAULT_METRICS, RagasScorer

        scorer = RagasScorer(DEFAULT_METRICS, model=args.ragas_model)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run %s: %d questions, modes %s, top_k=%s -> %s", run_id, len(items),
                args.modes, args.top_k or "default", run_dir)

    results: list[tuple[str, dict]] = []
    for mode in args.modes:
        name = f"langchain-rag[mode={mode}]"
        t0 = time.perf_counter()
        logger.info("[%s] querying %d questions ...", mode, len(items))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda it: evaluate_item(it, mode, args.top_k, judge), items))
        if scorer:
            from eval.ragas_scoring import select_sample

            sample = select_sample(rows, args.ragas_sample, args.seed)
            logger.info("[%s] RAGAS on %d questions ...", mode, len(sample))
            scorer.score(sample)
        summary = aggregate(rows)
        if scorer:
            summary["ragas.sample_size"] = len(sample)
        summary["wall_time_s"] = round(time.perf_counter() - t0, 2)
        groups = group_breakdown(rows, [g for g in args.group_by
                                        if any(f"group.{g}" in r for r in rows)])
        config = {
            "name": name,
            "description": f"langchain-rag {mode} pipeline, called in-process",
            "data.eval_dataset": str(args.dataset).replace("\\", "/"),
            "data.max_questions": args.limit,
            **PIPELINE,
            "retrieval.strategy": mode,
            "retrieval.top_k": args.top_k if args.top_k else
            ("3 (service default)" if mode == "vector" else "all matching (service default)"),
            "evaluation.relevance": "overlap",
            "evaluation.overlap_threshold": OVERLAP_THRESHOLD,
            "evaluation.judge.model": None if args.no_judge else args.judge_model,
            "evaluation.ragas.enabled": not args.no_ragas,
            "evaluation.ragas.model": None if args.no_ragas else args.ragas_model,
            "evaluation.ragas.sample_size": args.ragas_sample,
            "seed": args.seed,
        }
        save_experiment(run_dir, name, run_id, config, rows, summary, groups)
        results.append((mode, summary))
        logger.info("[%s] done: %s/%d correct, %d errors, %.0fs", mode, summary.get("correct"),
                    len(rows), summary["errors"], summary["wall_time_s"])

    write_comparison(run_dir, results)
    print_table(results)

    from eval.report import build_report

    page = build_report([run_dir], title=args.title,
                        notes=args.notes.read_text(encoding="utf-8") if args.notes else None)
    (run_dir / "report.html").write_text(page, encoding="utf-8")
    print(f"\nResults: {run_dir}\nReport:  {run_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

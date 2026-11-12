# Evaluation

`eval/` benchmarks this repo's two RAG pipelines on a fixed question set:

- **vector**: Chroma similarity search.
- **symbolic**: WSD entity linking with a Neo4j lookup.

It calls `api/services/*_service.query()` in-process, so it doesn't need the API server. It captures the answer and the retrieved chunk text, and scores them the same way RAG_Benchmark does. The HTML report has the same layout as the RAG_Benchmark reports.

## Setup

```bash
pip install -r requirements.txt -r eval/requirements.txt   # adds ragas (pins pyarrow<20 for numpy<2)
# Ingest first, either through the API or the scripts:
python create_database.py           # vector store (Chroma)
python symbolic_link_data.py        # WSD senses -> Neo4j (slow: about 2.3 s per chunk)
```

`.env` needs `OPENAI_API_KEY`, `SYMBOLIC_WSD_URL` and the `NEO4J_*` settings. The script loads `.env` with override, so a stale `OPENAI_API_KEY` in your shell can't take over.

## Run

```bash
python -m eval.run                                      # both modes, service defaults, all questions
python -m eval.run --top-k 5                            # equal budget: 5 chunks for both modes
python -m eval.run --modes vector --limit 10 --no-ragas # quick smoke run
python -m eval.run --ragas-sample 30 --title "Top-5 run" --notes findings.md
python -m eval.report eval/results/<run_id> [eval/results/<other_run>]   # rebuild a report
```

| Option | Default | What it does |
|---|---|---|
| `--dataset` | `data/eval/testset_stripped.csv` | `.csv` / `.jsonl` / `.json`. RAGAS testset columns are mapped automatically. |
| `--modes` | `vector symbolic` | Which pipelines to evaluate. |
| `--top-k` | service default | Chunks per question. The defaults are 3 for vector and 5 for symbolic (Personalized PageRank). |
| `--limit` | all | Use only the first N questions. |
| `--workers` | 1 | Questions in parallel. Symbolic calls the shared WSD service, so keep this low. |
| `--no-judge` / `--judge-model` / `--judge-metrics` | judge on, `gpt-4.1-mini` | LLM-as-judge settings. |
| `--no-ragas` / `--ragas-sample` / `--ragas-model` | RAGAS on all questions | RAGAS settings. The sample is seeded, so every mode is scored on the same questions. |
| `--group-by` | `query_style query_length persona_name` | Dataset columns to break the results down by. |
| `--out` | `eval/results` | Output folder (gitignored). |

## Output

`eval/results/<run_id>/` contains:
- `report.html`: scorecard, chart, per-document and per-question-type breakdowns, a question explorer, and a glossary of the metrics.
- `comparison.csv`: one row per mode.
- `<mode>/results.jsonl`: one row per question with the answer, reference, retrieved chunks, every score and the judge's reasons.
- `<mode>/summary.json` and `<mode>/config.json`.

## Metrics

| Group | Metric | Meaning |
|---|---|---|
| Retrieval, passage level | hit rate, MRR, nDCG, precision, recall | A chunk is relevant when it lies inside the question's gold passage (overlap coefficient ≥ 0.9). |
| Retrieval, document level | document hit rate, document MRR | A chunk is relevant when it comes from the expected source document. |
| Answer vs reference | answer rate, contains reference, token F1 | Deterministic checks. Unanswered questions score 0. |
| LLM judge | correctness, answer relevance, faithfulness, context relevance | Scored 1–5 and rescaled to 0–1. **Correctness ≥ 0.75 counts as "correct".** |
| RAGAS | answer correctness, faithfulness, context precision, context recall | RAGAS 0.4 metrics. Answer correctness is 75% factual F1 plus 25% semantic similarity. |
| Cost and speed | latency p50/p95, chunks per question, query synsets | End-to-end time per question, including the WSD call for symbolic. |

## Tests

`pytest tests/test_eval.py` runs offline against faked services, with no OpenAI, WSD or Neo4j calls.

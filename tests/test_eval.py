"""Offline tests for eval/: no OpenAI, WSD or Neo4j calls (services are faked)."""
import json

import pytest

from eval import run as eval_run
from eval.dataset import load_dataset, normalize_row, split_jsonl
from eval.metrics import (
    contains_reference,
    document_retrieval,
    overlap_coefficient,
    passage_retrieval,
    source_stem,
    token_f1,
)

GOLD = ("the transformer relies entirely on an attention mechanism to draw global dependencies "
        "between input and output and allows far more parallelization")


def test_ragas_csv_columns_are_mapped():
    item = normalize_row({"user_input": "Q?", "reference": "A", "source": "doc.pdf",
                          "reference_contexts": '["c1", "c2"]', "query_style": "SHORT"}, 0)
    assert (item.question, item.ground_truth, item.expected_sources) == ("Q?", "A", ["doc.pdf"])
    assert item.gold_context == "c1\n\nc2" and item.metadata == {"query_style": "SHORT"}


def test_jsonl_split_survives_pdf_control_chars():
    rows = [{"t": "a\x0cb c"}, {"t": "d"}]
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    assert [json.loads(x) for x in split_jsonl(text)] == rows


def test_answer_metrics():
    assert contains_reference("The answer is Insecure Storage.", "Insecure Storage") == 1.0
    assert contains_reference("Van Ripper", "Hans Van Ripper") == 0.0
    assert token_f1("v 3.0", "v 3.0") == 1.0


def test_retrieval_metrics():
    piece = "relies entirely on an attention mechanism to draw global dependencies"
    other = "convolutional networks stack many layers of filters over image pixels to learn"
    assert overlap_coefficient(piece, GOLD) == 1.0
    assert passage_retrieval([other, piece], GOLD)["mrr"] == 0.5
    assert source_stem("data/books/19_irving.pdf") == source_stem("19_irving.pdf (19_irving-3)")
    assert document_retrieval(["a.pdf (a-1)", "b.pdf (b-2)"], ["b.docx"])["mrr"] == 0.5


class _FakeService:
    def __init__(self, found_for):
        self.found_for = found_for

    def query(self, q, top_k=None, return_contexts=False):
        if self.found_for not in q:
            return {"found": False, "answer": None, "sources": [], "contexts": []}
        return {"found": True, "answer": "Insecure Storage", "sources": ["17_owasp_top10_2004.docx"],
                "contexts": ["A8 Insecure Storage is a common web application weakness " * 2][:(top_k or 3)]}


def test_full_run_offline(tmp_path, monkeypatch):
    items = load_dataset("data/eval/testset_stripped.csv", limit=4)
    target = items[0].question
    monkeypatch.setattr(eval_run, "get_service", lambda mode: _FakeService(target))
    rc = eval_run.main(["--limit", "4", "--modes", "vector", "symbolic", "--top-k", "2",
                        "--no-judge", "--no-ragas", "--out", str(tmp_path)])
    assert rc == 0
    run_dir = next(tmp_path.iterdir())
    for mode in ("vector", "symbolic"):
        s = json.loads((run_dir / f"langchain-rag_mode={mode}_" / "summary.json").read_text())["summary"]
        assert s["questions"] == 4 and s["errors"] == 0
        assert s["answer.answered"] == pytest.approx(0.25)
    assert (run_dir / "comparison.csv").exists()
    page = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "How to read the metrics" in page and "Document hit rate" in page

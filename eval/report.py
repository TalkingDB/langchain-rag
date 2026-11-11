"""Self-contained HTML report for one or more eval runs (``python -m eval.report``).

Ported from RAG_Benchmark's rag-bench report so both produce the same layout.

No external assets besides an optional Google Fonts link (with system fallbacks), so the file
can be emailed, attached, or published as-is. Charts are inline SVG; the question explorer is
a few lines of vanilla JS over embedded JSON.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.dataset import load_dataset, split_jsonl

# ------------------------------------------------------------------ metric catalogue
# (key, label, group, higher_is_better, is_unit_interval)
METRICS: list[tuple[str, str, str, bool, bool]] = [
    ("retrieval.hit_rate", "Hit rate", "Retrieval", True, True),
    ("retrieval.mrr", "MRR", "Retrieval", True, True),
    ("retrieval.ndcg", "nDCG", "Retrieval", True, True),
    ("retrieval.recall", "Recall", "Retrieval", True, True),
    ("retrieval.precision", "Precision", "Retrieval", True, True),
    ("retrieval.doc_hit_rate", "Document hit rate", "Retrieval", True, True),
    ("retrieval.doc_mrr", "Document MRR", "Retrieval", True, True),
    ("answer.answered", "Answer rate", "Answer vs reference", True, True),
    ("answer.contains_reference", "Contains reference", "Answer vs reference", True, True),
    ("answer.token_f1", "Token F1", "Answer vs reference", True, True),
    ("judge.correctness", "Correctness", "LLM judge", True, True),
    ("judge.faithfulness", "Faithfulness", "LLM judge", True, True),
    ("judge.context_relevance", "Context relevance", "LLM judge", True, True),
    ("judge.answer_relevance", "Answer relevance", "LLM judge", True, True),
    ("ragas.answer_correctness", "Answer correctness", "RAGAS", True, True),
    ("ragas.faithfulness", "Faithfulness", "RAGAS", True, True),
    ("ragas.context_precision", "Context precision", "RAGAS", True, True),
    ("ragas.context_recall", "Context recall", "RAGAS", True, True),
    ("ragas.answer_relevancy", "Answer relevancy", "RAGAS", True, True),
    ("ragas.factual_correctness", "Factual correctness", "RAGAS", True, True),
    ("context.sources_mean", "Sources returned (mean)", "Cost & speed", False, False),
    ("context.synsets_mean", "Query synsets (mean)", "Cost & speed", False, False),
    ("latency.total_p50_s", "Latency p50 (s)", "Cost & speed", False, False),
    ("latency.total_p95_s", "Latency p95 (s)", "Cost & speed", False, False),
    ("tokens.input_total", "Input tokens", "Cost & speed", False, False),
    ("tokens.output_total", "Output tokens", "Cost & speed", False, False),
]
KPI_LABELS = {
    "retrieval.hit_rate": "Hit rate",
    "retrieval.mrr": "MRR",
    "answer.answered": "Answer rate",
    "answer.contains_reference": "Contains reference",
    "judge.correctness": "Judge correctness",
    "judge.faithfulness": "Judge faithfulness",
    "ragas.answer_correctness": "RAGAS correctness",
    "latency.total_p50_s": "Latency p50 (s)",
}
HEADLINE = [
    "answer.answered",
    "retrieval.hit_rate",
    "retrieval.mrr",
    "answer.contains_reference",
    "judge.correctness",
    "judge.faithfulness",
    "ragas.answer_correctness",
    "latency.total_p50_s",
]

# Config keys shown in the "pipeline" section, in reading order.
API_SECTION = (
    "System under test",
    ["api.system", "api.base_url", "api.mode", "api.top_k", "api.endpoint", "api.concurrency"],
)
API_HIDDEN = {"Chunking", "Embedding & index", "Retrieval", "Generation"}

CONFIG_SECTIONS = [
    ("Data", ["data.eval_dataset", "data.corpus_dir", "data.max_questions"]),
    ("Chunking", ["chunking.splitter", "chunking.chunk_size", "chunking.chunk_overlap"]),
    ("Embedding & index", ["embedding.model", "embedding.dimensions", "vectorstore.distance"]),
    (
        "Retrieval",
        [
            "retrieval.strategy",
            "retrieval.top_k",
            "retrieval.mmr_lambda",
            "retrieval.hybrid.vector_weight",
            "retrieval.rerank.method",
            "retrieval.rerank.candidates",
            "retrieval.multi_query.num_queries",
        ],
    ),
    (
        "Generation",
        [
            "generation.model",
            "generation.temperature",
            "generation.max_tokens",
            "generation.prompt",
        ],
    ),
    (
        "Evaluation",
        [
            "evaluation.relevance",
            "evaluation.overlap_threshold",
            "evaluation.judge.model",
            "evaluation.ragas.enabled",
            "evaluation.ragas.model",
            "evaluation.ragas.sample_size",
        ],
    ),
]

# Categorical slots (validated reference palette, light / dark steps), fixed order.
SERIES = [
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
    ("#e34948", "#e66767"),
]

# Descriptive, not better/worse: never marked "best".
NEUTRAL = {"context.sources_mean", "context.synsets_mean"}

CORRECT_THRESHOLD = 0.75  # judge.correctness >= this counts as a correct answer


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def fmt(v: Any, unit_interval: bool = True) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.3f}" if unit_interval else (f"{v:,.0f}" if v >= 100 else f"{v:.2f}")
    return str(v)


# ------------------------------------------------------------------ loading
def load_experiments(run_dirs: list[Path]) -> list[dict[str, Any]]:
    exps = []
    for run_dir in run_dirs:
        for summary_path in sorted(run_dir.glob("*/summary.json")):
            exp_dir = summary_path.parent
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            s["config"] = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
            results = exp_dir / "results.jsonl"
            s["rows"] = (
                [json.loads(x) for x in split_jsonl(results.read_text(encoding="utf-8"))]
                if results.exists()
                else []
            )
            exps.append(s)
    if not exps:
        raise FileNotFoundError(f"No */summary.json found under {run_dirs}")
    return exps


def short_name(exp: dict[str, Any], varying: list[str]) -> str:
    """'team[strategy=hybrid]' -> 'hybrid'; falls back to the experiment name."""
    m = re.search(r"\[(.*)\]$", exp["experiment"])
    if m:
        parts = [p.split("=", 1)[-1] for p in m.group(1).split(",")]
        return " · ".join(parts)
    return exp["experiment"]


def varying_keys(exps: list[dict[str, Any]]) -> list[str]:
    keys = sorted({k for e in exps for k in e["config"]} - {"name", "description"})
    return [
        k for k in keys if len({json.dumps(e["config"].get(k), sort_keys=True) for e in exps}) > 1
    ]


def source_lookup(exps: list[dict[str, Any]]) -> dict[str, str]:
    """question id -> expected source document (from the run's dataset, if still on disk)."""
    path = Path(str(exps[0]["config"].get("data.eval_dataset", "")))
    if not path.exists():
        return {}
    try:
        return {
            i.id: (i.expected_sources[0] if i.expected_sources else "") for i in load_dataset(path)
        }
    except Exception:
        return {}


# ------------------------------------------------------------------ building blocks
def legend(names: list[str]) -> str:
    items = "".join(
        f'<span class="lg"><i class="sw s{i}"></i>{esc(n)}</span>' for i, n in enumerate(names)
    )
    return f'<div class="legend" aria-label="Legend">{items}</div>'


def quality_chart(exps: list[dict[str, Any]], names: list[str]) -> str:
    """Grouped horizontal bars, one row per 0-1 metric, one bar per experiment. One axis 0..1."""
    rows = [
        (k, lbl, grp)
        for k, lbl, grp, _, unit in METRICS
        if unit and any(e["summary"].get(k) is not None for e in exps)
    ]
    n = len(exps)
    bar_h, gap, row_pad = 9, 2, 14
    row_h = n * bar_h + (n - 1) * gap + row_pad
    label_w, plot_w, right = 168, 520, 12
    top = 26
    groups_seen: list[str] = []
    height = (
        top
        + len(rows) * row_h
        + sum(1 for _, _, g in rows if g not in groups_seen and not groups_seen.append(g)) * 22
    )
    width = label_w + plot_w + right
    x = lambda v: label_w + v * plot_w  # noqa: E731

    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height + 8}" role="img" '
        f'aria-label="Quality metrics per configuration, 0 to 1 scale">'
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{height}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x(t):.1f}" y="{top - 12}" text-anchor="middle">{t:g}</text>'
        )
    y = top
    current_group = None
    for key, lbl, grp in rows:
        if grp != current_group:
            current_group = grp
            parts.append(f'<text class="grp" x="0" y="{y + 14}">{esc(grp)}</text>')
            y += 22
        parts.append(
            f'<text class="rowlbl" x="{label_w - 10}" y="{y + (row_h - row_pad) / 2 + 4}"'
            f' text-anchor="end">{esc(lbl)}</text>'
        )
        for i, e in enumerate(exps):
            v = e["summary"].get(key)
            by = y + i * (bar_h + gap)
            if v is None:
                continue
            w = max(v * plot_w, 1.5)
            tip = f"{names[i]} · {grp} {lbl}: {v:.3f}"
            parts.append(
                f'<rect class="bar s{i}" x="{label_w}" y="{by}" width="{w:.1f}" '
                f'height="{bar_h}" rx="2" data-tip="{esc(tip)}"><title>{esc(tip)}'
                f"</title></rect>"
            )
        y += row_h
    parts.append(f'<line class="axis" x1="{label_w}" x2="{label_w}" y1="{top - 6}" y2="{height}"/>')
    parts.append("</svg>")
    return "".join(parts)


def scorecard(exps: list[dict[str, Any]], names: list[str]) -> str:
    head = "".join(
        f'<th scope="col"><i class="sw s{i}"></i>{esc(n)}</th>' for i, n in enumerate(names)
    )
    body, group = [], None
    for key, lbl, grp, higher, unit in METRICS:
        vals = [e["summary"].get(key) for e in exps]
        if all(v is None for v in vals):
            continue
        if grp != group:
            group = grp
            body.append(
                f'<tr class="grp-row"><th colspan="{len(exps) + 1}" scope="rowgroup">'
                f"{esc(grp)}</th></tr>"
            )
        present = [v for v in vals if v is not None]
        best = (
            (max(present) if higher else min(present))
            if len(present) > 1 and key not in NEUTRAL
            else None
        )
        cells = []
        for v in vals:
            is_best = best is not None and v == best and len(set(present)) > 1
            meter = (
                f'<span class="meter"><span style="width:{max(0, min(1, v)) * 100:.1f}%">'
                f"</span></span>"
                if unit and v is not None
                else ""
            )
            badge = '<span class="best" title="Best in row">best</span>' if is_best else ""
            cells.append(
                f'<td class="num{" is-best" if is_best else ""}">{meter}'
                f'<span class="v">{fmt(v, unit)}</span>{badge}</td>'
            )
        body.append(
            f'<tr><th scope="row">{esc(lbl)}<code>{esc(key)}</code></th>{"".join(cells)}</tr>'
        )
    return (
        f'<div class="scroll"><table class="score"><thead><tr><th scope="col">Metric</th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def heat_cell(v: float | None) -> str:
    if v is None:
        return '<td class="num heat">–</td>'
    pct = 6 + max(0.0, min(1.0, v)) * 50
    return f'<td class="num heat" style="--h:{pct:.0f}%">{v:.2f}</td>'


def doc_breakdown(exps: list[dict[str, Any]], names: list[str], src: dict[str, str]) -> str:
    if not src:
        return (
            '<p class="muted">Dataset file not found, so per-document results are unavailable.</p>'
        )
    ans_key = (
        "judge.correctness"
        if any("judge.correctness" in r for e in exps for r in e["rows"])
        else "answer.contains_reference"
    )
    ans_lbl = "correct" if ans_key == "judge.correctness" else "contains ref."
    docs = sorted({src.get(r["id"], "") for e in exps for r in e["rows"]} - {""})
    head1 = "".join(
        f'<th colspan="2" scope="colgroup"><i class="sw s{i}"></i>{esc(n)}</th>'
        for i, n in enumerate(names)
    )
    head2 = "".join(f'<th scope="col">hit rate</th><th scope="col">{ans_lbl}</th>' for _ in exps)
    body = []
    for d in docs:
        n_q = 0
        cells = []
        for e in exps:
            rs = [r for r in e["rows"] if src.get(r["id"]) == d and "error" not in r]
            n_q = max(n_q, len(rs))
            hits = [r["retrieval.hit_rate"] for r in rs if r.get("retrieval.hit_rate") is not None]
            ans = [r[ans_key] for r in rs if r.get(ans_key) is not None]
            cells.append(heat_cell(sum(hits) / len(hits) if hits else None))
            cells.append(heat_cell(sum(ans) / len(ans) if ans else None))
        pretty = re.sub(r"^\d+_", "", Path(d).stem).replace("_", " ")
        body.append(
            f'<tr><th scope="row"><span class="doc">{esc(pretty)}</span>'
            f'<code>{esc(d)}</code></th><td class="num">{n_q}</td>{"".join(cells)}</tr>'
        )
    return (
        f'<div class="scroll"><table class="grid-t"><thead><tr><th rowspan="2" scope="col">'
        f'Source document</th><th rowspan="2" scope="col">Qs</th>{head1}</tr>'
        f"<tr>{head2}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def group_breakdown(exps: list[dict[str, Any]], names: list[str]) -> str:
    fields = sorted({f for e in exps for f in (e.get("groups") or {})})
    if not fields:
        return ""
    keys = [
        k
        for k in ("retrieval.hit_rate", "judge.correctness", "answer.contains_reference")
        if any(k in v for e in exps for g in (e.get("groups") or {}).values() for v in g.values())
    ][:2]
    blocks = []
    for f in fields:
        values = sorted({v for e in exps for v in (e.get("groups") or {}).get(f, {})})
        head = "".join(
            f'<th scope="col"><i class="sw s{i}"></i>{esc(n)}<br>'
            f"<small>{esc(k.split('.', 1)[1].replace('_', ' '))}</small></th>"
            for i, n in enumerate(names)
            for k in keys
        )
        rows = []
        for val in values:
            n_q = max((e.get("groups") or {}).get(f, {}).get(val, {}).get("n", 0) for e in exps)
            cells = "".join(
                heat_cell((e.get("groups") or {}).get(f, {}).get(val, {}).get(k))
                for e in exps
                for k in keys
            )
            rows.append(
                f'<tr><th scope="row">{esc(val.replace("_", " ").title())}</th>'
                f'<td class="num">{n_q}</td>{cells}</tr>'
            )
        title = f.replace("_", " ").replace("name", "").strip().capitalize()
        blocks.append(
            f'<div class="gblock"><h3>By {esc(title.lower())}</h3>'
            f'<div class="scroll"><table class="grid-t"><thead><tr>'
            f'<th scope="col">{esc(title)}</th><th scope="col">Qs</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div></div>"
        )
    return f'<div class="gwrap">{"".join(blocks)}</div>'


def config_section(exps: list[dict[str, Any]], varying: list[str]) -> str:
    cfg = exps[0]["config"]
    blocks = []
    is_api = cfg.get("api.mode") is not None
    sections = (
        [API_SECTION] + [c for c in CONFIG_SECTIONS if c[0] not in API_HIDDEN]
        if is_api
        else CONFIG_SECTIONS
    )
    for title, keys in sections:
        items = []
        for k in keys:
            if k not in cfg:
                continue
            if k in varying:
                val = '<span class="varies">varies, see above</span>'
            else:
                val = f"<code>{esc(fmt(cfg[k], unit_interval=False))}</code>"
            items.append(f"<dt>{esc(k.split('.', 1)[1])}</dt><dd>{val}</dd>")
        if items:
            blocks.append(f'<div class="cfg"><h3>{esc(title)}</h3><dl>{"".join(items)}</dl></div>')
    return f'<div class="cfg-grid">{"".join(blocks)}</div>'


def _metric_help(is_api: bool) -> dict[str, str]:
    relevant = (
        "an item counts as relevant when it is the question's expected source document "
        "(the service's API returns source names, not chunk text)"
        if is_api
        else "a chunk counts as relevant only when it lies inside the question's gold passage "
        "(word-overlap coefficient of at least 0.9)"
    )
    return {
        "retrieval.hit_rate": f"Share of questions with at least one relevant retrieved item; {relevant}.",
        "retrieval.mrr": "Mean reciprocal rank: 1 if the first relevant item is ranked 1st, 0.5 if "
        "2nd, 0.33 if 3rd, 0 if none, averaged over questions.",
        "retrieval.ndcg": "Ranking quality: 1.0 when relevant items sit at the top of the results, "
        "lower when they are buried below irrelevant ones.",
        "retrieval.recall": "Share of the expected material that was retrieved. With one gold item "
        "per question it equals the hit rate.",
        "retrieval.doc_hit_rate": "Share of questions where at least one retrieved chunk comes from the "
        "expected source document (more lenient than the passage-level hit rate).",
        "retrieval.doc_mrr": "Like MRR, but a chunk counts when it comes from the expected document.",
        "retrieval.precision": "Share of retrieved items that are relevant. Low precision means much "
        "of the context sent to the LLM is off-topic.",
        "answer.answered": "Share of questions the system answered. Unanswered questions (for "
        "example, no chunk passed a relevance cutoff) score 0 on every answer metric.",
        "answer.contains_reference": "1 when the reference answer appears word for word inside the "
        "generated answer (ignoring case, punctuation and articles), otherwise 0. Strict about "
        "wording.",
        "answer.token_f1": "Word-overlap F1 between the answer and the reference. Long sentence-style "
        "answers score low even when correct.",
        "judge.correctness": "LLM judge: does the answer state the same facts as the reference? "
        "Scored 1–5 and rescaled to 0–1. A score of at least 0.75 counts as a correct answer.",
        "judge.faithfulness": "LLM judge: is every claim in the answer supported by the retrieved "
        "context?",
        "judge.context_relevance": "LLM judge: does the retrieved context contain the information "
        "needed to answer?",
        "judge.answer_relevance": "LLM judge: does the answer address the question directly and "
        "completely?",
        "ragas.answer_correctness": "RAGAS: 75% factual F1 between the claims in the answer and the "
        "reference, plus 25% semantic similarity. Extra claims are penalised, so long answers "
        "score lower than with the judge.",
        "ragas.faithfulness": "RAGAS: share of the answer's claims that the retrieved context supports.",
        "ragas.context_precision": "RAGAS: are the useful chunks ranked above the useless ones?",
        "ragas.context_recall": "RAGAS: share of the reference answer's claims found in the "
        "retrieved context.",
        "ragas.answer_relevancy": "RAGAS: how well questions generated back from the answer match "
        "the original question. Unreliable for one-word answers.",
        "ragas.factual_correctness": "RAGAS: claim-level F1 between the answer and the reference.",
        "context.sources_mean": "Average number of chunks used as context per question. LLM cost "
        "grows roughly with this number.",
        "context.synsets_mean": "Average number of word senses the WSD service linked in each "
        "question (symbolic only).",
        "latency.total_p50_s": "Median time per question, end to end: half of the questions finish "
        "faster than this.",
        "latency.total_p95_s": "95th-percentile time per question: 95% of questions finish faster.",
        "tokens.input_total": "Prompt tokens sent to the answer model across all questions.",
        "tokens.output_total": "Tokens generated by the answer model across all questions.",
    }


def metrics_glossary(exps: list[dict[str, Any]], is_api: bool) -> str:
    """'How to read the metrics': one entry per metric present in these runs."""
    help_text = _metric_help(is_api)
    groups: dict[str, list[str]] = {}
    for key, lbl, grp, _, _ in METRICS:
        if key in help_text and any(e["summary"].get(key) is not None for e in exps):
            groups.setdefault(grp, []).append(
                f"<dt>{esc(lbl)} <code>{esc(key)}</code></dt><dd>{esc(help_text[key])}</dd>"
            )
    blocks = "".join(
        f'<div class="cfg"><h3>{esc(g)}</h3><dl class="gloss">{"".join(items)}</dl></div>'
        for g, items in groups.items()
    )
    return (
        '<section><div class="sec-head"><h2>How to read the metrics</h2><p>Every score is '
        "0–1 and higher is better, except latency, where lower is better, and counts.</p></div>"
        f'<div class="panel"><div class="cfg-grid gloss-grid">{blocks}</div></div></section>'
    )


def explorer_data(exps: list[dict[str, Any]], src: dict[str, str]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for i, e in enumerate(exps):
        for r in e["rows"]:
            q = by_id.setdefault(
                r["id"],
                {
                    "id": r["id"],
                    "q": r.get("question", ""),
                    "ref": r.get("ground_truth") or "",
                    "doc": src.get(r["id"], ""),
                    "a": [None] * len(exps),
                },
            )
            corr = r.get("judge.correctness")
            ok = (
                corr >= CORRECT_THRESHOLD
                if corr is not None
                else (
                    r.get("answer.contains_reference") == 1.0
                    if r.get("answer.contains_reference") is not None
                    else None
                )
            )
            q["a"][i] = {
                "t": r.get("answer", r.get("error", "")),
                "ok": ok,
                "hit": r.get("retrieval.hit_rate"),
                "c": corr,
            }
    return list(by_id.values())


# ------------------------------------------------------------------ page
CSS = """
/* Layout: one reading column (max 1120px) of stacked sections; data tables scroll inside
   their own containers. Instrument-panel feel: cool neutrals, one teal accent. */
:root {
  --bg: #f4f6f6; --surface: #ffffff; --ink: #142023; --ink-2: #48575c; --muted: #6f7d82;
  --line: #dbe2e4; --line-2: #ebeff0; --accent: #0d6b70; --accent-soft: #e2f0f0;
  --good: #1f7a45; --bad: #b8392f; --seq: #2a78d6;
  --s0: #2a78d6; --s1: #eb6834; --s2: #1baf7a; --s3: #eda100; --s4: #e87ba4; --s5: #008300;
  --s6: #6250d6; --s7: #e34948;
  --f-display: "Schibsted Grotesk", "Segoe UI", system-ui, sans-serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1517; --surface: #152023; --ink: #e6eeef; --ink-2: #b2c0c3; --muted: #869599;
  --line: #2a383c; --line-2: #1d292c; --accent: #4fb3b8; --accent-soft: #173336;
  --good: #4cc27c; --bad: #ef7a6e; --seq: #3987e5;
  --s0: #3987e5; --s1: #d95926; --s2: #199e70; --s3: #c98500; --s4: #d55181; --s5: #008300;
  --s6: #9085e9; --s7: #e66767; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1517; --surface: #152023; --ink: #e6eeef; --ink-2: #b2c0c3; --muted: #869599;
  --line: #2a383c; --line-2: #1d292c; --accent: #4fb3b8; --accent-soft: #173336;
  --good: #4cc27c; --bad: #ef7a6e; --seq: #3987e5;
  --s0: #3987e5; --s1: #d95926; --s2: #199e70; --s3: #c98500; --s4: #d55181; --s5: #008300;
  --s6: #9085e9; --s7: #e66767; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font: 16px/1.55 var(--f-body); margin: 0; }
.wrap > * { min-width: 0; }
.wrap { max-width: 1120px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: flex; flex-direction: column; gap: 44px; }
h1, h2, h3 { font-family: var(--f-display); text-wrap: balance; margin: 0; letter-spacing: -0.01em; }
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); line-height: 1.1; font-weight: 700; }
h2 { font-size: 1.35rem; font-weight: 650; }
h3 { font-size: 1rem; font-weight: 650; }
p { margin: 0; max-width: 68ch; }
code { font-family: var(--f-mono); font-size: 0.8em; }
.muted, small { color: var(--muted); }
.eyebrow { font: 600 0.75rem/1 var(--f-mono); letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); }
header.top { display: flex; flex-direction: column; gap: 14px; border-bottom: 1px solid var(--line);
  padding-bottom: 28px; }
.lede { color: var(--ink-2); font-size: 1.08rem; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 22px; margin: 6px 0 0; padding: 0; list-style: none;
  font-size: 0.92rem; color: var(--ink-2); }
.facts b { color: var(--ink); font-variant-numeric: tabular-nums; font-weight: 650; }
section { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.sec-head { display: flex; flex-direction: column; gap: 6px; }
.sec-head p { color: var(--ink-2); font-size: 0.95rem; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 18px; min-width: 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px;
  padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.card h3 { display: flex; align-items: center; gap: 8px; }
.card .chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: 0.78rem/1.6 var(--f-mono); background: var(--accent-soft); color: var(--ink);
  border-radius: 4px; padding: 0 7px; }
.kpis { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 14px; margin: 0; }
.kpis div { display: flex; flex-direction: column; }
.kpis dt { font-size: 0.78rem; color: var(--muted); }
.kpis dd { margin: 0; font: 600 1.25rem/1.2 var(--f-display); font-variant-numeric: tabular-nums; }
.kpis dd.lead { color: var(--accent); }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; flex: none;
  vertical-align: baseline; margin-right: 6px; }
.s0 { --c: var(--s0); } .s1 { --c: var(--s1); } .s2 { --c: var(--s2); } .s3 { --c: var(--s3); }
.s4 { --c: var(--s4); } .s5 { --c: var(--s5); } .s6 { --c: var(--s6); } .s7 { --c: var(--s7); }
.sw, .bar { background: var(--c); fill: var(--c); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.9rem; color: var(--ink-2); }
.lg { display: inline-flex; align-items: center; }
.scroll { overflow-x: auto; min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line-2); text-align: left; vertical-align: top; }
thead th { font: 600 0.78rem/1.3 var(--f-body); color: var(--ink-2); border-bottom: 1px solid var(--line);
  white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--f-mono);
  font-size: 0.85rem; white-space: nowrap; }
.score tbody th { font-weight: 500; }
.score tbody th code, .grid-t tbody th code { display: block; color: var(--muted); font-size: 0.72rem; }
.grp-row th { font: 600 0.72rem/1 var(--f-mono); letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--accent); padding-top: 18px; border-bottom: 1px solid var(--line); }
.meter { display: inline-block; width: 64px; height: 6px; border-radius: 3px; background: var(--line-2);
  vertical-align: middle; margin-right: 10px; overflow: hidden; }
.meter span { display: block; height: 100%; background: var(--muted); border-radius: 3px; }
.is-best .meter span { background: var(--accent); }
.is-best .v { color: var(--accent); font-weight: 700; }
.best { font: 600 0.62rem/1 var(--f-mono); text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--accent); border: 1px solid currentColor; border-radius: 3px; padding: 2px 4px;
  margin-left: 6px; vertical-align: 1px; }
.heat { background: color-mix(in oklab, var(--seq) var(--h, 0%), var(--surface)); }
.doc { text-transform: capitalize; }
.chart { width: 100%; height: auto; display: block; min-width: 560px; }
.chart text { fill: var(--ink-2); font: 12px var(--f-body); }
.chart .tick { fill: var(--muted); font: 11px var(--f-mono); }
.chart .grp { fill: var(--accent); font: 600 11px var(--f-mono); letter-spacing: 0.06em;
  text-transform: uppercase; }
.chart .grid { stroke: var(--line-2); stroke-width: 1; }
.chart .axis { stroke: var(--line); stroke-width: 1; }
.chart .bar:hover { opacity: 0.8; }
.tip { position: fixed; pointer-events: none; z-index: 20; background: var(--ink); color: var(--bg);
  font: 12px/1.4 var(--f-body); padding: 6px 9px; border-radius: 5px; max-width: 280px; }
.gwrap { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap: 14px; }
.gblock { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px;
  display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.cfg-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 14px; }
.cfg { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.cfg dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 0.88rem; }
.cfg dt { color: var(--muted); }
.cfg dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.gloss-grid { grid-template-columns: repeat(auto-fit, minmax(min(300px, 100%), 1fr)); }
.cfg dl.gloss { grid-template-columns: 1fr; gap: 2px; }
.cfg dl.gloss dt { color: var(--ink); font-weight: 600; margin-top: 8px; }
.cfg dl.gloss dt code { color: var(--muted); font-weight: 400; margin-left: 4px; }
.cfg dl.gloss dd { color: var(--ink-2); }
.varies { color: var(--accent); font-weight: 600; font-size: 0.85rem; }
.notes { display: flex; flex-direction: column; gap: 10px; border-left: 3px solid var(--accent);
  padding: 4px 0 4px 18px; }
.notes ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; max-width: 72ch; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.controls input { font: inherit; padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--surface); color: var(--ink); min-width: 0; flex: 1 1 220px; max-width: 360px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.seg button { font: 500 0.85rem var(--f-body); background: var(--surface); color: var(--ink-2);
  border: 0; padding: 7px 12px; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-pressed="true"] { background: var(--accent-soft); color: var(--ink); font-weight: 650; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.qtable td.q { min-width: 240px; }
.qtable td.ans { min-width: 170px; }
.qtable .qid { font: 0.72rem var(--f-mono); color: var(--muted); display: block; }
.qtable .ref { display: block; color: var(--ink-2); font-size: 0.85rem; margin-top: 3px; }
.st { font: 600 0.7rem/1 var(--f-mono); display: inline-flex; align-items: center; gap: 4px;
  margin-bottom: 3px; }
.st.ok { color: var(--good); } .st.no { color: var(--bad); } .st.na { color: var(--muted); }
.miss { font: 0.7rem var(--f-mono); color: var(--muted); }
.count { font-size: 0.88rem; color: var(--muted); font-variant-numeric: tabular-nums; }
footer { border-top: 1px solid var(--line); padding-top: 18px; color: var(--muted); font-size: 0.85rem;
  display: flex; flex-direction: column; gap: 6px; }
@media (max-width: 560px) { .wrap { padding-block: 24px 48px; gap: 34px; } .kpis dd { font-size: 1.1rem; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = """
(function () {
  var tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true;
  document.body.appendChild(tip);
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    var t = el.querySelector('title'); if (t) t.remove();
    el.addEventListener('mousemove', function (e) {
      tip.textContent = el.getAttribute('data-tip'); tip.hidden = false;
      var x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      tip.style.left = x + 'px'; tip.style.top = (e.clientY + 14) + 'px';
    });
    el.addEventListener('mouseleave', function () { tip.hidden = true; });
  });

  var data = JSON.parse(document.getElementById('qdata').textContent);
  var names = JSON.parse(document.getElementById('qnames').textContent);
  var body = document.getElementById('qbody'), count = document.getElementById('qcount');
  var search = document.getElementById('qsearch'), mode = 'fail';
  function esc(s) { var d = document.createElement('div'); d.textContent = s == null ? '' : s; return d.innerHTML; }
  function status(a) {
    if (!a) return '<span class="st na">not run</span>';
    if (a.ok === true) return '<span class="st ok">\\u2713 correct</span>';
    if (a.ok === false) return '<span class="st no">\\u2717 wrong</span>';
    return '<span class="st na">\\u2013 unscored</span>';
  }
  function keep(r) {
    var oks = r.a.map(function (a) { return a ? a.ok : null; });
    if (mode === 'fail' && oks.indexOf(false) < 0) return false;
    if (mode === 'diff' && !(oks.indexOf(true) >= 0 && oks.indexOf(false) >= 0)) return false;
    var q = search.value.trim().toLowerCase();
    return !q || (r.q + ' ' + r.ref + ' ' + r.doc + ' ' + r.id).toLowerCase().indexOf(q) >= 0;
  }
  function render() {
    var rows = data.filter(keep), out = [];
    rows.forEach(function (r) {
      var cells = r.a.map(function (a) {
        if (!a) return '<td class="ans">' + status(a) + '</td>';
        var miss = a.hit === 0 ? '<span class="miss">__MISS_LABEL__</span>' : '';
        return '<td class="ans">' + status(a) + '<div>' + esc(a.t) + '</div>' + miss + '</td>';
      }).join('');
      out.push('<tr><td class="q"><span class="qid">' + esc(r.id) + (r.doc ? ' \\u00b7 ' + esc(r.doc) : '') +
        '</span>' + esc(r.q) + '<span class="ref">Reference: ' + esc(r.ref) + '</span></td>' + cells + '</tr>');
    });
    body.innerHTML = out.join('') || '<tr><td colspan="' + (names.length + 1) + '" class="muted">No questions match.</td></tr>';
    count.textContent = rows.length + ' of ' + data.length + ' questions';
  }
  document.querySelectorAll('.seg button').forEach(function (b) {
    b.addEventListener('click', function () {
      mode = b.getAttribute('data-mode');
      document.querySelectorAll('.seg button').forEach(function (x) {
        x.setAttribute('aria-pressed', String(x === b)); });
      render();
    });
  });
  search.addEventListener('input', render);
  render();
})();
"""


def _notes_html(text: str) -> str:
    """Tiny markdown subset: paragraphs, '- ' bullets, **bold**, `code`."""

    def inline(s: str) -> str:
        s = esc(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    out, bullets = [], []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines and all(ln.startswith(("- ", "* ")) for ln in lines):
            bullets = "".join(f"<li>{inline(ln[2:])}</li>" for ln in lines)
            out.append(f"<ul>{bullets}</ul>")
        elif lines and lines[0].startswith("#"):
            out.append(f"<h3>{inline(lines[0].lstrip('# '))}</h3>")
            if lines[1:]:
                out.append(f"<p>{inline(' '.join(lines[1:]))}</p>")
        else:
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    return "".join(out)


def build_report(
    run_dirs: list[Path],
    title: str | None = None,
    notes: str | None = None,
    standalone: bool = True,
) -> str:
    exps = load_experiments(run_dirs)
    varying = varying_keys(exps)
    names = [short_name(e, varying) for e in exps]
    src = source_lookup(exps)
    cfg = exps[0]["config"]
    run_ids = sorted({e.get("run_id", "") for e in exps})
    n_q = max(e["summary"].get("questions", 0) for e in exps)
    n_docs = len(set(src.values())) if src else None
    ragas_n = max((e["summary"].get("ragas.sample_size") or 0) for e in exps)
    try:
        when = datetime.strptime(run_ids[-1], "%Y%m%d-%H%M%S").strftime("%d %b %Y, %H:%M")
    except ValueError:
        when = run_ids[-1]
    heading = title or (
        "RAG pipeline benchmark"
        if len(exps) == 1
        else "Comparing " + " vs ".join(names)
        if len(exps) <= 3
        else f"{len(exps)} RAG configurations compared"
    )
    page_title = title or f"RAG Benchmark {when.split(',')[0]}"

    # ---- configuration cards: what differs + a few headline numbers
    cards = []
    for i, (e, n) in enumerate(zip(exps, names, strict=True)):
        chips = (
            "".join(
                f'<span class="chip">{esc(k.split(".")[-1])}={esc(e["config"].get(k))}</span>'
                for k in varying
            )
            or '<span class="chip">single run</span>'
        )
        kpis = []
        for k in HEADLINE:
            v = e["summary"].get(k)
            if v is None:
                continue
            meta = next(m for m in METRICS if m[0] == k)
            vals = [x["summary"].get(k) for x in exps if x["summary"].get(k) is not None]
            lead = (
                len(exps) > 1 and len(set(vals)) > 1 and v == (max(vals) if meta[3] else min(vals))
            )
            kpis.append(
                f"<div><dt>{esc(KPI_LABELS.get(k, meta[1]))}</dt>"
                f'<dd class="{"lead" if lead else ""}">'
                f"{fmt(v, meta[4])}</dd></div>"
            )
        cards.append(
            f'<article class="card"><h3><i class="sw s{i}"></i>{esc(n)}</h3>'
            f'<div class="chips">{chips}</div><dl class="kpis">{"".join(kpis[:6])}</dl>'
            f"</article>"
        )

    facts = [
        f"<li><b>{n_q}</b> questions</li>",
        f"<li><b>{len(exps)}</b> configuration{'s' if len(exps) > 1 else ''}</li>",
    ]
    if n_docs:
        facts.append(f"<li><b>{n_docs}</b> source documents</li>")
    is_api = cfg.get("api.mode") is not None
    if is_api:
        facts.append(
            f"<li>System under test <b>{esc(cfg.get('api.system'))}</b> API at "
            f"<b>{esc(cfg.get('api.base_url'))}</b></li>"
        )
    else:
        facts.append(f"<li>Answers by <b>{esc(cfg.get('generation.model'))}</b></li>")
        facts.append(f"<li>Embeddings <b>{esc(cfg.get('embedding.model'))}</b></li>")
    facts.append(f"<li>Judge <b>{esc(cfg.get('evaluation.judge.model'))}</b></li>")
    if ragas_n:
        facts.append(f"<li>RAGAS on <b>{ragas_n}</b> questions</li>")

    relevance = cfg.get("evaluation.relevance")
    if is_api:
        retrieval_note = (
            "Retrieval is scored at document level over every source the service returned, "
            "because its API returns source names but not chunk text. A hit means the expected "
            "document is among them. Questions the service could not answer score 0 on all "
            "answer metrics"
        )
    else:
        rel_note = (
            "a chunk counts as a hit only if it comes from the question's gold passage"
            if relevance == "overlap"
            else "a chunk counts as a hit if it comes from the expected source document"
        )
        retrieval_note = (
            f"Retrieval metrics are computed at top-{esc(cfg.get('retrieval.top_k'))}; {rel_note}"
        )

    sections = [
        f'<header class="top"><span class="eyebrow">rag-bench · run {esc(", ".join(run_ids))}'
        f'</span><h1>{esc(heading)}</h1><p class="lede">Retrieval-augmented generation over '
        f"{f'{n_docs} source documents' if n_docs else 'the corpus'}, scored on retrieval, answer "
        f"accuracy, an LLM judge{' and RAGAS' if ragas_n else ''}. Generated {esc(when)}.</p>"
        f'<ul class="facts">{"".join(facts)}</ul></header>',
    ]
    if notes:
        sections.append(
            f'<section><div class="sec-head"><h2>Findings</h2></div>'
            f'<div class="notes">{_notes_html(notes)}</div></section>'
        )
    sections.append(
        f'<section><div class="sec-head"><h2>Configurations</h2><p>What differs '
        f"between the runs, with headline scores. The best value per metric is "
        f'highlighted.</p></div><div class="cards">{"".join(cards)}</div></section>'
    )
    sections.append(
        f'<section><div class="sec-head"><h2>Quality at a glance</h2><p>All 0–1 metrics on one '
        f"scale; higher is better. Hover a bar for its value. Exact numbers are in the scorecard "
        f'below.</p></div><div class="panel">{legend(names) if len(exps) > 1 else ""}'
        f'<div class="scroll">{quality_chart(exps, names)}</div></div></section>'
    )
    sections.append(
        f'<section><div class="sec-head"><h2>Scorecard</h2><p>{retrieval_note}. Judge scores '
        f"are rescaled from "
        f'1–5 to 0–1.</p></div><div class="panel">{scorecard(exps, names)}</div></section>'
    )
    sections.append(
        f'<section><div class="sec-head"><h2>By source document</h2><p>Where the pipeline '
        f'struggles. Darker cells are higher scores.</p></div><div class="panel">'
        f"{doc_breakdown(exps, names, src)}</div></section>"
    )
    gb = group_breakdown(exps, names)
    if gb:
        sections.append(
            f'<section><div class="sec-head"><h2>By question type</h2><p>Breakdowns '
            f"by the dataset's own labels.</p></div>{gb}</section>"
        )
    qhead = "".join(
        f'<th scope="col"><i class="sw s{i}"></i>{esc(n)}</th>' for i, n in enumerate(names)
    )
    filters = (
        '<button type="button" data-mode="fail" aria-pressed="true">Has a wrong answer</button>'
        + (
            '<button type="button" data-mode="diff" aria-pressed="false">Configs disagree</button>'
            if len(exps) > 1
            else ""
        )
        + '<button type="button" data-mode="all" aria-pressed="false">All</button>'
    )
    sections.append(
        f'<section><div class="sec-head"><h2>Question explorer</h2><p>Every question with each '
        f"configuration's answer. An answer counts as correct when the judge's correctness score is "
        f'at least {CORRECT_THRESHOLD}.</p></div><div class="panel" style="display:flex;'
        f'flex-direction:column;gap:12px"><div class="controls"><input id="qsearch" type="search" '
        f'placeholder="Search questions, answers, documents" aria-label="Search questions">'
        f'<div class="seg" role="group" aria-label="Filter">{filters}</div>'
        f'<span class="count" id="qcount"></span></div><div class="scroll"><table class="qtable">'
        f'<thead><tr><th scope="col">Question</th>{qhead}</tr></thead><tbody id="qbody"></tbody>'
        f"</table></div></div></section>"
    )
    sections.append(metrics_glossary(exps, is_api))
    sections.append(
        f'<section><div class="sec-head"><h2>Pipeline settings</h2><p>Shared by all '
        f'configurations unless marked.</p></div><div class="panel">'
        f"{config_section(exps, varying)}</div></section>"
    )
    sections.append(
        f"<footer><span>Generated by <code>python -m eval.report</code> from "
        f"{esc(', '.join(str(d) for d in run_dirs))}.</span></footer>"
    )

    miss_label = "expected document not in sources" if is_api else "gold passage not retrieved"
    qdata = json.dumps(explorer_data(exps, src), ensure_ascii=False).replace("</", "<\\/")
    head = (
        f"<title>{esc(page_title)}</title>"
        '<meta name="claude:viewport-fit-sides" content="cover">'
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:'
        "wght@400;600&family=Schibsted+Grotesk:wght@600;700&family=Source+Sans+3:wght@400;"
        '500;600;700&display=swap">'
        f"<style>{CSS}</style>"
    )
    body = (
        f'<main class="wrap">{"".join(sections)}</main>'
        f'<script type="application/json" id="qdata">{qdata}</script>'
        f'<script type="application/json" id="qnames">{json.dumps(names)}</script>'
        f"<script>{JS.replace('__MISS_LABEL__', miss_label)}</script>"
    )
    if not standalone:  # page fragment for hosts that supply their own document skeleton
        return head + body
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width, initial-scale=1, viewport-fit=cover">{head}</head>'
        f"<body>{body}</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    """python -m eval.report eval/results/<run_id> [more run dirs] [--title T] [--notes N.md]"""
    import argparse

    ap = argparse.ArgumentParser(description="Build an HTML report from eval run folders.")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--notes", type=Path, default=None, help="markdown findings shown at the top")
    ap.add_argument("--out", type=Path, default=None, help="default: <first run dir>/report.html")
    args = ap.parse_args(argv)
    page = build_report(args.run_dirs, title=args.title,
                        notes=args.notes.read_text(encoding="utf-8") if args.notes else None)
    out = args.out or args.run_dirs[0] / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"Report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

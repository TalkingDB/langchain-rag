"""Render benchmarks/benchmark_retrieval.py JSON results as a self-contained HTML report.

    python benchmarks/report_retrieval.py benchmarks/results_leiden_first20.json \
        --title "Leiden Retrieval Benchmark" --notes benchmarks/notes_leiden.md

Sections: verdict tiles, aggregate metrics per method, execution-time breakdown,
per-question results (question, reference, answer, verdict, gold rank, timings),
observations (from --notes), and run settings. Styling is shared with eval/report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.report import CSS, _notes_html, esc  # noqa: E402

SYMBOLIC = {"exact_all", "exact_top5", "leiden_top5", "lpa_top5"}

METHOD_LABELS = {
    "vector_top5": "Vector (top 5)",
    "exact_all": "Exact match (all chunks)",
    "exact_top5": "Exact match (top 5)",
    "leiden_top5": "Leiden (top 5)",
    "lpa_top5": "Label propagation (top 5)",
}

EXTRA_CSS = """
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(170px, 100%), 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.tile .k { font-size: 0.8rem; color: var(--muted); }
.tile .v { font: 650 1.6rem/1.15 var(--f-display); font-variant-numeric: tabular-nums; }
.tl { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
.tm { font-size: 0.8rem; color: var(--ink-2); }
.tv { font: 650 1.15rem/1.3 var(--f-display); font-variant-numeric: tabular-nums; text-align: right; }
.tv small { font: 400 0.75rem var(--f-body); color: var(--muted); }
.pills { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.tile .s { font-size: 0.8rem; color: var(--ink-2); }
.pill { font: 600 0.72rem/1 var(--f-mono); padding: 3px 6px; border-radius: 4px; white-space: nowrap;
  display: inline-flex; gap: 4px; align-items: center; }
.pill.ok { color: var(--good); border: 1px solid currentColor; }
.pill.no { color: var(--bad); border: 1px solid currentColor; }
.pill.na { color: var(--muted); border: 1px solid currentColor; }
.qcell { min-width: 220px; }
.acell { min-width: 260px; font-size: 0.88rem; }
.small { font-size: 0.8rem; color: var(--muted); display: block; margin-top: 2px; }
.callout { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; display: flex; flex-direction: column; gap: 6px; }
"""


def fmt_ms(v: float) -> str:
    return f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms"


def pill(ok: bool | None, yes: str, no: str) -> str:
    if ok is None:
        return '<span class="pill na">–</span>'
    return f'<span class="pill {"ok" if ok else "no"}">{"✓" if ok else "✗"} {yes if ok else no}</span>'


def load_merged(paths: list[Path]) -> tuple[list[dict], dict, dict]:
    """Merge result files that cover the same questions (matched by sheet row) into one view."""
    merged: dict[int, dict] = {}
    summary: dict = {}
    args: dict = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary.update(data["summary"])
        args = args or data.get("args", {})
        for i, r in enumerate(data["results"], 1):
            key = r.get("row", i)
            entry = merged.setdefault(key, {k: v for k, v in r.items() if k != "methods"} | {"methods": {}})
            if r.get("wsd_ms"):
                entry["wsd_ms"] = r["wsd_ms"]
            entry["methods"].update(r["methods"])
    return [merged[k] for k in sorted(merged)], summary, args


def build(paths: list[Path], title: str | None, notes: str | None, reingest: str | None) -> str:
    results, summary, args = load_merged(paths)
    methods = list(summary)
    n = len(results)
    rows_desc = (f"rows {results[0].get('row', 1)}-{results[-1].get('row', n)} of the sheet"
                 if args.get("first") else f"random sample (seed {args.get('seed')})")
    try:
        when = datetime.fromtimestamp(max(p.stat().st_mtime for p in paths)).strftime("%d %b %Y, %H:%M")
    except OSError:
        when = ""
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    def best(key: str, higher: bool = True) -> set:
        vals = {m: summary[m].get(key) for m in methods if summary[m].get(key) is not None}
        if len(set(vals.values())) < 2:
            return set()
        target = max(vals.values()) if higher else min(vals.values())
        return {m for m, v in vals.items() if v == target}

    def n_correct(m: str) -> str:
        return f"{sum(r['methods'][m]['correct'] for r in results if m in r['methods'])}/{n}"

    # ---- comparison tiles: one tile per metric, one line per method
    tile_specs = [
        ("Answer accuracy", "accuracy", "{:.0%}", True, n_correct),
        ("Retrieval hit@5", "hit_rate", "{:.0%}", True, None),
        ("MRR", "mrr", "{:.2f}", True, None),
        ("Answer in context", "ans_in_ctx", "{:.0%}", True, None),
        ("Median retrieval", "median_retrieval_ms", None, False, None),
        ("Median end-to-end", "median_pipeline_ms", None, False, None),
    ]
    tiles_html = ""
    for label, key, f, higher, extra in tile_specs:
        winners = best(key, higher)
        lines = ""
        for m, lab in zip(methods, labels):
            v = summary[m].get(key)
            txt = "-" if v is None else (fmt_ms(v) if f is None else f.format(v))
            if extra:
                txt += f" <small>({extra(m)})</small>"
            mark = ' <span class="pill ok">best</span>' if m in winners else ""
            lines += f'<div class="tl"><span class="tm">{esc(lab)}</span><span class="tv">{txt}{mark}</span></div>'
        tiles_html += f'<div class="tile"><span class="k">{esc(label)}</span>{lines}</div>'

    # ---- aggregate table
    agg_rows = [
        ("Answer accuracy (LLM judge)", "accuracy", "{:.0%}", True), ("Retrieval hit@5", "hit_rate", "{:.0%}", True),
        ("MRR", "mrr", "{:.2f}", True), ("Answer in retrieved context", "ans_in_ctx", "{:.0%}", True),
        ("Chunks sent to the LLM (mean)", "avg_chunks", "{:.1f}", None),
        ("Context characters (mean)", "avg_context_chars", "{:,.0f}", None),
        ("Retrieval time, median", "median_retrieval_ms", None, False),
        ("Retrieval time, mean", "mean_retrieval_ms", None, False),
        ("Retrieval time, max", "max_retrieval_ms", None, False),
        ("End-to-end time, median", "median_pipeline_ms", None, False),
    ]
    head = "".join(f'<th scope="col" class="num">{esc(lab)}</th>' for lab in labels)
    body = ""
    for label, key, f, higher in agg_rows:
        winners = best(key, higher) if higher is not None else set()
        cells = ""
        for m in methods:
            v = summary[m].get(key)
            txt = "-" if v is None else (fmt_ms(v) if f is None else f.format(v))
            cls = ' style="color:var(--accent);font-weight:700"' if m in winners else ""
            cells += f'<td class="num"{cls}>{txt}</td>'
        body += f'<tr><th scope="row">{esc(label)}</th>{cells}</tr>'
    agg_html = (f'<div class="scroll"><table><thead><tr><th scope="col">Metric</th>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>")

    # ---- execution time by stage (median and max per method)
    def stage_vals(m: str, stage: str) -> list:
        if stage == "wsd":
            return [r.get("wsd_ms", 0) for r in results] if m in SYMBOLIC else []
        return [r["methods"][m].get(stage, 0) for r in results if m in r["methods"]]

    stages = [("Question WSD linking (symbolic only)", "wsd"),
              ("Retrieval", "retrieval_ms"), ("Answer generation (gpt-4.1-mini)", "answer_ms"),
              ("End-to-end per question", "pipeline_ms")]
    t_head = "".join(f'<th class="num">{esc(lab)}<br><small>median / max</small></th>' for lab in labels)
    t_body = ""
    for label, stage in stages:
        cells = ""
        for m in methods:
            v = stage_vals(m, stage)
            cells += (f'<td class="num">{fmt_ms(st.median(v))} / {fmt_ms(max(v))}</td>' if v and any(v)
                      else '<td class="num">-</td>')
        t_body += f'<tr><th scope="row">{esc(label)}</th>{cells}</tr>'
    timing_html = (f'<div class="scroll"><table><thead><tr><th scope="col">Stage</th>{t_head}</tr></thead>'
                   f"<tbody>{t_body}</tbody></table></div>")

    # ---- per-question comparison
    q_head = "".join(f"<th>{esc(lab)}</th>" for lab in labels)
    q_body = ""
    for i, r in enumerate(results, 1):
        cells = ""
        for m in methods:
            x = r["methods"].get(m)
            if not x:
                cells += "<td>-</td>"
                continue
            rank = x["first_gold_rank"]
            cells += (
                f'<td class="acell"><div class="pills">{pill(x["correct"], "correct", "wrong")}'
                f'{pill(x["hit"], f"gold rank {rank}" if rank else "hit", "gold missed")}'
                f'{pill(x["ans_in_ctx"], "answer in ctx", "answer not in ctx")}</div>'
                f'{esc(x["answer"])}<span class="small">{x["n_chunks"]} chunks / retrieval '
                f'{fmt_ms(x["retrieval_ms"])} / end-to-end {fmt_ms(x.get("pipeline_ms", 0))}</span></td>'
            )
        q_body += (f'<tr><td class="num">{r.get("row", i)}</td>'
                   f'<td class="qcell">{esc(r["question"])}<span class="small">{esc(r["source"])}</span>'
                   f'<span class="small"><b>Reference:</b> {esc(r["reference"])}</span></td>{cells}</tr>')
    q_html = (f'<div class="scroll"><table><thead><tr><th class="num">#</th><th>Question</th>{q_head}</tr>'
              f"</thead><tbody>{q_body}</tbody></table></div>")

    gloss = (
        '<dl class="gloss" style="display:grid;gap:4px 14px;grid-template-columns:minmax(0,max-content) 1fr">'
        "<dt><b>Answer accuracy</b></dt><dd>An LLM judge (gpt-4.1-mini) decides whether the generated answer conveys the same fact as the reference.</dd>"
        "<dt><b>Retrieval hit@5</b></dt><dd>A retrieved chunk counts as gold when at least 50% of its word 4-grams fall inside the question's reference passage. Hit = at least one gold chunk among those retrieved.</dd>"
        "<dt><b>MRR</b></dt><dd>1 / rank of the first gold chunk (1.0 = ranked first, 0 = none).</dd>"
        "<dt><b>Answer in context</b></dt><dd>The normalised reference answer appears word for word inside the retrieved chunks.</dd>"
        "<dt><b>Execution time</b></dt><dd>Wall-clock per question. Symbolic = WSD linking + retrieval (Neo4j subgraph + ranking) + answer. Vector = retrieval (query embedding + Chroma search) + answer.</dd></dl>")

    heading = title or (" vs ".join(labels) if len(methods) > 1 else "Retrieval Benchmark")
    sections = [
        f'<header class="top"><span class="eyebrow">langchain-rag / benchmarks/benchmark_retrieval.py</span>'
        f'<h1>{esc(heading)}</h1><p class="lede">{n} questions ({esc(rows_desc)}), same chunk budget '
        f"(top 5), same answer prompt, model and judge for every method. Generated {esc(when)}.</p></header>",
        f'<section><div class="tiles">{tiles_html}</div></section>',
    ]
    if reingest:
        sections.append(f'<section><div class="sec-head"><h2>Do we need to re-ingest?</h2></div>'
                        f'<div class="callout">{_notes_html(reingest)}</div></section>')
    if notes:
        sections.append(f'<section><div class="sec-head"><h2>Observations</h2></div>'
                        f'<div class="notes">{_notes_html(notes)}</div></section>')
    sections += [
        f'<section><div class="sec-head"><h2>Aggregate metrics</h2><p>The better value in each row is highlighted.</p></div>'
        f'<div class="panel">{agg_html}</div></section>',
        f'<section><div class="sec-head"><h2>Execution time</h2><p>Per question, by pipeline stage.</p></div>'
        f'<div class="panel">{timing_html}</div></section>',
        f'<section><div class="sec-head"><h2>Per-question results</h2></div><div class="panel">{q_html}</div></section>',
        f'<section><div class="sec-head"><h2>How the metrics are computed</h2></div><div class="panel">{gloss}</div></section>',
        f'<footer><span>Sources: {esc(", ".join(str(p).replace(chr(92), "/") for p in paths))}</span></footer>',
    ]
    page_title = title or "Retrieval Benchmark"
    head = (f"<title>{esc(page_title)}</title>"
            '<meta name="claude:viewport-fit-sides" content="cover">'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600'
            '&family=Schibsted+Grotesk:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap">'
            f"<style>{CSS}{EXTRA_CSS}</style>")
    return head + f'<main class="wrap">{"".join(sections)}</main>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", type=Path, nargs="+", help="one or more result JSON files")
    ap.add_argument("--title", default=None)
    ap.add_argument("--notes", type=Path, default=None, help="markdown observations")
    ap.add_argument("--reingest", type=Path, default=None, help="markdown re-ingestion note")
    ap.add_argument("--out", type=Path, default=None, help="default: <results>.html")
    ap.add_argument("--fragment", action="store_true", help="omit the <html>/<head> wrapper")
    args = ap.parse_args()
    page = build(args.results, args.title,
                 args.notes.read_text(encoding="utf-8") if args.notes else None,
                 args.reingest.read_text(encoding="utf-8") if args.reingest else None)
    if not args.fragment:
        page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                'content="width=device-width, initial-scale=1, viewport-fit=cover">'
                + page.replace("<main", "</head><body><main", 1) + "</body></html>")
    out = args.out or args.results[0].with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    print(f"Report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

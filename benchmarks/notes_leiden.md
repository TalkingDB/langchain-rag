- **Retrieval is accurate.** The gold passage is in the top 5 for 19 of 20 questions (hit@5 0.95), ranked first for 17 of them (MRR 0.89). For 14 questions, all 5 chunks came from the expected document.
- **Answer accuracy is 80% (16 of 20).** All 4 wrong answers had the gold passage retrieved (ranked 1st, 1st, 2nd and 5th), but the exact answer text was not in the five 300-character chunks. The right area of the document was found, but not the sentence holding the answer. Larger chunks or a few more of them would likely fix most of these.
- **One "correct" verdict is generous.** For "What is the name of Katrina's father?" the gold passage was missed, and the answer says the name is not mentioned. The judge still accepted it, and the reference ("A substantial Dutch farmer") is itself wrong. A strict count would be 15 of 20.
- **Retrieval is the slowest step.** It takes a median of 1.7 s out of 3.0 s end to end; WSD linking takes 0.3 s and answer generation 1.1 s. The 3-hop "local" subgraph is almost global: common senses such as "object" pull in 935 of the 1,012 chunks. Each question therefore runs Leiden on about 6,500 nodes (around 1 s), plus a 0.6 s Neo4j query.
- **The partition is clean.** On a sample question Leiden found 18 communities with modularity 0.52. Results are deterministic (fixed seed).

## Recommended next steps

- **Precompute the Leiden partition once for the whole graph** and cache it, or store a community ID on each chunk. Because the per-question subgraph is nearly global, this should give almost the same ranking while cutting retrieval from about 1.7 s to milliseconds. It is a one-off graph computation, not a WSD re-ingest.
- **Test a larger context** (top-k 8, or 500-character chunks) to address the 4 answer misses where the right passage was found.
- **Re-run on more than 20 questions**, or on all 91, before drawing firm conclusions. With 20 questions, one question moves accuracy by 5 points.

## What changed

- `GraphRepository.find_chunks_by_ppr()` is replaced by `find_chunks_by_leiden(synset_ids, top_k=5, resolution=1.0, seed=42)`. It returns the same `{chunk_id, source, text, score, matches}` records.
- `symbolic_service.query()` and `query_data_by_wsd.py` call it. The API's `top_k` is now passed through, where before it was capped at 5. The API endpoints and response formats are unchanged.

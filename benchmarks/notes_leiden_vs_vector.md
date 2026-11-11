- **Answer accuracy is effectively tied.** The judge counted 16 of 20 correct for Leiden and 15 of 20 for vector. Both get 14 right; Leiden alone gets 2, vector alone 1, and 3 defeat both. Reading the answers, one Leiden "correct" is generous (Katrina's father: it said the name was not mentioned, while vector retrieved the passage and answered). One vector "correct" is lenient (#16: "tampering with HTTP requests" for "parameter fuzzing"). With 20 questions, one question moves accuracy by 5 points, so treat the two as level on accuracy.
- **Leiden ranks the gold passage higher; vector finds it more reliably.** Leiden puts the gold chunk first for 17 of 20 questions (MRR 0.89), against 12 for vector (MRR 0.77). Vector, however, has the gold chunk somewhere in its top 5 for all 20 questions (hit@5 1.00), while Leiden misses it once (0.95).
- **Vector is twice as fast.** Median end-to-end time is 1.5 s for vector and 3.0 s for Leiden. The gap is almost all retrieval: 0.4 s (query embedding plus Chroma search) against 1.7 s (Neo4j subgraph query plus Leiden). Leiden also needs the 0.3 s WSD call per question. Answer generation costs the same (about 1.0–1.1 s).
- **Where they disagree.** Leiden answered #2 (AI SBOM guidance, gold at rank 1 vs rank 4 for vector) and #9 (Ichabod's companion, gold at rank 1 vs 2). Vector answered #16 (input manipulation), where Leiden had the gold passage only at rank 5. In each case, the method that ranked the gold passage higher answered correctly.
- **The shared failures are context-size problems, not retrieval failures.** Both methods retrieved the gold passage for #8, #14 and #15, but neither's five 300-character chunks contained the answering sentence (answer-in-context 0 for both).

## Recommendations

- **Precompute the Leiden communities once** to remove most of its 1.7 s retrieval time. The per-question "local" subgraph already covers about 92% of the graph, so a cached global partition should rank almost identically. This needs no WSD re-ingest.
- **Combine the two:** vector for recall (it always found the passage), Leiden or sense matching to reorder its candidates. Leiden's ranking advantage was what decided every question where the two disagreed.
- **Give both more context** (top-k 8 or larger chunks) to fix the 3 shared misses.
- **Re-run on all 91 questions** before deciding. The difference in accuracy here is within the noise of 20 questions.

## Setup

- **Questions:** the first 20 rows of `data/eval/testset_stripped.csv`.
- **Leiden:** symbolic WSD retrieval with Leiden community ranking (`find_chunks_by_leiden`, top 5).
- **Vector:** the vector pipeline's Chroma similarity search (top 5, with the service's 0.7 relevance cutoff, which did not trigger on any of these 20 questions).
- **Shared:** the same answer prompt, gpt-4.1-mini at temperature 0, and the same judge.
- **Data:** no re-ingestion. Both used the existing Chroma store and Neo4j graph.

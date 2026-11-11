- **Vector edges out label propagation on these 20 questions: 15 correct against 14.** Both get 13 right. LPA alone answers #2 (AI SBOM guidance: gold passage ranked 1st, against 4th for vector). Vector alone answers #3 (Katrina's father) and #16 (input manipulation), where LPA never retrieved the gold passage. Four questions defeat both (#8, #9, #14, #15). With 20 questions the gap is one question, so treat them as close, with vector slightly ahead.
- **Vector finds the right passage more reliably; LPA ranks it first slightly more often.** Vector has the gold chunk in its top 5 for all 20 questions (hit@5 1.00), against 17 for LPA (0.85). LPA ranks it first for 13 (MRR 0.75), vector for 12 (MRR 0.77). All three LPA misses came from chunks of *other* documents entering the top 5.
- **Vector is twice as fast.** Median end-to-end time is 1.5 s for vector and 2.9 s for LPA. The gap is retrieval: 0.4 s (query embedding plus Chroma search) against 1.5 s (Neo4j 3-hop subgraph query plus 5 label-propagation runs). LPA also needs a 0.3 s WSD call per question. Answer generation costs the same (about 1.0 s).
- **Why LPA mixes documents.** On this graph label propagation collapses into one giant community (97–98% of nodes, modularity about 0.01). Common senses connect almost every chunk, so the communities don't separate topics, and ranking falls back to sense matching. Vector similarity has no such hub problem.
- **The shared failures are about context size.** For #8, #14 and #15 both retrieved the gold passage, but the five 300-character chunks lacked the answering sentence.

## Recommendations

- **Between these two, vector is the better default:** it is slightly more accurate, always finds the passage, and is twice as fast.
- **For symbolic retrieval, prefer Leiden over LPA** (16 correct on the same questions). Or make LPA usable by removing hub senses before propagation.
- **Give both more context** (top-k 8 or larger chunks), and **re-run on all 91 questions** before deciding.

## Setup

- **Questions:** the first 20 rows of `data/eval/testset_stripped.csv`. Both methods use the top 5 chunks, the same answer prompt, gpt-4.1-mini at temperature 0, and the same judge.
- **Vector:** Chroma similarity search with the service's 0.7 relevance cutoff, which never triggered here.
- **LPA:** `find_chunks_by_label_propagation`, 5 seeded runs averaged.
- **Data:** no re-ingestion.

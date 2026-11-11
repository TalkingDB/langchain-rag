- **Label propagation is the weakest of the three on these 20 questions.** It gets 14 correct, against 16 for Leiden and 15 for vector. It finds the gold passage for 17 of 20 (hit@5 0.85) and ranks it first for 13 (MRR 0.75). LPA never answers a question that Leiden gets wrong; Leiden answers #3 and #9, where LPA fails.
- **LPA collapses into one giant community on this graph.** On a sample question it found 12–16 communities, but the largest held 97–98% of all nodes (modularity about 0.01, against 0.52 for Leiden's 18 communities). This is a known LPA weakness on dense graphs with hub nodes, here common senses such as "object" linked to hundreds of chunks. With almost everything in one community, the community part of the score barely separates chunks, and ranking falls back to each chunk's own rarity-weighted sense matches.
- **That causes cross-document mixing.** All three LPA misses (#3, #9, #16) put chunks from other documents into the top 5. For #9 it mixed OWASP, Thoreau and Sleepy Hollow chunks. For #3 it produced a confident wrong answer ("Hans Van Ripper") from an unrelated passage. Leiden's topical communities kept those out.
- **The algorithm is fast, but the retrieval step isn't.** One LPA run takes about 20 ms (Leiden takes about 1 s), and LPA runs 5 times with averaged scores. Retrieval still has a median of 1.46 s, because almost all of it is the Neo4j 3-hop subgraph query. End to end is 2.9 s, the same as Leiden; vector is 1.5 s.
- **The shared failures remain.** #8, #14 and #15 fail for all three methods: the gold passage was retrieved, but the five 300-character chunks didn't contain the answering sentence.

## Recommendations

- **Leiden is the better community method for this graph.** Keep LPA only if the graph becomes much sparser, for example after removing very common senses.
- **If LPA stays,** drop or down-weight hub senses (the most common 1–2% of senses) before propagation. That should stop the giant-community collapse.
- **For speed with either method,** cache the communities once for the whole graph instead of running a 3-hop Neo4j query per question.
- **Re-run on all 91 questions** before a final decision. With 20 questions, one question moves accuracy by 5 points.

## Setup

- **Questions:** the first 20 rows of `data/eval/testset_stripped.csv`. These are the same questions, prompt, model (gpt-4.1-mini) and judge as the earlier Leiden and vector runs, which are shown here for comparison.
- **LPA:** `find_chunks_by_label_propagation(synset_ids, top_k=5, runs=5, seed=42)`. It builds the same 3-hop local subgraph as Leiden, runs igraph's Raghavan–Albert–Kumara label propagation 5 times with fixed seeds, and averages the chunk scores.
- **Data:** no re-ingestion.

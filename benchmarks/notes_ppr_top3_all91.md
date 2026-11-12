- **Answer accuracy: 58 of 91 (64%).** Two of the 33 "wrong" verdicts are judge mistakes. In #90 the answer is the reference, "12 June 2004", word for word. In #68 it names Brom Van Brunt, whose given name in the story is Abraham, the reference. Counting those two, answer accuracy is 60 of 91 (66%).
- **Retrieval accuracy: 66 of 91 (73%).** For 66 questions a gold chunk is among the 3 retrieved. It is ranked first for 47 (52%), second for 16 and third for 3 (MRR 0.62). PPR often finds the right passage but puts it at rank 2 behind a neighbouring chunk that shares the same senses.
- **Why the 33 wrong answers failed:**
  - **20: the gold chunk wasn't in the top 3.** In 15 of these the chunks came from the right document but the wrong passage. In 5 (#22, #26, #80, #81, #87) none of the chunks came from the right document. #81 and #87 are Sleepy Hollow questions about Ichabod that returned only OWASP chunks: the walk drifted from the seed senses to chunks connected through common senses.
  - **11: the gold chunk was retrieved, but the answer sentence wasn't in it** (#8, #9, #14, #15, #16, #31, #40, #50, #60, #77, #83). The chunk overlaps the reference passage, but the sentence with the answer is in a neighbouring 300-character chunk. The model then says "not in the context" or guesses. Ranking can't fix these.
  - **2: the answer was in the context** and the judge marked it wrong (#68, #90, above).
- **5 answers were correct without a gold hit** (#3, #29, #38, #44, #45). The fact was in a chunk that overlaps the reference passage by less than 50%, so retrieval accuracy slightly understates what the model saw.
- **By document:** the OpenChain AI SBOM guide is best (7/9 correct, retrieval 9/9), followed by the OWASP Top 10 (19/25, 76%) and Thoreau (14/22, 64%). The Legend of Sleepy Hollow is weakest (18/35, 51%), with retrieval at 23/35: its questions ask about names and small narrative details.
- **PPR sometimes leaves the right document.** 214 of the 273 retrieved chunks (78%) come from the question's source document. PPR also scores chunks that share senses with the matching chunks, so common senses can pull in chunks from other books.
- **Speed:** the median end-to-end time is 4.8 s per question. That is WSD 0.6 s + retrieval 2.9 s + answer 1.1 s. Retrieval peaked at 13.5 s. Each query fetches the 3-hop subgraph around the seed senses (sense → chunk → sense → chunk), which covers most of the chunk graph. It then runs the PageRank iteration in Python, up to 100 iterations over every edge.

## Recommendations

- **Send neighbouring chunks with each top chunk** (n-1, n+1), or use larger chunks. This targets the 11 questions where the right chunk was found but the answer sentence was cut off.
- **Keep the walk in the seed chunks' document,** or lower the damping (0.85 now) so more score returns to the query's own senses. This targets the 5 cross-document misses and the rank-2 placements.
- **Precompute or cache the walk's graph** instead of fetching a near-global subgraph for every query, and run the PageRank iteration with a vectorised or sparse-matrix implementation. That should cut the 2.9 s median and the 13.5 s worst case.
- **Fix the judge** with an exact-match short-circuit (reference contained in the answer counts as correct), so cases like #90 aren't marked wrong.
- **Clean the testset:** questions that don't name their document (#22, #26, #80) penalise every retriever.

## Setup

- **Questions:** all 91 rows of `data/eval/testset_stripped.csv` (in sheet order).
- **Method:** symbolic WSD retrieval ranked by Personalized PageRank (`find_chunks_by_ppr`, `top_k=3`, damping 0.85), on branch `feat/ppr-retrieval` at `329167e`.
- **Answer accuracy:** the pipeline's answer prompt with gpt-4.1-mini at temperature 0. The same model judges whether the answer conveys the reference fact.
- **Retrieval accuracy:** a retrieved chunk counts as gold when at least 50% of its word 4-grams fall inside the question's reference passage. A question scores when any of the 3 chunks is gold.
- **Data:** the existing Neo4j chunk graph (1,034 chunks, 20,516 MENTIONS). The WordNet graph loaded earlier today is in the same database but isn't traversed. No re-ingestion.

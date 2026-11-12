- **Leiden (top 3) answers 61 of 91 questions correctly (67%).** Two of the 30 "wrong" verdicts are judge mistakes. In #90 the answer is the reference, "12 June 2004", word for word. In #68 it names Brom Van Brunt, whose given name in the story is Abraham, the reference. Counting those two, accuracy is 63 of 91 (69%).
- **Retrieval puts the gold passage first for 55 questions (60%).** It is somewhere in the top 3 for 68 (hit@3 75%, MRR 0.67). When retrieval finds it, it is usually at rank 1: 55 at rank 1, 9 at rank 2 and 4 at rank 3.
- **Why the 30 wrong answers failed:**
  - **17: the gold passage wasn't retrieved.** In 14 of these the chunks came from the right document but the wrong passage. In 3 (#22, #26, #80) none of the chunks came from the right document. Those are questions that don't name their document, such as "What is the title of the fourth section?" or "What is the name of the main character?". The WSD synsets for such generic words don't point to any particular document.
  - **10: the gold passage was retrieved, but the answer sentence wasn't in it** (#8, #14, #15, #27, #31, #40, #70, #71, #77, #83). Each of these chunks overlaps the reference passage, but the sentence with the answer falls in a neighbouring 300-character chunk. The model then says "not in the context" or guesses. This is a context-size problem, not a ranking problem.
  - **3: the answer was in the context.** Two are the judge mistakes above. In #45 the model answered the question and the reference ("Let it stand") is a weak gold answer.
- **6 answers were correct without a gold hit** (#3, #34, #37, #41, #44, #81). The fact was in another chunk that overlaps the reference passage by less than 50%, so hit@3 slightly understates retrieval.
- **By document:** the OWASP Top 10 is best (20/25 correct, 80%), followed by the OpenChain AI SBOM guide (7/9, 78%) and Thoreau (14/22, 64%). The Legend of Sleepy Hollow is weakest (20/35, 57%). Its questions ask about names and small narrative details ("Who is the preceptor?", "What did Brom teach his dog?") that sit one chunk away from where the synsets match.
- **Leiden stays inside the right document.** 238 of the 273 retrieved chunks (87%) come from the question's source document.
- **Top 3 vs the earlier top 5:** on the first 20 questions, top 3 got 15 correct against 16 for the earlier top-5 run. The only difference is #9, where the gold passage was at rank 4 or 5. Hit fell from 19 to 17 and answer-in-context stayed at 12. That run was before the WordNet load. Cutting the context from 5 chunks to 3 costs little accuracy here, but gives the 300-character chunks even less room.
- **Speed:** the median end-to-end time is 5.2 s per question. That is WSD 0.8 s + retrieval 3.1 s + answer 1.2 s. Retrieval is slower than the 1.7 s measured on the same questions before the WordNet load, and it is noisy (2.3–8.2 s when I re-timed 8 questions). Each query fetches a "local" subgraph of 16,000–20,000 MENTIONS edges, which is close to the whole chunk graph (20,516 edges). It then re-partitions that subgraph with Leiden: about 1 s for the Cypher query and 1.5–2 s for the partition. The WordNet edges loaded today are not traversed: the subgraph row counts match the MENTIONS edges alone.

## Recommendations

- **Expand the context around the top-ranked chunks.** Sending each retrieved chunk together with its neighbouring chunks (n-1, n+1), or using larger chunks, targets the 10 questions where the right passage was found but the answer sentence was cut off. That is the largest group that ranking can't fix.
- **Precompute the Leiden partition once.** The per-query subgraph is almost the entire graph, so a cached global partition should rank nearly the same and remove most of the 3 s retrieval time.
- **Fix the judge before comparing methods.** Add an exact-match short-circuit (reference contained in the answer counts as correct), so cases like #90 aren't marked wrong. With 91 questions, each judge error moves accuracy by about 1 point.
- **Clean the testset.** Questions that don't name their document (#22, #26, #80) and weak references (#45) penalise any retriever. Mark or rewrite them so the numbers reflect retrieval quality.
- **Try WordNet expansion next.** The WordNet graph is now loaded but unused. Following GLOSS_LINKS_TO from the question's synsets could help the 14 right-document, wrong-passage misses, where the passage uses related words rather than the question's own.

## Setup

- **Questions:** all 91 rows of `data/eval/testset_stripped.csv` (in sheet order).
- **Method:** symbolic WSD retrieval with Leiden community ranking (`find_chunks_by_leiden`, `top_k=3`), on branch `feat/leiden-retrieval` at `2ba636b`.
- **Answer and judge:** the pipeline's answer prompt, gpt-4.1-mini at temperature 0, and the same model as judge.
- **Data:** the existing Neo4j chunk graph (1,034 chunks, 20,516 MENTIONS), now alongside the WordNet graph loaded earlier today. No re-ingestion.
- **Run-to-run variance:** a first attempt crashed at question 69 when the network dropped. Of the questions both runs completed, #50 changed from wrong to correct, so expect about ±1 question of noise.

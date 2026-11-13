- **Answer accuracy: 64 of 91 (70%).** One of the 27 "wrong" verdicts is a judge mistake: in #42 the answer says Ichabod had "a soft and foolish heart", the reference word for word. Counting it, answer accuracy is 65 of 91 (71%).
- **Retrieval accuracy: 65 of 91 (71%).** For 65 questions a gold chunk is among the 3 retrieved. It is ranked first for 47 (52%), second for 15 and third for 3 (MRR 0.61).
- **Why the 27 wrong answers failed:**
  - **19: the gold chunk wasn't in the top 3.**
    - 5 retrieved nothing (#3, #33, #54, #69, #80). The best chunk scored below the pipeline's 0.7 relevance cutoff, so it answered with no context, as the real pipeline would.
    - 1 retrieved the wrong book (#22, "What is the title of the fourth section?", answered from Alice in Wonderland).
    - 13 got the right document but the wrong passage.
  - **6: the gold chunk was retrieved, but the answer sentence wasn't in it** (#9, #14, #15, #26, #40, #84). The chunk overlaps the reference passage, but the sentence with the answer is in a neighbouring 300-character chunk.
  - **2: the answer was in the context.** #42 is the judge mistake above. In #91 the model hedged instead of quoting the passage.
- **7 answers were correct without a gold hit** (#46, #48, #52, #55, #79, #81, #88). The fact was in a chunk that overlaps the reference passage by less than 50%, so retrieval accuracy slightly understates what the model saw.
- **Vector stays in the right document.** 252 of the 258 retrieved chunks (98%) come from the question's source document. Every wrong-document or empty result is one of the 6 questions above.
- **Names are the weak spot.** The Legend of Sleepy Hollow scores 17/35 (49%), against 22/25 for the OWASP Top 10 (88%), 8/9 for the OpenChain AI SBOM guide and 17/22 for Thoreau (77%). Most Sleepy Hollow misses ask for a name or a short detail ("Katrina", "Hans Van Ripper", "Wiley's Swamp", "Brom Bones"). The embedding search returns the same general passages about Ichabod instead. The chunks starting at characters 3149, 714 and 1459 of the story come back for many different questions.
- **Speed:** the median end-to-end time is 1.4 s per question. That is retrieval 0.47 s (query embedding plus Chroma search) + answer 0.9 s. There is no WSD step. The slowest question took 6.4 s, and retrieval peaked at 4.0 s on a slow embedding call.

## Recommendations

- **Send neighbouring chunks with each top chunk** (n-1, n+1), or use larger chunks. This targets the 6 questions where the gold chunk was found but the answer sentence was cut off.
- **Add a keyword or entity signal for names.** A hybrid search (BM25 plus embeddings), or reranking the vector candidates by exact sense or entity match, targets the Sleepy Hollow name questions that embeddings miss.
- **Revisit the 0.7 relevance cutoff.** It discarded all context for 5 questions. Lowering it, or still answering from the best chunks with a caution, would give the model a chance.
- **Fix the judge** with an exact-match short-circuit (reference contained in the answer counts as correct), so cases like #42 aren't marked wrong.
- **Clean the testset:** questions that don't name their document (#22, #80) penalise every retriever.

## Setup

- **Questions:** all 91 rows of `data/eval/testset_stripped.csv` (in sheet order).
- **Method:** the vector pipeline as `query_data.py` and `api/services/vector_service.py` run it. That is a Chroma similarity search with k=3 and OpenAI embeddings, and no context when the best chunk's relevance is below 0.7.
- **Answer accuracy:** the pipeline's answer prompt with gpt-4.1-mini at temperature 0. The same model judges whether the answer conveys the reference fact.
- **Retrieval accuracy:** a retrieved chunk counts as gold when at least 50% of its word 4-grams fall inside the question's reference passage. A question scores when any of the 3 chunks is gold.
- **Data:** the existing Chroma store, with no re-embedding. Same questions, prompt, model and judge as the Leiden and PPR top-3 reports.

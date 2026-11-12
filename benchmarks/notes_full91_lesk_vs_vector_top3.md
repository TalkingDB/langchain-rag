- **Vector is more accurate at top 3: 63 of 91 correct (69%) against 58 (64%) for Extended Lesk.** Both get 48 right. Lesk alone gets 10, vector alone 15, and 18 defeat both. For context, label propagation scored 61 on the same 91 questions at top 3.
- **Vector also retrieves better.** It has the gold passage in its top 3 for 65 questions (hit@3 0.71), against 61 for Lesk (0.67). Both rank it first about equally often (47 against 46; MRR 0.61 against 0.58). The reference answer is inside the retrieved chunks for 52 questions with vector but only 42 with Lesk, and that gap explains most of the accuracy difference.
- **Lesk usually finds the right document but not the right passage.** Of its 30 retrieval misses, only 4 drew all three chunks from other documents; in the other 26 the chunks came from the right document but from the wrong part of it. Lesk ranks chunks by how related their *senses* are to the question's senses. Within one book many passages share related vocabulary, so relatedness often fails to single out the passage holding the answer.
- **Lesk wins on narrative text.** On The Legend of Sleepy Hollow it gets 57% against vector's 51%, matching paraphrased wording through related senses (for example "steed" and "horse"). Vector is clearly ahead on the OpenChain guide (78% against 56%), OWASP 2004 (88% against 76%) and Civil Disobedience (73% against 64%).
- **Lesk has the fastest retrieval of any method tested: median 109 ms** (vector 431 ms, label propagation 1,182 ms). After a one-off 6.5 s index build, each question needs only an in-memory relatedness lookup and one small Neo4j fetch. Lesk also always returns chunks, while vector returned none for 5 questions because of its 0.7 relevance cutoff.
- **The end-to-end time in this run is distorted by the WSD service.** Median end-to-end is 2.4 s for Lesk against 1.4 s for vector, but 1.2 s of Lesk's time was the WSD service linking the question. That call was about 4 times slower in this run than in the earlier runs (0.3 s). With a normal WSD response, Lesk's end-to-end time would be about the same as vector's (roughly 1.4 s).

## Recommendations

- **Vector remains the better default** for accuracy on this dataset.
- **Lesk's speed and narrative-text strength make it a good reranker or second signal,** for example fused with vector scores, rather than a replacement.
- **To improve Lesk itself:** blend in a passage-level signal, such as how many query senses a chunk mentions directly or the chunk's vector similarity, so related senses don't outweigh the exact passage. Try top 5 as well: on the other methods, 5 chunks gave 2 more correct answers than 3.

## Setup

- **Questions:** all 91 rows of `data/eval/testset_stripped.csv`.
- **Shared:** both methods used the top 3 chunks, the same answer prompt, gpt-4.1-mini at temperature 0, and the same judge.
- **Vector:** results come from the 91-question run of the same benchmark script on the label-propagation branch.
- **Extended Lesk:** `find_chunks_by_extended_lesk(synset_ids, top_k=3)`. Banerjee & Pedersen phrasal gloss overlap, with each shared phrase of n words scoring n²; extended glosses include the glosses of up to 25 related WordNet senses per relation; identical senses score 1 and related senses up to 0.8; query senses are idf-weighted.
- **Data:** no re-ingestion.

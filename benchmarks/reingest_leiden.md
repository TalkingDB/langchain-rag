**No re-ingestion is needed.** Leiden runs at query time on the same `Chunk -MENTIONS-> Sense` graph that the WSD ingestion already stored in Neo4j: 1,012 chunks, 5,714 senses and 19,925 links. Chunking and chunk IDs are unchanged, the vector store isn't involved, and nothing new is written to the database.

Only the Python dependencies change: `pip install -r requirements.txt` now also installs `leidenalg` and `igraph`.

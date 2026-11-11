**No re-ingestion is needed.** Label propagation runs at query time on the same `Chunk -MENTIONS-> Sense` graph that the WSD ingestion already stored in Neo4j: 1,012 chunks, 5,714 senses and 19,925 links. Chunk IDs are unchanged and nothing new is written to the database.

The only new Python dependency is `igraph`, now in `requirements.txt`.

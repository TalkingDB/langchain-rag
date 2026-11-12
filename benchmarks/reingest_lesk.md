**No re-ingestion is needed.** Extended Lesk reads the `Chunk -MENTIONS-> Sense` graph and the gloss stored on every Sense node, both written by the existing WSD ingestion (1,012 chunks, 5,714 senses, 19,925 links). WordNet relations come from NLTK's WordNet 3.0 at query time; the WSD service's sense IDs are WordNet 3.0 offsets, which I checked.

The new Python dependency is `nltk` (its WordNet data, about 10 MB, downloads automatically on first use). The relatedness index is built in memory once per process, in about 6.5 s.


import math
import os
import random
from typing import Any

from neo4j import Driver, GraphDatabase


class GraphRepository:
    """
    Handles graph database operations using Neo4j.

    Current responsibility:
    - Insert or update Chunk nodes.
    - Insert or update Sense nodes.
    - Create MENTIONS relationships between Chunk and Sense nodes.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
    ):
        uri = uri or os.getenv("NEO4J_URI")
        username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD")

        if not uri:
            raise ValueError("NEO4J_URI is required.")

        if not password:
            raise ValueError("NEO4J_PASSWORD is required.")

        self.database = database
        self.driver: Driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            notifications_min_severity="WARNING",
        )

    # --------------------------------------------------
    # Connection management
    # --------------------------------------------------

    def verify_connection(self) -> None:
        """Verify the Neo4j database connection."""
        self.driver.verify_connectivity()

    def close(self) -> None:
        """Close the Neo4j connection."""
        self.driver.close()

    # --------------------------------------------------
    # Database constraints
    # --------------------------------------------------

    def create_constraints(self) -> None:
        """Create uniqueness constraints for graph identifiers."""

        queries = [
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk)
            REQUIRE c.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT sense_synset_id_unique IF NOT EXISTS
            FOR (s:Sense)
            REQUIRE s.synset_id IS UNIQUE
            """
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.run(query).consume()

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    @staticmethod
    def _validate_input(data: list[dict[str, Any]]) -> None:
        """Validate the input structure before database insertion."""

        if not isinstance(data, list):
            raise ValueError("Input data must be a list of objects.")

        for index, item in enumerate(data):

            if not isinstance(item, dict):
                raise ValueError(f"Item at index {index} must be an object.")

            required_fields = ["chunk_id", "source", "text", "senses"]

            for field in required_fields:
                if field not in item:
                    raise ValueError(
                        f"Missing field '{field}' in item at index {index}."
                    )

            if not isinstance(item["chunk_id"], str) or not item["chunk_id"].strip():
                raise ValueError(f"Invalid chunk_id at index {index}.")

            if not isinstance(item["source"], str):
                raise ValueError(f"Invalid source at index {index}.")

            if not isinstance(item["text"], str):
                raise ValueError(f"Invalid text at index {index}.")

            if not isinstance(item["senses"], list):
                raise ValueError(f"Senses must be a list at index {index}.")

            for sense_index, sense in enumerate(item["senses"]):

                if not isinstance(sense, dict):
                    raise ValueError(
                        f"Sense at index {sense_index} in item {index} must be an object."
                    )

                sense_fields = ["surface", "synset_id", "sense"]

                for field in sense_fields:
                    if field not in sense:
                        raise ValueError(
                            f"Missing '{field}' in sense at index {sense_index} "
                            f"of item {index}."
                        )

                    if not isinstance(sense[field], str) or (field != "sense" and not sense[field].strip()):
                        raise ValueError(
                            f"Invalid '{field}' in sense at index {sense_index} "
                            f"of item {index}."
                        )

    # --------------------------------------------------
    # Chunk node insertion
    # --------------------------------------------------

    @staticmethod
    def _insert_chunks(tx, data: list[dict[str, Any]]) -> None:
        """Create or update Chunk nodes."""

        query = """
        UNWIND $data AS item

        MERGE (chunk:Chunk {id: item.chunk_id})

        SET
            chunk.source = item.source,
            chunk.text = item.text
        """

        tx.run(query, data=data).consume()

    # --------------------------------------------------
    # Sense node and relationship insertion
    # --------------------------------------------------

    @staticmethod
    def _insert_senses(tx, data: list[dict[str, Any]]) -> None:
        """Create or update Sense nodes and MENTIONS relationships."""

        query = """
        UNWIND $data AS item

        MATCH (chunk:Chunk {id: item.chunk_id})

        UNWIND item.senses AS sense_data

        MERGE (sense:Sense {
            synset_id: sense_data.synset_id
        })

        // Don't overwrite a stored definition with an empty one.
        SET sense.sense = CASE WHEN sense_data.sense <> "" THEN sense_data.sense ELSE sense.sense END

        MERGE (chunk)-[rel:MENTIONS {
            surface: sense_data.surface
        }]->(sense)
        """

        tx.run(query, data=data).consume()

    # --------------------------------------------------
    # Main insertion method
    # --------------------------------------------------

    def insert_data(self, data: list[dict[str, Any]]) -> dict[str, int]:
        """
        Insert a list of Chunk and Sense records into Neo4j.

        Expected input:
        [
            {
                "chunk_id": "...",
                "source": "...",
                "text": "...",
                "senses": [
                    {
                        "surface": "...",
                        "synset_id": "...",
                        "sense": "..."
                    }
                ]
            }
        ]
        """

        self._validate_input(data)

        if not data:
            return {
                "chunks_inserted": 0,
                "senses_processed": 0
            }

        def transaction_work(tx):
            self._insert_chunks(tx, data)
            self._insert_senses(tx, data)

        with self.driver.session(database=self.database) as session:
            session.execute_write(transaction_work)

        return {
            "chunks_inserted": len(data),
            "senses_processed": sum(
                len(item["senses"]) for item in data
            )
        }
    
    def insert_single_data(self, data: dict[str, Any]) -> dict[str, int]:
        """
        Insert a single Chunk and its associated Sense records into Neo4j.
        """
    
        if not isinstance(data, dict):
            raise ValueError("Input data must be a single object (dictionary).")
    
        return self.insert_data([data])

    # --------------------------------------------------
    # Querying
    # --------------------------------------------------

    def find_chunks_by_synset_ids(self, synset_ids: list[str]) -> list[dict[str, Any]]:
        """
        Return all chunks that mention any of the given synset ids, matched exactly.
        The lookup on Sense.synset_id uses the range index behind the
        sense_synset_id_unique constraint. Chunks that match more distinct
        synsets rank first.
        """

        query = """
        MATCH (sense:Sense)
        WHERE sense.synset_id IN $synset_ids
        MATCH (chunk:Chunk)-[:MENTIONS]->(sense)
        WITH chunk, count(DISTINCT sense) AS matches
        RETURN chunk.id AS chunk_id, chunk.source AS source, chunk.text AS text, matches
        ORDER BY matches DESC, chunk_id
        """

        records, _, _ = self.driver.execute_query(
            query, synset_ids=synset_ids, database_=self.database
        )
        return [record.data() for record in records]

    # Local subgraph around the query: seed sense -> chunk -> sense -> chunk, with the
    # number of MENTIONS edges between each chunk and sense as the edge weight.
    _LOCAL_SUBGRAPH_QUERY = """
        MATCH (seed:Sense) WHERE seed.synset_id IN $synset_ids
        MATCH (seed)<-[:MENTIONS]-(c1:Chunk)
        WITH collect(DISTINCT c1) AS hop1
        CALL (hop1) {
            UNWIND hop1 AS c1
            MATCH (c1)-[:MENTIONS]->(:Sense)<-[:MENTIONS]-(c2:Chunk)
            RETURN collect(DISTINCT c2) AS hop3
        }
        UNWIND hop1 + hop3 AS chunk
        WITH DISTINCT chunk
        MATCH (chunk)-[:MENTIONS]->(sense:Sense)
        RETURN chunk.id AS chunk_id, sense.synset_id AS synset_id, count(*) AS weight
        """

    def find_chunks_by_label_propagation(
        self,
        synset_ids: list[str],
        top_k: int = 5,
        runs: int = 5,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """
        Rank chunks with the Label Propagation Algorithm (Raghavan, Albert & Kumara, 2007)
        around the query's synsets.

        1. Build the local subgraph within three hops of the query's Sense nodes
           (seed sense -> chunk -> sense -> chunk) from the bipartite Chunk-MENTIONS-Sense
           graph, weighted by mention counts.
        2. Detect communities with label propagation: every node starts with its own label
           and repeatedly adopts the label with the largest total edge weight among its
           neighbours (random update order, random tie-breaks) until labels stop changing.
        3. LPA is randomised and single runs often differ, so it runs `runs` times with
           seeds seed, seed+1, ... and chunk scores are averaged over the runs
           (deterministic for a given seed).
        4. In each run, query senses are weighted by rarity, idf = log(1 + N / df) over the
           N chunks of the subgraph. A community's relevance is the share of the query's
           idf weight inside it; a chunk scores its community's relevance plus its own
           direct match (the share of the query's idf weight it mentions).
           Ties: stronger connection inside its community, then chunk id.

        Returns the top_k chunks as {chunk_id, source, text, score, matches}, where matches is
        the number of query synsets the chunk mentions directly.
        """
        try:
            import igraph as ig
        except ImportError as e:  # pragma: no cover - dependency guard
            raise RuntimeError("Label propagation retrieval needs: pip install igraph") from e

        seeds = list(dict.fromkeys(synset_ids))
        if not seeds:
            return []

        records, _, _ = self.driver.execute_query(
            self._LOCAL_SUBGRAPH_QUERY, synset_ids=seeds, database_=self.database
        )
        if not records:
            return []

        # Node keys are prefixed so a chunk id can never collide with a synset id.
        neighbours: dict[str, dict[str, float]] = {}
        for r in records:
            c, s, w = "c:" + r["chunk_id"], "s:" + r["synset_id"], float(r["weight"])
            neighbours.setdefault(c, {})[s] = w
            neighbours.setdefault(s, {})[c] = w

        present = ["s:" + sid for sid in seeds if "s:" + sid in neighbours]
        if not present:
            return []

        nodes = sorted(neighbours)
        index = {node: i for i, node in enumerate(nodes)}
        edges, weights = [], []
        for node, nbrs in neighbours.items():
            if node.startswith("c:"):
                for nbr, w in nbrs.items():
                    edges.append((index[node], index[nbr]))
                    weights.append(w)
        graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
        graph.es["weight"] = weights

        chunks = [node for node in nodes if node.startswith("c:")]
        idf = {s: math.log(1 + len(chunks) / len(neighbours[s])) for s in present}
        total = sum(idf.values()) or 1.0
        direct = {c: sum(idf[s] for s in present if s in neighbours[c]) / total for c in chunks}

        score = {c: 0.0 for c in chunks}
        inside = {c: 0.0 for c in chunks}
        for run in range(max(1, runs)):
            ig.set_random_number_generator(random.Random(seed + run))
            membership = graph.community_label_propagation(weights="weight").membership
            community = {node: membership[index[node]] for node in nodes}
            relevance: dict[int, float] = {}
            for s in present:
                relevance[community[s]] = relevance.get(community[s], 0.0) + idf[s] / total
            for c in chunks:
                comm = community[c]
                score[c] += relevance.get(comm, 0.0) + direct[c]
                inside[c] += sum(w for nbr, w in neighbours[c].items() if community[nbr] == comm)
        ig.set_random_number_generator(random)  # restore igraph's default generator

        n_runs = max(1, runs)
        ranked = sorted(chunks, key=lambda c: (-score[c], -inside[c], c))[:top_k]
        ranked = [c for c in ranked if score[c] > 0]
        if not ranked:
            return []

        details, _, _ = self.driver.execute_query(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c.id AS chunk_id, c.source AS source, c.text AS text",
            ids=[c[2:] for c in ranked],
            database_=self.database,
        )
        by_id = {d["chunk_id"]: d.data() for d in details}
        seed_set = set(present)
        return [
            {
                **by_id[c[2:]],
                "score": round(score[c] / n_runs, 6),
                "matches": sum(1 for s in neighbours[c] if s in seed_set),
            }
            for c in ranked
        ]

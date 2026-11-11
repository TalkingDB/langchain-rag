"""Label-propagation retrieval on a small in-memory graph (Neo4j driver faked, no database)."""
from neo4j_graphdb import GraphRepository

# Two topics. Topic A: chunks a1-a3 about "dragon"/"castle"; topic B: chunks b1-b3 about
# "tax"/"vote". "common" appears everywhere. The query asks about dragon + common.
MENTIONS = {
    "a1": ["dragon", "castle", "common"],
    "a2": ["dragon", "castle", "knight"],
    "a3": ["castle", "knight", "common"],
    "b1": ["tax", "vote", "common"],
    "b2": ["tax", "vote", "law"],
    "b3": ["vote", "law", "common"],
}


class _Record(dict):
    def data(self):
        return dict(self)


class _FakeDriver:
    def execute_query(self, query, **params):
        if "UNWIND hop1" in query:  # local subgraph query
            rows = [_Record(chunk_id=c, synset_id=s, weight=1)
                    for c, senses in MENTIONS.items() for s in senses]
            return rows, None, None
        return [_Record(chunk_id=c, source=f"{c[0]}.pdf", text=f"text {c}") for c in params["ids"]], None, None


def _repo():
    repo = GraphRepository.__new__(GraphRepository)  # skip the real Neo4j connection
    repo.driver, repo.database = _FakeDriver(), "neo4j"
    return repo


def test_lpa_contract_and_topic_preference():
    results = _repo().find_chunks_by_label_propagation(["dragon", "common"], top_k=3)
    assert len(results) == 3
    assert set(results[0]) == {"chunk_id", "source", "text", "score", "matches"}
    # The chunks mentioning the rare query sense "dragon" rank first.
    assert {r["chunk_id"] for r in results[:2]} == {"a1", "a2"}
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert next(r for r in results if r["chunk_id"] == "a1")["matches"] == 2


def test_lpa_is_deterministic_for_a_seed_and_handles_empty():
    repo = _repo()
    run = lambda: repo.find_chunks_by_label_propagation(["dragon"], seed=7)  # noqa: E731
    assert run() == run()
    assert repo.find_chunks_by_label_propagation([]) == []
    assert repo.find_chunks_by_label_propagation(["unknown-sense"]) == []

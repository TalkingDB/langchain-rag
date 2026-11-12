"""Extended Lesk: phrasal overlap, related-sense ranking, and the GraphRepository contract.
Uses non-WordNet ids (w...) so only the stored glosses matter -- no corpus download needed."""
import neo4j_graphdb
from extended_lesk import LeskIndex, phrasal_overlap, tokens
from neo4j_graphdb import GraphRepository


def test_tokens_drop_stop_words():
    assert tokens("The horse is an animal of the farm") == ["horse", "animal", "farm"]


def test_phrasal_overlap_squares_phrase_lengths():
    # one shared 3-word phrase -> 9; two separate single words -> 1 + 1
    assert phrasal_overlap(["big", "red", "barn", "x"], ["y", "big", "red", "barn"]) == 9
    assert phrasal_overlap(["red", "x", "barn"], ["barn", "y", "red"]) == 2
    assert phrasal_overlap(["a"], ["b"]) == 0


ROWS = [  # (chunk_id, synset_id, gloss)
    ("c1", "w_steed", "a horse ridden by a knight into battle"),
    ("c1", "w_battle", "a hostile meeting of opposing military forces"),
    ("c2", "w_stallion", "an uncastrated adult male horse ridden in battle"),
    ("c3", "w_tax", "a charge against a citizen's person or property by the government"),
]


def test_related_senses_rank_above_unrelated():
    index = LeskIndex.build(ROWS)
    ranked = index.rank(["w_steed"], top_k=3)
    ids = [c for c, _, _ in ranked]
    assert ids[0] == "c1"            # mentions the identical sense
    assert ids[1] == "c2"            # related sense: its gloss shares "horse ridden ... battle"
    assert "c3" not in ids           # unrelated chunk gets no score
    assert ranked[0][2] == 1 and ranked[1][2] == 0  # direct matches


class _Record(dict):
    def data(self):
        return dict(self)


class _FakeDriver:
    def execute_query(self, query, **params):
        if "count(m)" in query:
            return [_Record(n=len(ROWS))], None, None
        if "RETURN DISTINCT c.id" in query:
            return [_Record(chunk_id=c, synset_id=s, gloss=g) for c, s, g in ROWS], None, None
        return [_Record(chunk_id=c, source=f"{c}.pdf", text=f"text {c}") for c in params["ids"]], None, None


def test_graph_repository_contract(monkeypatch):
    monkeypatch.setattr(neo4j_graphdb, "_LESK_CACHE", {})
    monkeypatch.setattr("extended_lesk.ensure_wordnet", lambda: None)
    repo = GraphRepository.__new__(GraphRepository)  # skip the real Neo4j connection
    repo.driver, repo.database = _FakeDriver(), "neo4j"
    assert repo.prepare_extended_lesk() == {"chunks": 3, "senses": 4, "mentions": 4}
    results = repo.find_chunks_by_extended_lesk(["w_steed"], top_k=2)
    assert [r["chunk_id"] for r in results] == ["c1", "c2"]
    assert set(results[0]) == {"chunk_id", "source", "text", "score", "matches"}
    assert results[0]["score"] >= results[1]["score"] > 0
    assert repo.find_chunks_by_extended_lesk([]) == []

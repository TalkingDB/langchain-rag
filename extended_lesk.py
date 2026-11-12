"""Extended Lesk relatedness (Banerjee & Pedersen, 2002/2003) for symbolic retrieval.

Each sense gets an *extended gloss*: its own gloss plus the glosses of WordNet senses linked to
it (hypernyms, hyponyms, meronyms, holonyms, also-see, similar-to, attributes, entailments,
causes). Two senses are related in proportion to the phrasal overlap of their extended
glosses: every maximal shared phrase of n content words scores n^2, so longer shared phrases
count more than scattered single words. Identical senses are maximally related.

Retrieval: a chunk's score sums, over the query's senses, the query sense's idf weight times
its best relatedness to any sense mentioned in the chunk, so chunks can match on senses that
are related to the query's senses, not only identical ones.

Pure functions plus a small index; Neo4j access lives in neo4j_graphdb.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Function words ignored when matching glosses (an overlap made only of these is meaningless).
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each either etc few for from
further had has have having he her here hers him his how i if in into is it its itself just
like may me might more most my no nor not of off on once one only or other our ours out over
own same she should so some something someone such than that the their them then there these
they this those through to too under until up upon us used very was we were what when where
which while who whom why will with would you your
""".split())

_WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-cased content words, in order (stop words removed)."""
    return [w for w in _WORD.findall((text or "").lower()) if w not in STOP_WORDS and len(w) > 1]


def phrasal_overlap(a: list[str], b: list[str]) -> int:
    """Banerjee & Pedersen overlap: repeatedly take the longest common contiguous phrase,
    add len(phrase)^2, remove it from both sides, until nothing is shared."""
    a, b = list(a), list(b)
    score = 0
    marker = 0
    while a and b:
        m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
        if m.size == 0:
            break
        score += m.size * m.size
        # Replace the matched phrase with unique markers so it can't match again.
        a[m.a:m.a + m.size] = [f"\x00a{marker}"]
        b[m.b:m.b + m.size] = [f"\x00b{marker}"]
        marker += 1
    return score


def ensure_wordnet() -> None:
    """Make sure NLTK's WordNet 3.0 corpus is available (downloads ~10 MB once if missing)."""
    import nltk

    try:
        from nltk.corpus import wordnet as wn

        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def _wordnet_synset(synset_id: str):
    """'n05174653' -> WordNet 3.0 synset (adjective ids may be satellites, 's')."""
    from nltk.corpus import wordnet as wn

    pos, offset = synset_id[0], synset_id[1:]
    if pos not in "nvar" or not offset.isdigit():
        return None
    for p in ([pos, "s"] if pos == "a" else [pos]):
        try:
            return wn.synset_from_pos_and_offset(p, int(offset))
        except Exception:  # noqa: BLE001 - unknown offset for this POS
            continue
    return None


_RELATIONS = (
    "hypernyms", "instance_hypernyms", "hyponyms", "instance_hyponyms",
    "member_holonyms", "part_holonyms", "substance_holonyms",
    "member_meronyms", "part_meronyms", "substance_meronyms",
    "also_sees", "similar_tos", "attributes", "entailments", "causes",
)


# Very general senses (e.g. "person") have hundreds of hyponyms; including all their glosses
# makes them overlap with almost anything. Cap the senses taken from each relation.
MAX_PER_RELATION = 25


def extended_gloss(synset_id: str, stored_gloss: str | None) -> list[str]:
    """Tokens of the sense's gloss plus the glosses of its WordNet neighbours
    (at most MAX_PER_RELATION senses per relation)."""
    parts = [stored_gloss or ""]
    syn = _wordnet_synset(synset_id)
    if syn is not None:
        parts.append(syn.definition())
        parts.extend(syn.examples()[:1])
        for rel in _RELATIONS:
            for other in getattr(syn, rel)()[:MAX_PER_RELATION]:
                parts.append(other.definition())
    return tokens(" ".join(parts))


@dataclass
class LeskIndex:
    """Extended glosses for every sense in the graph plus chunk -> senses."""

    chunk_senses: dict[str, set[str]]
    glosses: dict[str, list[str]]
    df: Counter = field(default_factory=Counter)
    postings: dict[str, set[str]] = field(default_factory=dict)  # word -> senses using it

    @classmethod
    def build(cls, rows: list[tuple[str, str, str | None]]) -> LeskIndex:
        """rows: (chunk_id, synset_id, stored gloss) for every MENTIONS edge."""
        chunk_senses: dict[str, set[str]] = defaultdict(set)
        stored: dict[str, str | None] = {}
        for chunk_id, synset_id, gloss in rows:
            chunk_senses[chunk_id].add(synset_id)
            stored.setdefault(synset_id, gloss)
        glosses = {sid: extended_gloss(sid, g) for sid, g in stored.items()}
        df = Counter(s for senses in chunk_senses.values() for s in senses)
        postings: dict[str, set[str]] = defaultdict(set)
        for sid, toks in glosses.items():
            for w in set(toks):
                postings[w].add(sid)
        return cls(dict(chunk_senses), glosses, df, dict(postings))

    def related_senses(self, query_id: str, query_gloss: list[str], candidates: int) -> dict[str, float]:
        """Relatedness in [0, 1] of the query sense to corpus senses (top `candidates`).
        The identical sense scores 1.0; others are scaled by the best non-identical overlap."""
        q_words = Counter(query_gloss)
        shared: Counter = Counter()
        for w in q_words:
            for sid in self.postings.get(w, ()):
                shared[sid] += 1
        shared.pop(query_id, None)
        scores = {}
        for sid, _ in shared.most_common(candidates):
            s = phrasal_overlap(query_gloss, self.glosses[sid])
            if s > 0:
                scores[sid] = float(s)
        best = max(scores.values(), default=0.0)
        related = {sid: s / best for sid, s in scores.items()} if best else {}
        if query_id in self.glosses:
            related[query_id] = 1.0
        return related

    def rank(self, query_ids: list[str], top_k: int, candidates: int = 100,
             related_weight: float = 0.8) -> list[tuple[str, float, int]]:
        """Return [(chunk_id, score, direct_matches)] best first.

        score = sum over query senses q of idf(q) * max over chunk senses s of rel(q, s), where
        rel = 1 for the identical sense and related_weight * scaled overlap otherwise;
        normalised by the total idf so a chunk mentioning every query sense scores 1.0.
        """
        n = max(1, len(self.chunk_senses))
        idf = {q: math.log(1 + n / max(1, self.df.get(q, 0))) for q in query_ids}
        total = sum(idf.values()) or 1.0
        rel_by_q = {}
        for q in query_ids:
            q_gloss = self.glosses.get(q) or extended_gloss(q, None)
            rel = self.related_senses(q, q_gloss, candidates)
            rel_by_q[q] = {s: (1.0 if s == q else related_weight * r) for s, r in rel.items()}

        scored = []
        for chunk_id, senses in self.chunk_senses.items():
            score, coverage, direct = 0.0, 0.0, 0
            for q in query_ids:
                rel = rel_by_q[q]
                hits = [rel[s] for s in senses if s in rel]
                if hits:
                    score += idf[q] * max(hits)
                    coverage += sum(hits)
                direct += q in senses
            if score > 0:
                scored.append((chunk_id, score / total, coverage, direct))
        scored.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return [(c, round(s, 6), d) for c, s, _, d in scored[:top_k]]

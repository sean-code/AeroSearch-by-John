# src/bm25.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Set
from .index import InvertedIndex
from .preprocessing import word_tokenize, STOPWORDS

class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self.index = index
        self.k1, self.b = k1, b
        self.N = index.num_docs
        self.df: Dict[str, int] = {t: len(pl) for t, pl in index.postings.items()}
        self.avgdl = (sum(index.doc_len.values()) / self.N) if self.N else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_text: str, candidate_docs: List[int] | None = None) -> List[Tuple[int, float]]:
        q_terms = [t for t in word_tokenize(query_text) if t not in STOPWORDS]
        if not q_terms: return []
        cand: Set[int] = set()
        for t in q_terms:
            for d, _ in self.index.get_postings(t):
                cand.add(d)
        if candidate_docs is not None:
            cand &= set(candidate_docs)

        scores: Dict[int, float] = {}
        for d in cand:
            dl = self.index.doc_len[d]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            s = 0.0
            for t in set(q_terms):
                plist = dict(self.index.get_postings(t))
                tf = len(plist[d]) if d in plist else 0
                if tf == 0: continue
                s += self.idf(t) * (tf * (self.k1 + 1)) / (tf + denom_norm)
            if s: scores[d] = s
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

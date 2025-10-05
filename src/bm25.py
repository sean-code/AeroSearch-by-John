# BM- 25 Algorithm: - To enable us rank our retrived docs
import math
from typing import List, Dict, Tuple
from collections import Counter
from .index import InvertedIndex
from .preprocessing import word_tokenize, DEFAULT_STOPWORDS

class BM25:
    def __init__(self, index: InvertedIndex, k1: float=1.2, b: float=0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.N = index.num_docs
        self._df: Dict[str, int] = {t: len(pl) for t, pl in index.postings.items()}
        self._avgdl = sum(index.doc_len.values())/self.N if self.N>0 else 0.0

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        # BM25+ style idf (with +0.5 smoothing)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score_query(self, query_text: str, candidate_docs: List[int]=None, stopwords=DEFAULT_STOPWORDS) -> List[Tuple[int, float]]:
        q_terms = [t for t in word_tokenize(query_text) if t not in stopwords]
        if not q_terms: return []
        # Candidate set: union of postings for query terms (if none provided)
        cand = set()
        for t in q_terms:
            for d,_ in self.index.get_postings(t):
                cand.add(d)
        if candidate_docs is not None:
            cand = cand & set(candidate_docs)
        scores: Dict[int, float] = {}
        q_counts = Counter(q_terms)
        for d in cand:
            dl = self.index.doc_len[d]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self._avgdl+1e-9)))
            s = 0.0
            # need term frequency in doc: can derive from postings
            tf_cache = {}
            for t in set(q_terms):
                if t not in self.index.postings: 
                    continue
                # get tf for doc d
                if t not in tf_cache:
                    # postings are (doc_id, [positions]); tf = len(positions)
                    plist = dict(self.index.postings[t])
                    tf_cache[t] = len(plist[d]) if d in plist else 0
                tf = tf_cache[t]
                if tf == 0: 
                    continue
                idf = self.idf(t)
                s += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
            if s != 0.0:
                scores[d] = s
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

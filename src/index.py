from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict
from .preprocessing import preprocess

Posting = Tuple[int, List[int]]  # (doc_id, and the [positions])

class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_len: Dict[int, int] = {}
        self.num_docs: int = 0
        self._sent_bounds: Dict[int, List[Tuple[int,int]]] = {}

    def build(self, docs: Iterable[Tuple[int, str]]) -> None:
        """
        docs: iterable of (doc_id, text)
        """
        for doc_id, text in docs:
            flat, sent_tokens = preprocess(text)
            self.doc_len[doc_id] = len(flat)
            self.num_docs += 1

            # sentence boundaries in token-position space (for /s)
            bounds, pos_cur = [], 0
            for s in sent_tokens:
                if s:
                    start, end = pos_cur, pos_cur + len(s) - 1
                    bounds.append((start, end))
                    pos_cur += len(s)
            self._sent_bounds[doc_id] = bounds

            # positional postings
            tmp = defaultdict(list)
            for i, tok in enumerate(flat):
                tmp[tok].append(i)
            for term, pos_list in tmp.items():
                self.postings[term].append((doc_id, pos_list))

        # postings sorted by doc_id
        for t in list(self.postings.keys()):
            self.postings[t].sort(key=lambda x: x[0])

    def get_postings(self, term: str) -> List[Posting]:
        return self.postings.get(term, [])

    def sent_bounds(self, doc_id: int) -> List[Tuple[int,int]]:
        return self._sent_bounds.get(doc_id, [])

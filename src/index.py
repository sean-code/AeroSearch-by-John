from typing import Dict, List, Tuple, Iterable
from collections import defaultdict
from .preprocessing import preprocess

Posting = Tuple[int, List[int]]  # (doc_id, [positions])

class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.doc_len: Dict[int, int] = {}
        self.num_docs = 0
        self._sent_boundaries: Dict[int, List[Tuple[int,int]]] = {}  # per doc: (start_pos, end_pos) per sentence

    def build(self, docs: Iterable[Tuple[int, str]]):
        """
        docs: iterable of (doc_id, text)
        """
        for doc_id, text in docs:
            flat, sent_tokens = preprocess(text)
            self.doc_len[doc_id] = len(flat)
            self.num_docs += 1

            # record sentence boundaries in token-position space for /s
            boundaries = []
            pos_cursor = 0
            for s in sent_tokens:
                start, end = pos_cursor, pos_cursor + len(s) - 1
                boundaries.append((start, end))
                pos_cursor += len(s)
            self._sent_boundaries[doc_id] = boundaries

            # add positional postings
            pos = 0
            positions_by_term = defaultdict(list)
            for tok in flat:
                positions_by_term[tok].append(pos)
                pos += 1
            for term, pos_list in positions_by_term.items():
                self.postings[term].append((doc_id, pos_list))

        # sort postings by doc_id to keep merges linear
        for term in list(self.postings.keys()):
            self.postings[term].sort(key=lambda x: x[0])

    def get_postings(self, term: str) -> List[Posting]:
        return self.postings.get(term, [])

    def sentence_boundaries(self, doc_id: int) -> List[Tuple[int,int]]:
        return self._sent_boundaries.get(doc_id, [])

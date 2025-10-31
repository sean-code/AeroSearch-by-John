from __future__ import annotations
from typing import List
import math

def dcg(rels: List[int], k: int) -> float:
    return sum((2**rels[i]-1) / math.log2(i+2) for i in range(min(k, len(rels))))

def ndcg_at_k(rels: List[int], k: int) -> float:
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0

def average_precision(binary_rels: List[int]) -> float:
    hits, s = 0, 0.0
    for i, r in enumerate(binary_rels, start=1):
        if r:
            hits += 1
            s += hits / i
    return s / hits if hits else 0.0

def recall_at_k(binary_rels: List[int], total_relevant: int, k: int) -> float:
    return sum(binary_rels[:k]) / max(total_relevant, 1)

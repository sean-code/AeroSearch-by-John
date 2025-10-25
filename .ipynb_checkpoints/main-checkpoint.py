# main.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25
from src.eval import ndcg_at_k, average_precision, recall_at_k

DATA_PATH = Path("data/sample_docs.jsonl")  # each line: {"id": 1, "text": "..."}
K = 10  # cutoff for metrics display

def load_jsonl(path: Path) -> Dict[int, str]:
    docs: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            docs[int(row["id"])] = (row.get("text") or "").strip()
    return docs

def eval_with_labels(run: List[int], qid: str, labels: Dict[str, Dict[int, int]]):
    """labels format: {qid: {doc_id: 1/0 or graded (0/1/2)}}"""
    rel_map = labels.get(qid, {})
    rels = [rel_map.get(doc_id, 0) for doc_id in run]  # align to run order
    bin_rels = [1 if r > 0 else 0 for r in rels]
    total_rel = sum(1 for v in rel_map.values() if v > 0)
    return {
        "nDCG@10": round(ndcg_at_k(rels, 10), 4),
        "MAP": round(average_precision(bin_rels), 4),
        "R@10": round(recall_at_k(bin_rels, total_rel, 10), 4),
        "TotalRel": total_rel,
    }

def main():
    if not DATA_PATH.exists():
        # Tiny in-memory fallback
        docs = {
            1: "Night visual approach with unexpected tailwind led to long landing and runway excursion.",
            2: "Climb-out icing after takeoff; pitot heat oversight led to airspeed discrepancies; returned to land.",
            3: "Taxiway incursion due to similar call signs; readback-hearback breakdown with ATC.",
            4: "RNAV GPS approach in mountainous terrain; false glidepath capture; CFIT risk mitigated by go-around.",
            5: "Landing with crosswind gusts; unstable short final; veer-off avoided by go-around.",
        }
        print("[i] Using built-in demo docs (create data/sample_docs.jsonl to override).")
    else:
        docs = load_jsonl(DATA_PATH)
        print(f"[i] Loaded {len(docs)} docs from {DATA_PATH}")

    # Build index
    idx = InvertedIndex()
    idx.build(docs.items())
    bm25 = BM25(idx)

    # === Boolean search smoke tests ===
    q1 = '("runway excursion" OR overrun) AND landing'
    q2 = '(icing /5 pitot) AND (airspeed OR indications)'
    q3 = '(incursion AND taxiway) AND (atc OR "readback hearback")'
    q4 = '(approach AND mountainous) /s (cfit OR terrain)'
    for q in [q1, q2, q3, q4]:
        matches = evaluate_query(idx, q)
        print(f"\n[Boolean] {q}\n -> doc_ids: {matches}")

    # === BM25 ranked retrieval (query-by-narrative text) ===
    query_text = "unstable approach with tailwind, long landing and excursion off runway"
    ranking = bm25.score(query_text)
    print(f"\n[BM25] Query: {query_text}")
    for d, s in ranking[:K]:
        print(f"  doc {d:>3}  score={s:.3f}  :: {docs[d][:90]}")

    # === (Optional) Evaluate with small hand labels ===
    # Put your labels here when ready (qid -> {doc_id: gain}), e.g., 0/1/2 graded relevance:
    labels = {
        "Q-runway": {1: 2, 5: 1},          # example graded relevance for the runway excursion theme
        "Q-icing": {2: 2},                 # icing/pitot theme
    }

    # Evaluate Boolean as candidate set + BM25 re-rank (common pipeline)
    bool_docs = evaluate_query(idx, q1)                 # candidate set for "runway excursion"
    reranked = bm25.score(query_text, candidate_docs=bool_docs)
    run = [d for d, _ in reranked[:K]]
    print("\n[Eval] Boolean→BM25 for 'Q-runway':")
    print(eval_with_labels(run, "Q-runway", labels))

if __name__ == "__main__":
    main()

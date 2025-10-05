from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

DOCS = {
    1: "Night visual approach with unexpected tailwind led to long landing and runway excursion.",
    2: "Climb-out icing after takeoff; pitot heat oversight led to airspeed discrepancies; returned to land.",
    3: "Taxiway incursion due to similar call signs; readback-hearback breakdown with ATC.",
    4: "RNAV GPS approach in mountainous terrain; false glidepath capture; CFIT risk mitigated by go-around.",
    5: "Landing with crosswind gusts; unstable short final; veer-off avoided by go-around.",
}

def main():
    idx = InvertedIndex()
    idx.build(DOCS.items())

    # Boolean queries
    q1 = '("runway excursion" OR overrun) AND landing'
    print("Boolean:", q1, "=>", evaluate_query(idx, q1))  # expect doc 1

    q2 = '(icing /5 pitot) AND (airspeed OR indications)'
    print("Boolean:", q2, "=>", evaluate_query(idx, q2))  # expect doc 2

    q3 = '(incursion AND taxiway) AND (atc OR "readback hearback")'
    print("Boolean:", q3, "=>", evaluate_query(idx, q3))  # expect doc 3

    q4 = '(approach AND mountainous) /s (cfit OR terrain)'
    print("Boolean:", q4, "=>", evaluate_query(idx, q4))  # expect doc 4

    # BM25 ranking (query-by-narrative)
    bm25 = BM25(idx)
    query_text = "unstable approach with tailwind, long landing and excursion off runway"
    scores = bm25.score_query(query_text)
    print("\nBM25 ranking for:", query_text)
    for d, s in scores[:5]:
        print(f"  doc {d}: {s:.3f}  :: {DOCS[d]}")

if __name__ == "__main__":
    main()

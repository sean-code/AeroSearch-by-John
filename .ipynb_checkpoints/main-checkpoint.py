# main.py
from __future__ import annotations
import csv, json, html, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25
from src.eval import ndcg_at_k, average_precision, recall_at_k

# -----------------------
# Data loaders
# -----------------------
def load_ntsb_csv(
    path: Path,
    text_col: str = "ProbableCause",
    keep_meta_cols: List[str] | None = None,
    limit: int | None = None
) -> Tuple[Dict[int, str], Dict[int, dict]]:
    if keep_meta_cols is None:
        keep_meta_cols = [
            "ReportNo","EventDate","City","State","Country",
            "Make","Model","AirCraftCategory","EventType","AirportID","AirportName",
            "WeatherCondition","HighestInjuryLevel","AirCraftDamage","DocketUrl",
            "Latitude","Longitude",
        ]
    docs: Dict[int, str] = {}
    meta: Dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue
            i += 1
            docs[i] = text
            m = {k: row.get(k) for k in keep_meta_cols}
            # cast lat/lon if present
            for k in ("Latitude", "Longitude"):
                if m.get(k) not in (None, ""):
                    try:
                        m[k] = float(m[k])
                    except Exception:
                        pass
            # stable id if available
            for stable_id in ("ReportNo","EventID","NtsbNo"):
                if row.get(stable_id):
                    m["orig_id"] = row[stable_id]
                    break
            meta[i] = m
            if limit and i >= limit:
                break
    return docs, meta

def load_ntsb_json(
    path: Path,
    text_fields: List[str] | None = None,
    keep_meta_cols: List[str] | None = None,
    limit: int | None = None,
) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """
    Loads a regular JSON file containing a list of records (array or object with list under items/data/results/Records).
    Concatenates any non-empty narrative fields into 'text'.
    """
    if text_fields is None:
        # We concatenate ANY of these that are non-empty
        text_fields = ["ProbableCause","AnalysisNarrative","FactualNarrative","PrelimNarrative"]
    if keep_meta_cols is None:
        keep_meta_cols = [
            "NtsbNumber","ReportNo","EventDate","City","State","Country",
            "AirportId","AirportName","EventType","HighestInjury","AccidentSiteCondition",
            "Latitude","Longitude",
        ]

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        for k in ("items","data","results","Records"):
            if isinstance(raw.get(k), list):
                records = raw[k]
                break
        else:
            records = list(raw.values())
    else:
        records = []

    docs: Dict[int, str] = {}
    meta: Dict[int, dict] = {}
    n = 0

    for r in records:
        # concatenate any narrative fields that are present and non-empty
        chunks: List[str] = []
        for tf in text_fields:
            val = r.get(tf)
            if isinstance(val, str) and val.strip():
                cleaned = html.unescape(val).replace("\r\n", " ").replace("\n", " ").strip()
                if cleaned:
                    chunks.append(cleaned)
        text = "  ".join(chunks).strip()
        if not text:
            continue

        n += 1
        docs[n] = text

        # metadata
        m: Dict[str, Any] = {k: r.get(k) for k in keep_meta_cols}
        # cast lat/lon to floats when possible
        for k in ("Latitude","Longitude"):
            if m.get(k) not in (None, ""):
                try:
                    m[k] = float(m[k])
                except Exception:
                    pass
        # prefer a stable id if available
        for stable in ("NtsbNumber","ReportNo","EventID","MKey","Oid"):
            v = r.get(stable)
            if v not in (None, ""):
                m["orig_id"] = v
                break

        # Vehicle info (Make/Model etc.) – take first vehicle if present
        vlist = r.get("Vehicles") or r.get("vehicles")
        if isinstance(vlist, list) and vlist:
            v0 = vlist[0]
            for k in ("Make","Model","AircraftCategory","DamageLevel","RegulationFlightConductedUnder"):
                if v0.get(k) not in (None, ""):
                    m[k] = v0[k]

        meta[n] = m
        if limit and n >= limit:
            break

    return docs, meta

# -----------------------
# Pretty printing
# -----------------------
def show_results(
    docs: Dict[int, str],
    meta: Dict[int, dict],
    ranked,  # List[Tuple[int,float]] | List[int]
    topk: int = 10,
    with_scores: bool = True
) -> None:
    print()
    for idx, item in enumerate(ranked[:topk], start=1):
        if isinstance(item, tuple):
            d, s = item
        else:
            d, s = item, None
        m = meta.get(d, {})
        title_bits = [
            m.get("EventDate"), m.get("City"), m.get("State"),
            m.get("Make"), m.get("Model"), m.get("EventType")
        ]
        title = " | ".join([x for x in title_bits if x])
        print(f"{idx:>2}. doc {d}", end="")
        if with_scores and s is not None:
            print(f"  score={s:.3f}", end="")
        print()
        if title:
            print(f"    {title}")

        # Coordinates if available
        lat = m.get("Latitude"); lon = m.get("Longitude")
        if lat is not None and lon is not None:
            print(f"    Lat: {lat}, Lon: {lon}")

        text = docs[d]
        snippet = (text[:240] + "…") if len(text) > 240 else text
        print(f"    {snippet}")
        if m.get("DocketUrl"):
            print(f"    Docket: {m['DocketUrl']}")
        if m.get("orig_id"):
            print(f"    ID: {m['orig_id']}")
        print()

# -----------------------
# REPL
# -----------------------
HELP = """
Type queries and press Enter.

- BM25 ranked search: just type your query text.
- Boolean/proximity search: prefix with  'bool: '  then use operators.
  Examples:
    bool: ("runway excursion" OR overrun) AND landing
    bool: (icing /5 pitot) AND (airspeed OR indications)
    bool: (approach AND mountainous) /s (cfit OR terrain)

Commands:
  :help             show this help
  :quit             exit
  :topk 20          set display cutoff
"""

def repl(idx: InvertedIndex, docs: Dict[int, str], meta: Dict[int, dict]) -> None:
    # If you've added speed-ups in BM25/index, you can pass params here (e.g., idf_prune, default_msm)
    bm25 = BM25(idx)
    topk = 10
    print(HELP)
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return
        if not q:
            continue
        if q == ":quit":
            print("[bye]"); return
        if q == ":help":
            print(HELP); continue
        if q.startswith(":topk"):
            try:
                topk = int(q.split()[1])
                print(f"[i] topk={topk}")
            except Exception:
                print("[!] usage: :topk <int>")
            continue

        if q.lower().startswith("bool:"):
            bq = q[5:].strip()
            doc_ids = evaluate_query(idx, bq)
            print(f"[i] Boolean matched {len(doc_ids)} docs. Showing first {topk}.")
            show_results(docs, meta, doc_ids, topk=topk, with_scores=False)
            if doc_ids:
                reranked = bm25.score(bq, candidate_docs=doc_ids)
                print(f"[i] BM25 re-rank of Boolean candidates:")
                show_results(docs, meta, reranked, topk=topk, with_scores=True)
            continue

        ranked = bm25.score(q)
        print(f"[i] BM25 returned {len(ranked)} candidates. Showing top {topk}.")
        show_results(docs, meta, ranked, topk=topk, with_scores=True)

# -----------------------
# Eval helper (optional)
# -----------------------
def eval_with_labels(run_ids: List[int], rel_map: Dict[int, int]) -> Dict[str, float]:
    rels = [rel_map.get(d, 0) for d in run_ids]
    bin_rels = [1 if r > 0 else 0 for r in rels]
    total_rel = sum(1 for v in rel_map.values() if v > 0)
    return {
        "nDCG@10": round(ndcg_at_k(rels, 10), 4),
        "MAP": round(average_precision(bin_rels), 4),
        "R@10": round(recall_at_k(bin_rels, total_rel, 10), 4),
        "TotalRel": total_rel,
    }

# -----------------------
# Entrypoint
# -----------------------
def main():
    """
    Usage:
      1) Auto-detect local JSON demo:   python main.py
      2) Load NTSB JSON array dump:     python main.py data.json
      3) Load NTSB CSV:                 python main.py ntsb.csv ProbableCause
    """
    args = sys.argv[1:]

    if not args:
        # Prefer your JSON dump if present, else in-memory demo
        demo_json = Path("data/sample_docs1.json")
        if demo_json.exists():
            docs, meta = load_ntsb_json(demo_json)
            print(f"[i] Loaded {len(docs)} docs from {demo_json}")
        else:
            # Sample Documents indexed to avoid having an empty corpora
            docs = {
                1: "Night visual approach with unexpected tailwind led to long landing and runway excursion.",
                2: "Climb-out icing after takeoff; pitot heat oversight led to airspeed discrepancies; returned to land.",
                3: "Taxiway incursion due to similar call signs; readback-hearback breakdown with ATC.",
                4: "RNAV GPS approach in mountainous terrain; false glidepath capture; CFIT risk mitigated by go-around.",
                5: "Landing with crosswind gusts; unstable short final; veer-off avoided by go-around.",
            }
            meta = {i: {} for i in docs}
            print("[i] Using in-memory demo docs.")
    else:
        path = Path(args[0])
        if not path.exists():
            print(f"[!] File not found: {path}")
            sys.exit(1)
        ext = path.suffix.lower()
        if ext == ".json":
            docs, meta = load_ntsb_json(path)
            print(f"[i] Loaded {len(docs)} docs from {path}")
        else:
            text_col = args[1] if len(args) >= 2 else "ProbableCause"
            docs, meta = load_ntsb_csv(path, text_col=text_col)
            print(f"[i] Loaded {len(docs)} docs from {path} (text_col='{text_col}')")

    idx = InvertedIndex()
    idx.build(docs.items())
    print(f"[i] Built index over {idx.num_docs} documents.")
    repl(idx, docs, meta)

if __name__ == "__main__":
    main()
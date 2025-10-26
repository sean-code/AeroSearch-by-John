# main.py
from __future__ import annotations
import csv, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

# -----------------------
# Data loaders
# -----------------------

PREFERRED_TEXT_FIELDS = [
    "ProbableCause",            # NTSB final report narrative (often concise)
    "AnalysisNarrative",        # richer narrative if present
    "Narrative",                # catch-all
    "PrelimNarrative",
    "FactualNarrative",
    "text"                      # already-normalized jsonl
]

META_KEEP = [
    "ReportNo","NtsbNumber","EventID","MKey","Oid",
    "EventDate","City","State","Country",
    "Make","Model","AirCraftCategory","EventType",
    "AirportID","AirportId","AirportName",
    "WeatherCondition","HighestInjuryLevel","AirCraftDamage",
    "DocketUrl","DocketURL","DocketOriginalPublishDate","OriginalPublishedDate"
]

def pick_text(record: Dict[str, Any]) -> str:
    """Choose the best available narrative field."""
    for k in PREFERRED_TEXT_FIELDS:
        if record.get(k):
            t = (record[k] or "").strip()
            if t:
                return t
    # If none, try to stitch a couple of fields
    t = " ".join([(record.get("ProbableCause") or ""), (record.get("AnalysisNarrative") or "")]).strip()
    return t

def load_jsonl_generic(path: Path) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """
    Accepts NTSB-style JSONL lines with arbitrary keys.
    Produces:
      docs: {int_id -> text}
      meta: {int_id -> selected metadata}
    """
    docs: Dict[int, str] = {}
    meta: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line: 
                continue
            rec = json.loads(line)
            text = pick_text(rec)
            if not text:
                continue
            i += 1
            docs[i] = text
            m = {k: rec.get(k) for k in META_KEEP if k in rec}
            # keep an original id if any of these exist
            for k in ("ReportNo","NtsbNumber","EventID","Oid","MKey"):
                if rec.get(k):
                    m["orig_id"] = rec[k]
                    break
            meta[i] = m
    return docs, meta

def load_ntsb_csv(
    path: Path,
    text_col: str = "ProbableCause",
    keep_meta_cols: List[str] | None = None,
    limit: int | None = None
) -> Tuple[Dict[int, str], Dict[int, dict]]:
    """CSV loader for NTSB export."""
    if keep_meta_cols is None:
        keep_meta_cols = META_KEEP
    docs: Dict[int, str] = {}
    meta: Dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            # Try preferred fields first, then fallback to requested text_col
            text = pick_text(row) or (row.get(text_col) or "").strip()
            if not text:
                continue
            i += 1
            docs[i] = text
            meta[i] = {k: row.get(k) for k in keep_meta_cols if k in row}
            for stable_id in ("ReportNo","NtsbNumber","EventID","Oid","MKey"):
                if row.get(stable_id):
                    meta[i]["orig_id"] = row[stable_id]
                    break
            if limit and i >= limit:
                break
    return docs, meta

def csv_to_jsonl(
    csv_path: Path,
    jsonl_path: Path,
    text_col: str = "ProbableCause",
    limit: int | None = None,
) -> int:
    """Convert CSV -> normalized JSONL: each line has id, text, meta."""
    docs, meta = load_ntsb_csv(csv_path, text_col=text_col, limit=limit)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as out:
        for did, text in docs.items():
            row = {"id": did, "text": text, "meta": meta.get(did, {})}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(docs)

# -----------------------
# Pretty printing
# -----------------------
def show_results(
    docs: Dict[int, str],
    meta: Dict[int, dict],
    ranked,
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
            m.get("Make"), m.get("Model"), m.get("EventType"),
            m.get("orig_id"),
        ]
        title = " | ".join([x for x in title_bits if x])
        print(f"{idx:>2}. doc {d}", end="")
        if with_scores and s is not None:
            print(f"  score={s:.3f}", end="")
        print()
        if title:
            print(f"    {title}")
        text = docs[d]
        snippet = (text[:240] + "…") if len(text) > 240 else text
        print(f"    {snippet}")
        if m.get("DocketUrl") or m.get("DocketURL"):
            print(f"    Docket: {m.get('DocketUrl') or m.get('DocketURL')}")
        print()

# -----------------------
# REPL
# -----------------------
HELP = """
Type queries and press Enter.

- BM25: type your query (free text)
- Boolean/proximity: prefix with 'bool: '

Examples:
  bool: ("runway excursion" OR overrun) AND landing
  bool: (icing /5 pitot) AND (airspeed OR indications)
  bool: (approach AND mountainous) /s (cfit OR terrain)

Commands:
  :topk 20    set results cutoff
  :quit       exit
"""

def repl(idx: InvertedIndex, docs: Dict[int, str], meta: Dict[int, dict]) -> None:
    bm25 = BM25(idx)
    topk = 10
    print(HELP)
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]"); return
        if not q: 
            continue
        if q == ":quit":
            print("[bye]"); return
        if q.startswith(":topk"):
            try:
                topk = int(q.split()[1]); print(f"[i] topk={topk}")
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
        else:
            ranked = bm25.score(q)
            print(f"[i] BM25 returned {len(ranked)} candidates. Showing top {topk}.")
            show_results(docs, meta, ranked, topk=topk, with_scores=True)

# -----------------------
# Entrypoint
# -----------------------
def main():
    """
    Usage:
      1) Auto-convert CSV then search:
         python main.py data/ntsb.csv           # converts → data/ntsb.jsonl and loads it

      2) Load existing JSONL and search:
         python main.py data/sample_docs.jsonl  # or your NTSB jsonl export

      3) Explicit convert target:
         python main.py data/ntsb.csv data/ntsb.jsonl
    """
    args = sys.argv[1:]

    # Paths
    default_jsonl = Path("data/sample_docs.jsonl")

    # Case A: CSV given → convert to JSONL and then load
    if args and args[0].lower().endswith(".csv"):
        csv_path = Path(args[0])
        out_jsonl = Path(args[1]) if len(args) >= 2 else Path("data/ntsb.jsonl")
        if not csv_path.exists():
            print(f"[!] CSV not found: {csv_path}"); sys.exit(1)
        n = csv_to_jsonl(csv_path, out_jsonl, text_col="ProbableCause")
        print(f"[i] Converted CSV → JSONL: {n} rows → {out_jsonl}")
        jsonl_path = out_jsonl

    # Case B: JSONL path provided
    elif args and args[0].lower().endswith(".jsonl"):
        jsonl_path = Path(args[0])
        if not jsonl_path.exists():
            print(f"[!] JSONL not found: {jsonl_path}"); sys.exit(1)

    # Case C: no args → use default JSONL
    else:
        jsonl_path = default_jsonl
        if not jsonl_path.exists():
            print(f"[!] Expected {jsonl_path} to exist. Provide a CSV or JSONL path.\n"
                  f"    e.g., python main.py data/ntsb.csv   OR   python main.py data/sample_docs.jsonl")
            sys.exit(1)

    # Load JSONL and build index
    docs, meta = load_jsonl_generic(jsonl_path)
    print(f"[i] Loaded {len(docs)} docs from {jsonl_path}")
    if not docs:
        print("[!] No documents with usable narrative text were found."); sys.exit(1)

    idx = InvertedIndex()
    idx.build(docs.items())
    print(f"[i] Built index over {idx.num_docs} documents.")

    # Start search loop
    repl(idx, docs, meta)

if __name__ == "__main__":
    main()
    
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import streamlit as st
from datetime import datetime

# --- project modules ---
from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

# ---------- Settings ----------
NARRATIVE_FIELDS = ["ProbableCause","AnalysisNarrative","FactualNarrative","PrelimNarrative"]
META_FIELDS = [
    "NtsbNumber","ReportNo","EventDate","City","State","Country",
    "AirportId","AirportName","EventType","HighestInjury","AccidentSiteCondition",
    "Latitude","Longitude",
    "FatalInjuryCount","SeriousInjuryCount","MinorInjuryCount",
]
DEFAULT_JSON = "data/sample_docs1.json"  # fixed corpus path

# ---------- Helpers ----------
def _concat_narratives(rec: dict) -> str:
    parts: List[str] = []
    for f in NARRATIVE_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            v = v.replace("\r\n", " ").replace("\n", " ").strip()
            parts.append(v)
    return "  ".join(parts).strip()

def _safe_float(x) -> Optional[float]:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except Exception:
        return None

def _event_year(s: Any) -> Optional[int]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z","")).year
    except Exception:
        ss = str(s)
        if len(ss) >= 4 and ss[:4].isdigit():
            return int(ss[:4])
        return None

@st.cache_data(show_spinner=False)
def load_json_records(path: str, limit: int | None = None) -> Tuple[Dict[int, str], Dict[int, dict]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k in ("items","data","results","Records"):
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
        else:
            raw = list(raw.values())
    assert isinstance(raw, list), "JSON must be an array or a dict that contains a list"

    docs: Dict[int, str] = {}
    meta: Dict[int, dict] = {}
    n = 0
    for r in raw:
        text = _concat_narratives(r)
        if not text:
            continue
        n += 1
        docs[n] = text
        m = {k: r.get(k) for k in META_FIELDS}

        # stable id preference
        for stable in ("NtsbNumber","ReportNo","EventID","MKey","Oid"):
            v = r.get(stable)
            if v not in (None, ""):
                m["orig_id"] = v
                break

        # lat/lon
        m["Latitude"] = _safe_float(m.get("Latitude"))
        m["Longitude"] = _safe_float(m.get("Longitude"))

        # Vehicles[0]
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

@st.cache_resource(show_spinner=False)
def build_index(docs: Dict[int, str]) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(docs.items())
    return idx

# ---------- UI Config ----------
st.set_page_config(page_title="AeroSearch", page_icon="✈️", layout="wide")

# Global CSS: sticky right pane, scrollable results
st.markdown("""
<style>
.results-pane {
  max-height: 70vh;
  overflow-y: auto;
  padding-right: .5rem;
}
.sticky-pane {
  position: sticky;
  top: 6rem;
}
.header-tight p { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

st.title("AeroSearch — Aviation Incident/Accident Narrative Retrieval")

# ---------- Data bootstrapping ----------
if "docs_cache" not in st.session_state:
    with st.status("Loading dataset and building index…", expanded=True) as status:
        st.write("Reading JSON …")
        docs, meta = load_json_records(DEFAULT_JSON, limit=None)
        st.write(f"Loaded {len(docs):,} records.")
        st.write("Building inverted index …")
        idx = build_index(docs)
        st.write(f"Indexed {idx.num_docs:,} documents.")
        st.session_state["docs_cache"] = docs
        st.session_state["meta_cache"] = meta
        st.session_state["index_cache"] = idx
        # persistent UI state
        st.session_state["detail_doc"] = None
        st.session_state["last_hits"] = None
        st.session_state["last_mode"] = None  # "bm25" | "bool" | "bool_rerank"
        st.session_state["last_topk"] = 10
        status.update(label="Ready", state="complete", expanded=False)

docs: Dict[int, str] = st.session_state["docs_cache"]
meta: Dict[int, dict] = st.session_state["meta_cache"]
idx: InvertedIndex = st.session_state["index_cache"]
bm25 = BM25(idx)

# Summary banner
with st.spinner("Preparing summary…"):
    years = [y for y in (_event_year(m.get("EventDate")) for m in meta.values()) if y is not None]
    min_y = min(years) if years else "—"
    max_y = max(years) if years else "—"
st.markdown(
    f"<div class='header-tight'><p><strong>Contains {len(docs):,} indexed narratives between {min_y} and {max_y} as reported by NTSB.</strong></p></div>",
    unsafe_allow_html=True
)

# ---------- Search Controls ----------
tabs = st.tabs(["Search", "Advanced (Boolean)"])

with tabs[0]:
    col_left, col_right = st.columns([2,1], vertical_alignment="center")
    with col_left:
        q_simple = st.text_input("Search", placeholder="Try: runway excursion landing", key="q_simple")
    with col_right:
        topk_simple = st.slider("Results", 5, 50, 10, key="topk_simple")
    run_simple = st.button("Run Search", type="primary", key="run_simple")

with tabs[1]:
    col_left_b, col_right_b = st.columns([2,1], vertical_alignment="center")
    with col_left_b:
        q_bool = st.text_input("Boolean query", placeholder="(icing /5 pitot) AND (airspeed OR indications)", key="q_bool")
    with col_right_b:
        topk_bool = st.slider("Results", 5, 50, 10, key="topk_bool")
    run_bool = st.button("Run Advanced Search", type="primary", key="run_bool")
    # Optional: separate button to run BM25 re-rank only when desired
    run_rerank = st.checkbox("Also re-rank Boolean matches with BM25", value=True, key="run_rerank")

st.divider()

# ---------- Persist results in session_state ----------
def _set_results(hits, mode: str, topk: int):
    st.session_state["last_hits"] = hits
    st.session_state["last_mode"] = mode
    st.session_state["last_topk"] = topk

# ---------- Renderers ----------
def header_line(m: dict) -> str:
    bits = [m.get("EventDate"), m.get("City"), m.get("State"), m.get("Country"),
            m.get("Make"), m.get("Model"), m.get("EventType")]
    return " | ".join([str(x) for x in bits if x])

def render_details_panel(detail_id: Optional[int]):
    st.markdown("<div class='sticky-pane'>", unsafe_allow_html=True)
    st.subheader("Case Details")
    if not detail_id:
        st.info("Select **Details** on a result to inspect full metadata.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    mm = meta.get(detail_id, {})
    tt = docs.get(detail_id, "")

    st.markdown(f"**doc {detail_id}**")
    st.write(header_line(mm))

    lat, lon = mm.get("Latitude"), mm.get("Longitude")
    if lat is not None and lon is not None:
        st.map(pd.DataFrame([{"latitude": lat, "longitude": lon}]))

    grid_left, grid_right = st.columns(2)
    with grid_left:
        st.markdown("**Location & Ops**")
        st.write(f"City: {mm.get('City') or '—'}")
        st.write(f"Country: {mm.get('Country') or '—'}")
        st.write(f"Airport: {mm.get('AirportName') or '—'} ({mm.get('AirportId') or '—'})")
        st.write(f"Event Type: {mm.get('EventType') or '—'}")
        st.write(f"Highest Injury: {mm.get('HighestInjury') or '—'}")
        st.write(f"Accident Site: {mm.get('AccidentSiteCondition') or '—'}")
        st.write(f"Latitude: {lat if lat is not None else '—'}")
        st.write(f"Longitude: {lon if lon is not None else '—'}")

    with grid_right:
        st.markdown("**Aircraft & IDs**")
        st.write(f"Make/Model: {(mm.get('Make') or '—')} {(mm.get('Model') or '')}".strip())
        st.write(f"Category: {mm.get('AircraftCategory') or '—'}")
        st.write(f"Damage Level: {mm.get('DamageLevel') or '—'}")
        st.write(f"NTSB Number: {mm.get('NtsbNumber') or '—'}")
        st.write(f"Report No: {mm.get('ReportNo') or '—'}")
        st.write(f"Fatal Injuries: {mm.get('FatalInjuryCount') or 0}")
        st.write(f"Serious Injuries: {mm.get('SeriousInjuryCount') or 0}")
        st.write(f"Minor Injuries: {mm.get('MinorInjuryCount') or 0}")
        if mm.get("DocketUrl"):
            st.link_button("Open Docket", mm["DocketUrl"])

    st.markdown("**Narrative**")
    st.write(tt)
    st.markdown("</div>", unsafe_allow_html=True)

def render_results_list(hits, topk: int, with_scores=True):
    res_col, det_col = st.columns([1.5, 1.0], gap="large")

    with res_col:
        st.markdown("<div class='results-pane'>", unsafe_allow_html=True)

        if not hits:
            st.warning("No results found. Try a broader query, or check your Boolean syntax.")
            st.markdown("</div>", unsafe_allow_html=True)
            with det_col:
                render_details_panel(st.session_state.get("detail_doc"))
            return

        for rank, item in enumerate(hits[:topk], start=1):
            d, s = (item if isinstance(item, tuple) else (item, None))
            m = meta.get(d, {})

            st.markdown(
                f"**{rank}. doc {d}**" + (f" &nbsp;&nbsp;`score={s:.3f}`" if (with_scores and s is not None) else "")
            )
            head = header_line(m)
            if head:
                st.write(head)

            lat, lon = m.get("Latitude"), m.get("Longitude")
            if lat is not None and lon is not None:
                st.caption(f"Lat: {lat}, Lon: {lon}")

            full_text = docs[d]
            short = full_text[:240] + ("…" if len(full_text) > 240 else "")
            st.write(short)
            with st.expander("Read more"):
                st.write(full_text)

            c1, c2, c3 = st.columns([0.25,0.25,0.5])
            with c1:
                if st.button("Details", key=f"det_{d}", use_container_width=True):
                    st.session_state["detail_doc"] = d
            with c2:
                if m.get("DocketUrl"):
                    st.link_button("Docket", m["DocketUrl"], use_container_width=True)
            with c3:
                if m.get("orig_id"):
                    st.caption(f"ID: {m['orig_id']}")

            st.divider()

        st.markdown("</div>", unsafe_allow_html=True)

    with det_col:
        render_details_panel(st.session_state.get("detail_doc"))

# ---------- Run queries (persist & render) ----------
if run_simple:
    with st.spinner("Searching…"):
        ranked = bm25.score(q_simple or "")
    _set_results(ranked, mode="bm25", topk=topk_simple)

if run_bool:
    with st.spinner("Running Boolean search…"):
        doc_ids = evaluate_query(idx, q_bool or "")
    # If checkbox on, re-rank; else show boolean only
    if run_rerank and doc_ids:
        with st.spinner("Re-ranking Boolean matches with BM25…"):
            reranked = bm25.score(q_bool or "", candidate_docs=doc_ids)
        _set_results(reranked, mode="bool_rerank", topk=topk_bool)
    else:
        _set_results(doc_ids, mode="bool", topk=topk_bool)

# Always render whatever results we have in session (so "Details" won't wipe the view)
last_hits = st.session_state.get("last_hits")
last_topk = st.session_state.get("last_topk", 10)
last_mode = st.session_state.get("last_mode")

if last_hits is not None:
    label = {"bm25": "BM25 results", "bool": "Boolean results", "bool_rerank": "BM25 re-rank of Boolean matches"}.get(last_mode, "Results")
    st.subheader(label)
    render_results_list(last_hits, topk=last_topk, with_scores=(last_mode != "bool"))
else:
    st.info("Type a query in **Search** or **Advanced (Boolean)** and click the button to see results.")

# app.py
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import streamlit as st

# --- import your project modules ---
from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

# ---------- App Config ----------
st.set_page_config(page_title="AeroSearch", page_icon="✈️", layout="wide")

# Fixed dataset settings (visible but not editable)
DATA_PATH_DEFAULT = "data/sample_docs1.json"  # change in code if needed
DATA_LIMIT_DEFAULT = 0                        # 0 = load all

# Narrative fields we stitch together for search
NARRATIVE_FIELDS = ["ProbableCause", "AnalysisNarrative", "FactualNarrative", "PrelimNarrative"]

# Metadata fields to show
META_FIELDS = [
    "NtsbNumber", "ReportNo", "EventDate", "Country", "City", "State",
    "AirportId", "AirportName",
    "EventType", "HighestInjury", "FatalInjuryCount", "SeriousInjuryCount", "MinorInjuryCount",
    "AirCraftDamage", "AccidentSiteCondition", "WeatherCondition",
    "Latitude", "Longitude",
]

VEHICLE_FIELDS = [
    "Make", "Model", "AircraftCategory", "DamageLevel", "RegulationFlightConductedUnder",
    "NumberOfEngines", "OperatorName", "RegistrationNumber", "flightScheduledType", "flightServiceType"
]

# ---------- Helpers ----------
def _concat_narratives(rec: dict) -> str:
    parts: List[str] = []
    for f in NARRATIVE_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(v.replace("\r\n", " ").replace("\n", " ").strip())
    return "  ".join(parts).strip()

@st.cache_data(show_spinner=True)
def load_json_records(path: str, limit: int | None = None) -> Tuple[Dict[int, str], Dict[int, dict]]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k in ("items", "data", "results", "Records"):
            if isinstance(raw.get(k), list):
                raw = raw[k]
                break
        else:
            raw = list(raw.values())
    assert isinstance(raw, list), "JSON must be an array or a dict containing a list"

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

        # cast lat/lon
        for k in ("Latitude", "Longitude"):
            if m.get(k) not in (None, ""):
                try:
                    m[k] = float(m[k])
                except Exception:
                    m[k] = None

        # prefer a stable id
        for stable in ("NtsbNumber", "ReportNo", "EventID", "MKey", "Oid"):
            v = r.get(stable)
            if v not in (None, ""):
                m["orig_id"] = v
                break

        # Vehicles[0] enrich
        vlist = r.get("Vehicles") or r.get("vehicles")
        if isinstance(vlist, list) and vlist:
            v0 = vlist[0]
            for k in VEHICLE_FIELDS:
                if v0.get(k) not in (None, ""):
                    m[k] = v0[k]

        meta[n] = m
        if limit and n >= limit:
            break
    return docs, meta

@st.cache_resource(show_spinner=True)
def build_index(docs: Dict[int, str]) -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(docs.items())
    return idx

def format_header(m: dict) -> str:
    bits = [m.get("EventDate"), m.get("City"), m.get("State"),
            m.get("Make"), m.get("Model"), m.get("EventType")]
    return " | ".join([str(x) for x in bits if x])

def _nice_label(k: str) -> str:
    return {
        "EventDate": "Event date",
        "EventType": "Event type",
        "HighestInjury": "Highest injury level",
        "FatalInjuryCount": "Fatalities",
        "SeriousInjuryCount": "Serious injuries",
        "MinorInjuryCount": "Minor injuries",
        "Country": "Country",
        "State": "State",
        "City": "City",
        "AirportId": "Airport ID",
        "AirportName": "Airport name",
        "AirCraftDamage": "Aircraft damage",
        "AccidentSiteCondition": "Site conditions",
        "WeatherCondition": "Weather",
        "Make": "Make",
        "Model": "Model",
        "AircraftCategory": "Aircraft category",
        "DamageLevel": "Damage level",
        "RegulationFlightConductedUnder": "Regulation",
        "NumberOfEngines": "Engines",
        "OperatorName": "Operator",
        "RegistrationNumber": "Registration",
        "flightScheduledType": "Schedule type",
        "flightServiceType": "Service type",
        "Latitude": "Latitude",
        "Longitude": "Longitude",
        "NtsbNumber": "NTSB number",
        "ReportNo": "Report number",
        "orig_id": "Record ID",
    }.get(k, k)

DETAIL_ORDER = [
    "orig_id", "NtsbNumber", "ReportNo",
    "EventDate", "Country", "State", "City", "AirportId", "AirportName",
    "EventType", "HighestInjury", "FatalInjuryCount", "SeriousInjuryCount", "MinorInjuryCount",
    "Make", "Model", "AircraftCategory", "NumberOfEngines",
    "OperatorName", "RegistrationNumber", "RegulationFlightConductedUnder",
    "WeatherCondition", "AccidentSiteCondition", "AirCraftDamage",
    "Latitude", "Longitude",
]

def _detail_dataframe(m: dict) -> pd.DataFrame:
    rows = []
    for k in DETAIL_ORDER:
        if k in m and m[k] not in (None, "", []):
            rows.append({"Field": _nice_label(k), "Value": m[k]})
    return pd.DataFrame(rows)

def render_hits(docs, meta, hits: List[tuple[int, float]] | List[int], topk: int, with_scores=True):
    rows_for_map = []
    for rank, item in enumerate(hits[:topk], start=1):
        d, s = item if isinstance(item, tuple) else (item, None)
        m = meta.get(d, {})
        header = format_header(m)
        snippet = docs[d][:240] + ("…" if len(docs[d]) > 240 else "")
        lat, lon = m.get("Latitude"), m.get("Longitude")

        st.markdown(f"**{rank}. doc {d}**" + (f" &nbsp;&nbsp;`score={s:.3f}`" if (with_scores and s is not None) else ""))
        if header:
            st.write(header)
        if lat is not None and lon is not None:
            st.write(f"Lat: {lat}, Lon: {lon}")
            rows_for_map.append({"latitude": lat, "longitude": lon, "doc": d})

        st.write(snippet)

        with st.expander("Read more & details"):
            st.markdown("**Full narrative**")
            st.write(docs[d])

            col1, col2 = st.columns([2, 1], vertical_alignment="top")
            with col1:
                st.markdown("**Metadata**")
                df = _detail_dataframe(m)
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No additional metadata available for this record.")
            with col2:
                if lat is not None and lon is not None:
                    st.markdown("**Location**")
                    _df = pd.DataFrame([{"latitude": lat, "longitude": lon}])
                    st.map(_df, latitude="latitude", longitude="longitude")
                if m.get("orig_id"):
                    st.markdown("**IDs**")
                    st.code(str(m["orig_id"]), language="text")
                if m.get("DocketUrl"):
                    st.markdown(f"[Docket link]({m['DocketUrl']})")

        st.divider()

    if rows_for_map:
        st.markdown("**Map of listed results**")
        df_all = pd.DataFrame(rows_for_map)
        st.map(df_all, latitude="latitude", longitude="longitude")

# ---------- Title ----------
st.title("AeroSearch: Aviation Incident/Accident Narrative Retrieval")

# ---------- Sidebar (fixed path shown, disabled, but the Limit/Reload enabled) ----------
with st.sidebar:
    st.header("Dataset (fixed)")
    st.text_input("JSON file path", DATA_PATH_DEFAULT, disabled=True)
    limit = st.number_input("Load first N records (0 = all)", value=DATA_LIMIT_DEFAULT, min_value=0, step=1000, key="limit")
    load_btn = st.button("Load / Reload Data", type="primary")

    st.divider()
    st.header("Search mode")
    mode = st.radio("Choose", ["BM25", "Boolean (with BM25 re-rank)"], index=0, key="mode")
    st.caption("Tip: Multi-term queries often perform better; use Boolean/proximity when needed.")

# ---------- Load data / build index ----------
if load_btn or "docs_cache" not in st.session_state:
    docs, meta = load_json_records(DATA_PATH_DEFAULT, None if limit == 0 else limit)
    st.session_state["docs_cache"] = docs
    st.session_state["meta_cache"] = meta
    st.session_state["index_cache"] = build_index(docs)
    st.session_state["results_cache"] = None
    st.session_state["last_query"] = ""
    st.session_state["last_topk"] = 10
    st.session_state["last_mode"] = mode

docs = st.session_state["docs_cache"]
meta = st.session_state["meta_cache"]
idx: InvertedIndex = st.session_state["index_cache"]
bm25 = BM25(idx)

st.success(f"Loaded {len(docs)} docs. Indexed over {idx.num_docs} documents.")

# ---------- Query UI ----------
colq1, colq2 = st.columns([4, 1])
with colq1:
    query = st.text_input(
        "Enter your query",
        placeholder='e.g., unstable approach tailwind long landing excursion OR (icing /5 pitot) AND (airspeed OR indications)',
        key="query_text"
    )
with colq2:
    topk = st.slider("Top-k", 5, 50, st.session_state.get("last_topk", 10), key="topk")

search_btn = st.button("Search", type="primary", use_container_width=True)

# ---------- Run search ----------
if search_btn:
    st.session_state["last_query"] = query
    st.session_state["last_topk"] = topk
    st.session_state["last_mode"] = mode

    if not query.strip():
        st.warning("Please enter a query.")
        st.session_state["results_cache"] = None
    else:
        with st.status("Searching…", expanded=False) as status:
            try:
                if mode.startswith("BM25"):
                    status.update(label="Scoring with BM25…")
                    ranked = bm25.score(query)
                    st.session_state["results_cache"] = {"type": "bm25", "primary": ranked}
                    status.update(label="Done.", state="complete")
                else:
                    status.update(label="Evaluating Boolean query…")
                    doc_ids = evaluate_query(idx, query)
                    status.update(label=f"Matched {len(doc_ids)} candidate docs. Re-ranking…")
                    reranked = bm25.score(query, candidate_docs=doc_ids) if doc_ids else []
                    st.session_state["results_cache"] = {
                        "type": "boolean",
                        "primary": doc_ids,
                        "rerank": reranked
                    }
                    status.update(label="Done.", state="complete")
            except Exception as e:
                status.update(label="Error during search.", state="error")
                st.error(str(e))
                st.session_state["results_cache"] = None

# ---------- Display results ----------
res = st.session_state.get("results_cache")
if res:
    q_disp = st.session_state.get("last_query", "")
    k_disp = st.session_state.get("last_topk", 10)
    mode_disp = st.session_state.get("last_mode", "BM25")

    st.caption(f"Showing results for **{mode_disp}** · **k={k_disp}** · Query: `{q_disp}`")

    if res["type"] == "bm25":
        st.subheader("BM25 results")
        if not res["primary"]:
            st.info("No results found.")
        else:
            render_hits(docs, meta, res["primary"], topk=k_disp, with_scores=True)
    elif res["type"] == "boolean":
        prim = res.get("primary", [])
        st.subheader(f"Boolean results (matched {len(prim)} docs)")
        if prim:
            render_hits(docs, meta, prim, topk=k_disp, with_scores=False)
        else:
            st.info("No matches for the Boolean query.")
        rr = res.get("rerank", [])
        st.subheader("BM25 re-rank of Boolean matches")
        if rr:
            render_hits(docs, meta, rr, topk=k_disp, with_scores=True)
        else:
            st.info("No matches to re-rank.")

# app.py
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import streamlit as st

# --- import your project modules ---
from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

# ---------- Data loading (reuse your JSON loader logic) ----------
NARRATIVE_FIELDS = ["ProbableCause", "AnalysisNarrative", "FactualNarrative", "PrelimNarrative"]

# Add more research-useful fields to metadata
META_FIELDS = [
    # IDs / dates / place
    "NtsbNumber", "ReportNo", "EventDate", "Country", "City", "State",
    "AirportId", "AirportName",
    # event / outcome
    "EventType", "HighestInjury", "FatalInjuryCount", "SeriousInjuryCount", "MinorInjuryCount",
    "AirCraftDamage", "AccidentSiteCondition", "WeatherCondition",
    # coords
    "Latitude", "Longitude",
]

# Also try to grab these from Vehicles[0]
VEHICLE_FIELDS = [
    "Make", "Model", "AircraftCategory", "DamageLevel", "RegulationFlightConductedUnder",
    "NumberOfEngines", "OperatorName", "RegistrationNumber", "flightScheduledType", "flightServiceType"
]

def _concat_narratives(rec: dict) -> str:
    parts: List[str] = []
    for f in NARRATIVE_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            v = v.replace("\r\n", " ").replace("\n", " ").strip()
            parts.append(v)
    return "  ".join(parts).strip()

@st.cache_data(show_spinner=False)
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

        # base meta
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

        # try Vehicles[0] for aircraft fields
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

# ---------- UI ----------
st.set_page_config(page_title="AeroSearch", page_icon="✈️", layout="wide")
st.title("AeroSearch — Aviation Incident/Accident Narrative Retrieval")

with st.sidebar:
    st.header("Dataset")
    data_path = st.text_input("JSON file path", "data/sample_docs1.json")
    limit = st.number_input("Load first N records (0 = all)", value=0, min_value=0, step=1000)
    load_btn = st.button("Load / Reload Data", type="primary")

    st.divider()
    st.header("Search mode")
    mode = st.radio("Choose", ["BM25", "Boolean (with BM25 re-rank)"], index=0)

    st.divider()
    st.caption("Tip: Multi-term queries are faster & better. Use `bool:` style without the prefix here (this toggle handles it).")

# Load data (once)
if load_btn or "docs_cache" not in st.session_state:
    docs, meta = load_json_records(data_path, None if limit == 0 else limit)
    st.session_state["docs_cache"] = docs
    st.session_state["meta_cache"] = meta
    st.session_state["index_cache"] = build_index(docs)

docs = st.session_state.get("docs_cache", {})
meta = st.session_state.get("meta_cache", {})
idx: InvertedIndex = st.session_state.get("index_cache", build_index({}))

st.success(f"Loaded {len(docs)} docs. Indexed over {idx.num_docs} documents.")

query = st.text_input(
    "Enter your query",
    placeholder='e.g., "runway excursion landing" or (icing /5 pitot) AND (airspeed OR indications)'
)
topk = st.slider("Results to display", 5, 50, 10)

# BM25 object (reuse per run)
bm25 = BM25(idx)

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
    # core identity
    "orig_id", "NtsbNumber", "ReportNo",
    # timing/location
    "EventDate", "Country", "State", "City", "AirportId", "AirportName",
    # event/outcome
    "EventType", "HighestInjury", "FatalInjuryCount", "SeriousInjuryCount", "MinorInjuryCount",
    # aircraft & ops
    "Make", "Model", "AircraftCategory", "NumberOfEngines",
    "OperatorName", "RegistrationNumber", "RegulationFlightConductedUnder",
    # environment
    "WeatherCondition", "AccidentSiteCondition", "AirCraftDamage",
    # coords at end
    "Latitude", "Longitude",
]

def _detail_dataframe(m: dict) -> pd.DataFrame:
    rows = []
    for k in DETAIL_ORDER:
        if k in m and m[k] not in (None, "", []):
            rows.append({"Field": _nice_label(k), "Value": m[k]})
    return pd.DataFrame(rows)

def show_hits(hits: List[tuple[int, float]] | List[int], with_scores=True):
    rows_for_map = []
    for rank, item in enumerate(hits[:topk], start=1):
        if isinstance(item, tuple):
            d, s = item
        else:
            d, s = item, None
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

        # --- READ MORE & DETAILS ---
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

    # map (if any coords in the current result set)
    if rows_for_map:
        st.markdown("**Map of listed results**")
        df_all = pd.DataFrame(rows_for_map)
        st.map(df_all, latitude="latitude", longitude="longitude")

if query:
    if mode.startswith("BM25"):
        ranked = bm25.score(query)
        st.subheader("BM25 results")
        show_hits(ranked, with_scores=True)
    else:
        doc_ids = evaluate_query(idx, query)
        st.subheader(f"Boolean results (matched {len(doc_ids)} docs)")
        show_hits(doc_ids, with_scores=False)
        if doc_ids:
            st.subheader("BM25 re-rank of Boolean matches")
            reranked = bm25.score(query, candidate_docs=doc_ids)
            show_hits(reranked, with_scores=True)

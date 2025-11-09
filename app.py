# app.py
import io
import re
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

# --- project modules ---
from src.index import InvertedIndex
from src.boolean_query import evaluate_query
from src.bm25 import BM25

# ---------- App Config ----------
st.set_page_config(page_title="AeroSearch", page_icon="✈️", layout="wide")

# Fixed dataset settings (visible but disabled)
DATA_PATH_DEFAULT = "data/sample_docs1.json"   # change in code if needed
DATA_LIMIT_DEFAULT = 0                         # 0 = all

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

# Domain synonyms (very small, optional expansion)
AVIATION_SYNONYMS = {
    "veer-off": ["runway excursion", "runway veer off", "runway overrun"],
    "overrun": ["long landing", "runway excursion"],
    "incursion": ["runway incursion", "surface incident"],
    "cfit": ["controlled flight into terrain", "terrain impact"],
    "icing": ["ice accretion", "airframe icing", "pitot icing"],
    "tailwind": ["downwind landing"],
    "unstable": ["unstabilized", "unstable approach"]
}

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

        # stable id
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

def _to_date(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    s = s.replace("Z","")
    try:
        return datetime.fromisoformat(s.split("T")[0])
    except Exception:
        return None

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

# ---------- Text utilities ----------
SENT_RX = re.compile(r"(?<=[.!?])\s+")
WORD_RX = re.compile(r"\w+")

def tokenize(s: str) -> List[str]:
    return WORD_RX.findall(s.lower())

def split_sentences(txt: str) -> List[str]:
    parts = SENT_RX.split(txt)
    if len(parts) == 1 and len(txt) > 240:
        return [txt[:240]+"…"]
    return parts

def make_highlight(text: str, terms: List[str]) -> str:
    terms = [t for t in terms if t]
    if not terms: return text
    pats = [re.escape(t) for t in terms]
    rx = re.compile(r"(" + "|".join(pats) + r")", re.IGNORECASE)
    return rx.sub(r"**\1**", text)

def best_sentence(text: str, query_terms: List[str], idf: Dict[str, float]) -> str:
    sents = split_sentences(text)
    if not sents: return text[:240] + ("…" if len(text) > 240 else "")
    scores = []
    qset = set(t.lower() for t in query_terms)
    for s in sents:
        toks = set(tokenize(s))
        score = sum(idf.get(t, 0.0) for t in qset if t in toks)
        scores.append((score, s))
    best = max(scores, key=lambda x: x[0])[1] if scores else sents[0]
    return best

# ---------- BM25 helpers ----------
def default_idf(idx: InvertedIndex, terms: List[str]) -> Dict[str, float]:
    idf = {}
    N = getattr(idx, "num_docs", 0) or getattr(idx, "N", 0) or 0
    postings = getattr(idx, "postings", None) or getattr(idx, "inverted", None)
    for t in set(terms):
        df = 0
        if postings and t in postings:
            p = postings[t]
            if isinstance(p, dict):
                df = len(p)
            elif isinstance(p, list):
                df = len(p)
        if N > 0 and df > 0:
            # standard BM25 idf
            idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        else:
            idf[t] = 0.0
    return idf

def bm25_explain_safe(bm25: BM25, idx: InvertedIndex, doc_id: int, terms: List[str]) -> Dict[str, float]:
    try:
        if hasattr(bm25, "explain"):
            exp = bm25.explain(doc_id, terms)  # expected [(term, score), ...]
            return {t: float(s) for t, s in exp}
    except Exception:
        pass
    postings = getattr(idx, "postings", None) or getattr(idx, "inverted", None)
    idf = default_idf(idx, terms)
    contrib = {}
    for t in terms:
        tf = 0
        if postings and t in postings:
            p = postings[t]
            if isinstance(p, dict) and doc_id in p:
                tf = len(p[doc_id]) if isinstance(p[doc_id], list) else int(p[doc_id] or 1)
        if tf:
            contrib[t] = round(tf * idf.get(t, 1.0), 4)
    return contrib

# ---------- Title ----------
st.title("AeroSearch — Aviation Incident/Accident Narrative Retrieval")

# ---------- Sidebar (fixed path + load/limit; filters; options) ----------
with st.sidebar:
    st.header("Dataset")
    st.text_input("JSON file path", DATA_PATH_DEFAULT, disabled=True)
    limit = st.number_input("Load first N records (0 = all)", value=DATA_LIMIT_DEFAULT, min_value=0, step=1000, key="limit")
    load_btn = st.button("Load / Reload Data", type="primary")

    st.divider()
    st.header("Search mode")
    mode = st.radio("Mode", ["BM25", "Boolean (BM25 re-rank)"], index=0, key="mode")

    st.divider()
    st.header("Filters")
    st.session_state.setdefault("filter_values_ready", False)
    if st.session_state["filter_values_ready"]:
        sel_event = st.multiselect("EventType", st.session_state["EVENT_TYPES"], default=[])
        sel_injury = st.multiselect("HighestInjury", st.session_state["HIGHEST_INJURY"], default=[])
        sel_make = st.multiselect("Make", st.session_state["MAKES"], default=[])
        sel_model = st.multiselect("Model", st.session_state["MODELS"], default=[])
        dmin, dmax = st.session_state["DATE_MINMAX"]
        date_range = st.slider("Event date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))
        lat_min, lat_max = st.session_state["LAT_MINMAX"]
        lon_min, lon_max = st.session_state["LON_MINMAX"]
        gb_lat = st.slider("Latitude range", min_value=lat_min, max_value=lat_max, value=(lat_min, lat_max))
        gb_lon = st.slider("Longitude range", min_value=lon_min, max_value=lon_max, value=(lon_min, lon_max))
    else:
        st.info("Load data to enable filters.")
        sel_event, sel_injury, sel_make, sel_model = [], [], [], []
        date_range = (datetime(1900,1,1), datetime(2100,1,1))
        gb_lat, gb_lon = (-90.0, 90.0), (-180.0, 180.0)

    st.divider()
    st.header("IR options")
    prf_on = st.checkbox("Improve recall with PRF (pseudo-relevance feedback)", value=False)
    prf_k = st.number_input("PRF: expand from top-k", 5, 50, 10)
    prf_terms = st.number_input("PRF: add top terms", 1, 15, 5)
    syn_on = st.checkbox("Use aviation synonyms", value=False)
    passage_on = st.checkbox("Show best passage per result", value=True)

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

    def _vals(field):
        return sorted({m.get(field) for m in meta.values() if m.get(field)})
    st.session_state["EVENT_TYPES"] = _vals("EventType")
    st.session_state["HIGHEST_INJURY"] = _vals("HighestInjury")
    st.session_state["MAKES"] = _vals("Make")
    st.session_state["MODELS"] = _vals("Model")

    dates = sorted([_to_date(m.get("EventDate")) for m in meta.values() if _to_date(m.get("EventDate"))])
    dmin = dates[0] if dates else datetime(1950,1,1)
    dmax = dates[-1] if dates else datetime(2035,1,1)
    st.session_state["DATE_MINMAX"] = (dmin, dmax)
    lats = [m.get("Latitude") for m in meta.values() if isinstance(m.get("Latitude"), float)]
    lons = [m.get("Longitude") for m in meta.values() if isinstance(m.get("Longitude"), float)]
    st.session_state["LAT_MINMAX"] = (min(lats) if lats else -90.0, max(lats) if lats else 90.0)
    st.session_state["LON_MINMAX"] = (min(lons) if lons else -180.0, max(lons) if lons else 180.0)

    st.session_state["filter_values_ready"] = True

docs = st.session_state.get("docs_cache", {})
meta = st.session_state.get("meta_cache", {})
idx: InvertedIndex = st.session_state.get("index_cache", build_index({}))
bm25 = BM25(idx)

st.success(f"Loaded {len(docs)} docs. Indexed over {idx.num_docs} documents.")

# ---------- Query + permalinks (NEW API ONLY) ----------
# Read incoming params
pre_q = st.query_params.get("q", "")
try:
    pre_k = int(st.query_params.get("k", "10"))
except Exception:
    pre_k = 10
pre_mode = st.query_params.get("m", "BM25")

colq1, colq2 = st.columns([4, 1])
with colq1:
    query = st.text_input(
        "Enter your query",
        value=pre_q or st.session_state.get("last_query", ""),
        placeholder='e.g., unstable approach tailwind long landing excursion OR (icing /5 pitot) AND (airspeed OR indications)',
        key="query_text"
    )
with colq2:
    topk = st.slider("Top-k", 5, 50, pre_k if pre_k else st.session_state.get("last_topk", 10), key="topk")

search_btn = st.button("Search", type="primary", use_container_width=True)

# ---------- Build candidate set from filters ----------
def doc_passes_filters(did: int) -> bool:
    m = meta.get(did, {})
    if sel_event and m.get("EventType") not in sel_event: return False
    if sel_injury and m.get("HighestInjury") not in sel_injury: return False
    if sel_make and m.get("Make") not in sel_make: return False
    if sel_model and m.get("Model") not in sel_model: return False
    dt = _to_date(m.get("EventDate"))
    if dt and not (date_range[0] <= dt <= date_range[1]): return False
    lat, lon = m.get("Latitude"), m.get("Longitude")
    if lat is not None and lon is not None:
        if not (gb_lat[0] <= lat <= gb_lat[1] and gb_lon[0] <= lon <= gb_lon[1]):
            return False
    return True

def filter_docs(dids: Optional[List[int]] = None) -> List[int]:
    if dids is None:
        dids = list(docs.keys())
    keep = [d for d in dids if doc_passes_filters(d)]
    return keep

# ---------- Query expansion ----------
STOP_SMALL = set(["the","a","an","and","or","of","to","in","on","for","with","by","from","as","at","is","are","be","was","were","it","that","this","these","those","into","over","under"])

def expand_with_synonyms(q: str) -> str:
    if not syn_on:
        return q
    terms = set(tokenize(q))
    expansions = []
    for t in terms:
        if t in AVIATION_SYNONYMS:
            for s in AVIATION_SYNONYMS[t]:
                expansions.append(f'"{s}"')
    if not expansions:
        return q
    return f'({q}) OR (' + " OR ".join(expansions) + ")"

def prf_expand(query_text: str, ranked: List[Tuple[int, float]], add_terms: int, top_k: int) -> str:
    if not prf_on or not ranked:
        return query_text
    tf: Dict[str, int] = {}
    for d, _ in ranked[:top_k]:
        for t in tokenize(docs[d]):
            if len(t) <= 2 or t in STOP_SMALL:
                continue
            tf[t] = tf.get(t, 0) + 1
    q_terms = set(tokenize(query_text))
    cand = [t for t, _ in sorted(tf.items(), key=lambda x: -x[1]) if t not in q_terms]
    add = cand[:add_terms]
    if not add:
        return query_text
    return query_text + " OR (" + " OR ".join(add) + ")"

# ---------- Run search ----------
if search_btn:
    # Update URL (new API only)
    st.query_params["q"] = query
    st.query_params["k"] = str(topk)
    st.query_params["m"] = mode.split()[0]

    st.session_state["last_query"] = query
    st.session_state["last_topk"] = topk
    st.session_state["last_mode"] = mode

    if not query.strip():
        st.warning("Please enter a query.")
        st.session_state["results_cache"] = None
    else:
        with st.status("Searching…", expanded=False) as status:
            try:
                base_candidates = filter_docs(None)
                q_use = expand_with_synonyms(query)

                if mode.startswith("BM25"):
                    status.update(label="Scoring with BM25…")
                    ranked = bm25.score(q_use, candidate_docs=base_candidates)
                    if prf_on and ranked:
                        status.update(label="Applying PRF expansion…")
                        q_use2 = prf_expand(q_use, ranked, prf_terms, prf_k)
                        ranked = bm25.score(q_use2, candidate_docs=base_candidates)
                        q_use = q_use2
                    st.session_state["results_cache"] = {"type": "bm25", "primary": ranked, "q_used": q_use}
                    status.update(label="Done.", state="complete")

                else:
                    status.update(label="Evaluating Boolean query…")
                    doc_ids = evaluate_query(idx, q_use)
                    status.update(label=f"Matched {len(doc_ids)} candidates. Filtering + re-ranking…")
                    doc_ids = filter_docs(doc_ids)
                    reranked = bm25.score(q_use, candidate_docs=doc_ids) if doc_ids else []
                    if prf_on and reranked:
                        status.update(label="Applying PRF expansion…")
                        q_use2 = prf_expand(q_use, reranked, prf_terms, prf_k)
                        reranked = bm25.score(q_use2, candidate_docs=doc_ids)
                        q_use = q_use2
                    st.session_state["results_cache"] = {
                        "type": "boolean",
                        "primary": doc_ids,
                        "rerank": reranked,
                        "q_used": q_use
                    }
                    status.update(label="Done.", state="complete")

            except Exception as e:
                status.update(label="Error during search.", state="error")
                st.error(str(e))
                st.session_state["results_cache"] = None

# ---------- Render results + export ----------
def strip_md(s: str) -> str:
    return s.replace("**", "")

def render_hits(hits: List[Tuple[int, float]] | List[int], topk_show: int, with_scores=True, q_used: str = ""):
    rows_for_map = []
    rows_for_csv = []
    q_terms = [t for t in tokenize(q_used)]
    idf = default_idf(idx, q_terms)

    for rank, item in enumerate(hits[:topk_show], start=1):
        d, s = item if isinstance(item, tuple) else (item, None)
        m = meta.get(d, {})
        header = format_header(m)

        if passage_on:
            base_snip = best_sentence(docs[d], q_terms, idf)
        else:
            base_snip = docs[d][:240] + ("…" if len(docs[d]) > 240 else "")

        snippet = make_highlight(base_snip, q_terms)
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
            st.write(make_highlight(docs[d], q_terms))

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

                st.markdown("**Why this result?**")
                contrib = bm25_explain_safe(bm25, idx, d, q_terms)
                if contrib:
                    st.json(contrib, expanded=False)
                else:
                    st.caption("No term-level contributions available.")

                if m.get("orig_id"):
                    st.markdown("**Record ID**")
                    st.code(str(m["orig_id"]), language="text")
                if m.get("DocketUrl"):
                    st.markdown(f"[Docket link]({m['DocketUrl']})")

        st.divider()

        rows_for_csv.append({
            "rank": rank,
            "doc_id": d,
            "score": s if s is not None else "",
            "snippet": strip_md(snippet),
            **{k: m.get(k, "") for k in DETAIL_ORDER}
        })

    if rows_for_map:
        st.markdown("**Map of listed results**")
        df_all = pd.DataFrame(rows_for_map)
        st.map(df_all, latitude="latitude", longitude="longitude")

    if rows_for_csv:
        df_export = pd.DataFrame(rows_for_csv)
        csv_buf = io.StringIO()
        df_export.to_csv(csv_buf, index=False)
        st.download_button("Download results (CSV)", csv_buf.getvalue(), file_name="aerosearch_results.csv", mime="text/csv")

res = st.session_state.get("results_cache")
if res:
    q_disp = st.session_state.get("last_query", "")
    k_disp = st.session_state.get("last_topk", 10)
    mode_disp = st.session_state.get("last_mode", "BM25")
    q_used = res.get("q_used", q_disp)

    st.caption(f"Showing results for **{mode_disp}** · **k={k_disp}** · Query: `{q_disp}`")
    # Show current URL params (new API only)
    st.write("Permalink params:")
    st.code(json.dumps(dict(st.query_params)), language="json")

    if res["type"] == "bm25":
        st.subheader("BM25 results")
        if not res["primary"]:
            st.info("No results found.")
        else:
            render_hits(res["primary"], topk_show=k_disp, with_scores=True, q_used=q_used)

    elif res["type"] == "boolean":
        prim = res.get("primary", [])
        st.subheader(f"Boolean results (matched {len(prim)} docs)")
        if prim:
            render_hits(prim, topk_show=k_disp, with_scores=False, q_used=q_used)
        else:
            st.info("No matches for the Boolean query.")
        rr = res.get("rerank", [])
        st.subheader("BM25 re-rank of Boolean matches")
        if rr:
            render_hits(rr, topk_show=k_disp, with_scores=True, q_used=q_used)
        else:
            st.info("No matches to re-rank.")

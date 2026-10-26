from collections import Counter
import streamlit as st
import requests
from cvss import CVSS3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_retrieval import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    filtered_metadata,
    get_asset_by_id,
    list_all_assets,
    retrieve_asset_context,
    update_asset_metadata,
)
from llm_assessment import ContextualAssessment, LlmAssessmentError, analyze_with_llm
from organizational_priority import calculate_organizational_priority
from cve_applicability import ApplicabilityStatus, determine_applicability
from asset_inventory import (
    SOFTWARE_INVENTORY_FIELD,
    collect_asset_cpes,
    is_valid_cpe,
    load_software_inventory,
    save_software_inventory,
)
from assessment_history import (
    delete_record,
    load_history,
    record_from_analysis,
    save_assessment,
)
from asset_schema_v2 import ALLOWED_VALUES
from ingest_assets import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME_V2,
    EMBEDDING_MODEL_NAME,
)

st.set_page_config(
    page_title="Asset Intelligence & Vulnerability Assessment",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_OPTIONS = [
    "Dashboard",
    "Asset Inventory",
    "Vulnerability Analysis",
    "Analysis History",
]

DARK_MODE_STYLES = """
<style>
:root { color-scheme: dark; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stHeader"] { background-color: #0d1117; }
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #161b22;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp li, .stApp [data-testid="stCaptionContainer"] {
    color: #e6edf3;
}
.stApp [data-testid="stVerticalBlockBorderWrapper"],
.stApp [data-testid="stMetric"],
.stApp [data-testid="stForm"],
.stApp [data-testid="stExpander"] details {
    background-color: #161b22;
    border-color: #30363d;
}
.stApp input, .stApp textarea,
.stApp [data-baseweb="select"] > div {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}
.stApp pre, .stApp code {
    background-color: #161b22;
    color: #e6edf3;
}
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stDataEditor"] { color-scheme: dark; }
</style>
"""

# ---------------------------------------------------------------------------
# 1. SETUP & CACHING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_vector_db():
    """Memuat ChromaDB koleksi Asset Schema v2 hanya sekali."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        collection_name=COLLECTION_NAME_V2,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


try:
    vector_store = load_vector_db()
    db_status = "Ready"
except Exception as e:
    vector_store = None
    db_status = f"Error: {e}"


@st.cache_data(ttl="5m")
def load_all_assets_metadata() -> list[dict]:
    """Load every asset metadata record from ChromaDB (small dataset)."""
    return list_all_assets(vector_store)


# ---------------------------------------------------------------------------
# 2. SESSION STATE
# ---------------------------------------------------------------------------
def _init_session_state():
    st.session_state.setdefault("analysis_asset_id", "")
    st.session_state.setdefault("analysis_asset_preselect", "")
    st.session_state.setdefault("inventory_selected_asset_id", "")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_record", None)
    st.session_state.setdefault("dark_mode", False)


_init_session_state()


def _toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Handle cross-page navigation triggered by buttons rendered later in the app.
# We use a query parameter because Streamlit disallows modifying a widget's
# session-state key after that widget has been instantiated in the same run.
_nav_target = st.query_params.get("nav")
if _nav_target:
    if _nav_target in PAGE_OPTIONS:
        st.session_state.nav_page = _nav_target
    del st.query_params["nav"]
if st.session_state.get("nav_page") not in PAGE_OPTIONS:
    st.session_state.nav_page = "Dashboard"


# ---------------------------------------------------------------------------
# 3. SHARED HELPERS
# ---------------------------------------------------------------------------
def fetch_cve_from_nvd(cve_id):
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data.get("vulnerabilities"):
            return None, "CVE tidak ditemukan."

        cve_data = data["vulnerabilities"][0]["cve"]
        descriptions = cve_data.get("descriptions", [])
        desc = next((d["value"] for d in descriptions if d["lang"] == "en"), "Description not provided by NVD.")

        metrics = cve_data.get("metrics", {})
        cvss_metrics = metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30") or []
        cvss_v3 = cvss_metrics[0].get("cvssData", {}) if cvss_metrics else {}

        return {
            "cve_id": cve_id,
            "description": desc,
            "base_score": cvss_v3.get("baseScore"),
            "vector_string": cvss_v3.get("vectorString", ""),
            "configurations": cve_data.get("configurations", {}),
        }, "Sukses"
    except Exception as e:
        return None, str(e)


def build_environmental_vector(original_vector, assessment: ContextualAssessment):
    """Deterministic Environmental vector construction.

    Appends the model-assessed CR/IR/AR onto the preserved NVD Base vector.
    Base metrics are NEVER modified, and MAV is left undefined (X).
    """
    if not original_vector or not original_vector.strip().startswith("CVSS:3."):
        raise ValueError("Base Vector NVD tidak valid.")
    CVSS3(original_vector.strip())

    parts = original_vector.strip().split("/")
    base_parts = [p for p in parts if not p.startswith(("CR:", "IR:", "AR:", "MAV:"))]

    cr = assessment.confidentiality_requirement.value
    ir = assessment.integrity_requirement.value
    ar = assessment.availability_requirement.value

    return "/".join(base_parts) + f"/CR:{cr}/IR:{ir}/AR:{ar}"


def _asset_label(asset: dict) -> str:
    asset_id = asset.get("asset_id", "?")
    name = asset.get("asset_name", asset_id)
    return f"{asset_id} — {name}"


def _navigate_to(page: str):
    st.query_params.nav = page
    st.rerun()


# ---------------------------------------------------------------------------
# 4. SIDEBAR — NAVIGATION + GLOBAL CONFIG
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(":material/security: Asset Intel")

    page = st.radio(
        "Navigation",
        PAGE_OPTIONS,
        key="nav_page",
        label_visibility="collapsed",
    )
    st.button(
        "Switch to light mode" if st.session_state.dark_mode else "Switch to dark mode",
        icon=":material/light_mode:" if st.session_state.dark_mode else ":material/dark_mode:",
        key="theme_toggle",
        on_click=_toggle_dark_mode,
    )

    st.divider()
    st.header(":material/settings: Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Dapatkan dari Google AI Studio",
    )
    confidence_threshold = st.slider(
        "Retrieval confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Normalized relevance score. Low-confidence semantic candidates are flagged; the selected stable asset ID remains the context source.",
    )

    st.metric(label="ChromaDB status", value=db_status)
    if db_status == "Ready":
        st.success(f"Collection '{COLLECTION_NAME_V2}' active.")
    else:
        st.error("Run `python ingest_assets.py --v2` first.")

if st.session_state.dark_mode:
    st.html(DARK_MODE_STYLES)


# ---------------------------------------------------------------------------
# 5. PAGE RENDERERS
# ---------------------------------------------------------------------------
def _render_dashboard():
    st.header(":material/dashboard: Dashboard")

    assets = load_all_assets_metadata()
    history = load_history()
    with st.container(horizontal=True):
        st.metric("Total assets", len(assets), border=True)
        st.metric("Completed assessments", len(history), border=True)

    severity_counts = Counter()
    priority_counts = Counter()
    for record in history:
        score = record.context_score if record.context_score is not None else record.base_score
        if score is not None:
            if score >= 9.0:
                severity_counts["Critical"] += 1
            elif score >= 7.0:
                severity_counts["High"] += 1
            elif score >= 4.0:
                severity_counts["Medium"] += 1
            elif score > 0:
                severity_counts["Low"] += 1
        if record.priority in {"P1", "P2", "P3", "P4"}:
            priority_counts[record.priority] += 1

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Severity distribution")
            if severity_counts:
                st.bar_chart(dict(severity_counts))
            else:
                st.info("No completed assessments with a CVSS score.")

    with col2:
        with st.container(border=True):
            st.subheader("Priority distribution")
            if priority_counts:
                st.bar_chart(dict(priority_counts))
            else:
                st.info("No completed assessments with a deterministic priority.")

    with st.container(border=True):
        st.subheader("Recent assessments")
        if history:
            recent = history[-5:][::-1]
            st.dataframe(
                [
                    {
                        "Time": r.timestamp[:19].replace("T", " "),
                        "CVE": r.cve_id,
                        "Asset": r.asset_name or r.asset_id,
                        "Base": r.base_score,
                        "Context": r.context_score,
                        "Priority": r.priority,
                    }
                    for r in recent
                ],
                hide_index=True,
            )
        else:
            st.info("No assessments recorded yet. Run an analysis from Vulnerability Analysis.")

    with st.container(horizontal=True):
        if st.button("Run Vulnerability Analysis", icon=":material/biotech:"):
            _navigate_to("Vulnerability Analysis")


def _render_asset_detail(assets: list[dict]):
    """Right-hand asset detail editor for the Asset Inventory page."""
    selected_id = st.session_state.inventory_selected_asset_id
    asset = next((a for a in assets if a.get("asset_id") == selected_id), None)

    if not asset:
        with st.container(border=True):
            st.info("Select an asset from the list to view and edit its metadata.")
        return

    st.subheader(f"{asset.get('asset_name', selected_id)}")
    st.caption(f"Asset ID: `{selected_id}` | IP: `{asset.get('ip_address', 'unknown')}`")

    tab_meta, tab_software = st.tabs(["Metadata", "Software Inventory"])

    with tab_meta:
        with st.form(f"asset_meta_form_{selected_id}"):
            st.markdown("**Editable fields**")
            e_name = st.text_input("Asset name", value=asset.get("asset_name", ""), key=f"am_name_{selected_id}")
            e_desc = st.text_area("Description", value=asset.get("description", ""), key=f"am_desc_{selected_id}")
            e_ip = st.text_input("IP address", value=asset.get("ip_address", ""), key=f"am_ip_{selected_id}")

            owner_col, business_col = st.columns(2)
            with owner_col:
                e_owner = st.text_input("Owner", value=asset.get("owner", ""), key=f"am_owner_{selected_id}")
                e_owner_contact = st.text_input("Owner contact", value=asset.get("owner_contact", ""), key=f"am_owner_contact_{selected_id}")
            with business_col:
                e_business_function = st.text_input("Business function", value=asset.get("business_function", ""), key=f"am_business_function_{selected_id}")
                e_vendor = st.text_input("Vendor", value=asset.get("vendor", ""), key=f"am_vendor_{selected_id}")

            c1, c2 = st.columns(2)
            def _safe_index(options, value):
                v = value if value in options else options[0]
                return options.index(v)

            with c1:
                e_type = st.selectbox("Asset type", options=ALLOWED_VALUES["asset_type"], index=_safe_index(ALLOWED_VALUES["asset_type"], asset.get("asset_type", "unknown")), key=f"am_type_{selected_id}")
                e_crit = st.selectbox("Business criticality", options=ALLOWED_VALUES["business_criticality"], index=_safe_index(ALLOWED_VALUES["business_criticality"], asset.get("business_criticality", "unknown")), key=f"am_crit_{selected_id}")
                e_sens = st.selectbox("Data sensitivity", options=ALLOWED_VALUES["data_sensitivity"], index=_safe_index(ALLOWED_VALUES["data_sensitivity"], asset.get("data_sensitivity", "unknown")), key=f"am_sens_{selected_id}")
                e_env = st.selectbox("Environment", options=ALLOWED_VALUES["environment"], index=_safe_index(ALLOWED_VALUES["environment"], asset.get("environment", "unknown")), key=f"am_env_{selected_id}")
            with c2:
                e_zone = st.selectbox("Network zone", options=ALLOWED_VALUES["network_zone"], index=_safe_index(ALLOWED_VALUES["network_zone"], asset.get("network_zone", "unknown")), key=f"am_zone_{selected_id}")
                e_exp = st.selectbox("Internet exposed", options=ALLOWED_VALUES["internet_exposed"], index=_safe_index(ALLOWED_VALUES["internet_exposed"], asset.get("internet_exposed", "unknown")), key=f"am_exp_{selected_id}")
                e_data_class = st.text_input("Data classification", value=asset.get("data_classification", ""), key=f"am_dc_{selected_id}")

            st.markdown("**CIA requirements**")
            c3, c4, c5 = st.columns(3)
            with c3:
                e_cr = st.selectbox("CR", options=ALLOWED_VALUES["confidentiality_requirement"], index=_safe_index(ALLOWED_VALUES["confidentiality_requirement"], asset.get("confidentiality_requirement", "X")), key=f"am_cr_{selected_id}")
            with c4:
                e_ir = st.selectbox("IR", options=ALLOWED_VALUES["integrity_requirement"], index=_safe_index(ALLOWED_VALUES["integrity_requirement"], asset.get("integrity_requirement", "X")), key=f"am_ir_{selected_id}")
            with c5:
                e_ar = st.selectbox("AR", options=ALLOWED_VALUES["availability_requirement"], index=_safe_index(ALLOWED_VALUES["availability_requirement"], asset.get("availability_requirement", "X")), key=f"am_ar_{selected_id}")

            st.markdown("**Legacy single-software fields**")
            e_sw_name = st.text_input("Software name", value=asset.get("software_name", ""), key=f"am_sw_name_{selected_id}")
            e_sw_version = st.text_input("Software version", value=asset.get("software_version", ""), key=f"am_sw_ver_{selected_id}")
            e_cpe = st.text_input("CPE (single)", value=asset.get("cpe", ""), key=f"am_cpe_{selected_id}", help="Optional: a primary CPE 2.3 URI. Use Software Inventory for multiple CPEs.")

            submitted = st.form_submit_button("Save changes", type="primary", icon=":material/save:")

        if submitted:
            updates = {
                "asset_name": e_name,
                "description": e_desc,
                "ip_address": e_ip,
                "owner": e_owner,
                "owner_contact": e_owner_contact,
                "business_function": e_business_function,
                "vendor": e_vendor,
                "asset_type": e_type,
                "business_criticality": e_crit,
                "data_sensitivity": e_sens,
                "environment": e_env,
                "network_zone": e_zone,
                "internet_exposed": e_exp,
                "data_classification": e_data_class,
                "confidentiality_requirement": e_cr,
                "integrity_requirement": e_ir,
                "availability_requirement": e_ar,
                "software_name": e_sw_name,
                "software_version": e_sw_version,
                "cpe": e_cpe,
            }
            if e_cpe and not is_valid_cpe(e_cpe):
                st.error(f"Primary CPE is not a valid CPE 2.3 URI: {e_cpe}")
            elif not vector_store:
                st.error("ChromaDB is not ready; cannot save changes.")
            else:
                # Preserve existing software inventory (managed in the other tab).
                existing_inventory = load_software_inventory(asset)
                updates[SOFTWARE_INVENTORY_FIELD] = save_software_inventory(existing_inventory)
                if update_asset_metadata(vector_store, selected_id, updates):
                    load_all_assets_metadata.clear()
                    st.success("Asset updated. Changes are reflected immediately in retrieval and analysis.")
                    st.rerun()
                else:
                    st.error("Failed to update asset in ChromaDB.")

        with st.expander("Complete metadata (read only)"):
            st.json(asset)

    with tab_software:
        st.markdown("**Installed software records**")
        st.caption("Add software products installed on this asset. Each record may include a valid CPE 2.3 URI for applicability matching.")
        software = load_software_inventory(asset)
        df_data = [
            {"name": r.get("name", ""), "version": r.get("version", ""), "cpe": r.get("cpe", "")}
            for r in software
        ] if software else [{"name": "", "version": "", "cpe": ""}]

        edited = st.data_editor(
            df_data,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Software name", required=True),
                "version": st.column_config.TextColumn("Version"),
                "cpe": st.column_config.TextColumn("CPE 2.3 URI", help="e.g. cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
            },
            hide_index=True,
            key=f"sw_editor_{selected_id}",
        )

        if st.button("Save software inventory", icon=":material/save:", key=f"sw_save_{selected_id}"):
            cleaned = []
            errors = []
            for idx, row in enumerate(edited):
                name = str(row.get("name") or "").strip()
                cpe = str(row.get("cpe") or "").strip()
                if not name:
                    continue
                if cpe and not is_valid_cpe(cpe):
                    errors.append(f"Row {idx + 1}: invalid CPE '{cpe}'")
                cleaned.append({
                    "name": name,
                    "version": str(row.get("version") or "").strip(),
                    "cpe": cpe,
                })
            if errors:
                for err in errors:
                    st.error(err)
            elif not vector_store:
                st.error("ChromaDB is not ready; cannot save changes.")
            else:
                updates = {SOFTWARE_INVENTORY_FIELD: save_software_inventory(cleaned)}
                if update_asset_metadata(vector_store, selected_id, updates):
                    load_all_assets_metadata.clear()
                    st.success("Software inventory saved.")
                    st.rerun()
                else:
                    st.error("Failed to update software inventory in ChromaDB.")

    with st.container(horizontal=True):
        if st.button("Analyze vulnerability", icon=":material/biotech:", key=f"analyze_detail_{selected_id}", type="primary"):
            st.session_state.analysis_asset_id = selected_id
            st.session_state.analysis_asset_preselect = selected_id
            _navigate_to("Vulnerability Analysis")


def _render_asset_inventory():
    st.header(":material/table_chart: Asset Inventory")

    assets = load_all_assets_metadata()
    if not assets:
        st.info("No assets found. Run `python ingest_assets.py --v2` to populate the knowledge base.")
        return

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    search = f1.text_input("Search assets", placeholder="Name / ID / IP", key="inv_search")
    type_filter = f2.multiselect("Asset type", options=sorted({a.get("asset_type", "unknown") for a in assets}), key="inv_type")
    crit_filter = f3.multiselect("Business criticality", options=sorted({a.get("business_criticality", "unknown") for a in assets}), key="inv_crit")
    exposure_filter = f4.multiselect("Internet exposure", options=sorted({a.get("internet_exposed", "unknown") for a in assets}), key="inv_exp")
    zone_filter = f5.multiselect("Network zone", options=sorted({a.get("network_zone", "unknown") for a in assets}), key="inv_zone")

    filtered = assets
    if search:
        query = search.casefold()
        filtered = [
            a for a in filtered
            if any(query in str(a.get(field, "")).casefold() for field in ("asset_name", "asset_id", "ip_address"))
        ]
    if type_filter:
        filtered = [a for a in filtered if a.get("asset_type") in type_filter]
    if crit_filter:
        filtered = [a for a in filtered if a.get("business_criticality") in crit_filter]
    if exposure_filter:
        filtered = [a for a in filtered if a.get("internet_exposed") in exposure_filter]
    if zone_filter:
        filtered = [a for a in filtered if a.get("network_zone") in zone_filter]

    st.caption(f"Showing {len(filtered)} of {len(assets)} assets")
    table_rows = []
    for asset in filtered:
        software = load_software_inventory(asset)
        summaries = []
        for record in software:
            label = " ".join(part for part in (record.get("name", ""), record.get("version", "")) if part)
            if record.get("cpe"):
                label += f" ({record['cpe']})"
            if label:
                summaries.append(label)
        if not summaries:
            legacy = " ".join(str(asset.get(field, "")).strip() for field in ("software_name", "software_version") if asset.get(field))
            cpe = str(asset.get("cpe") or "").strip()
            summaries = [f"{legacy} ({cpe})" if legacy and cpe else legacy or cpe]
        software_summary = "; ".join(s for s in summaries if s) or "—"
        table_rows.append({
            "Asset ID": asset.get("asset_id", ""),
            "Asset Name": asset.get("asset_name", ""),
            "Asset Type": asset.get("asset_type", "unknown"),
            "Business Criticality": asset.get("business_criticality", "unknown"),
            "Internet Exposure": asset.get("internet_exposed", "unknown"),
            "Software/CPE Summary": software_summary,
            "View/Edit Action": "Select below",
            "Analyze Vulnerability Action": "Select below",
        })
    st.dataframe(table_rows, hide_index=True)

    filtered_ids = [str(a.get("asset_id") or "") for a in filtered if a.get("asset_id")]
    if not filtered_ids:
        st.session_state.inventory_selected_asset_id = ""
        st.info("No assets match the current search and filters.")
        return

    current_selection = st.session_state.get("inventory_asset_select", "")
    if current_selection not in filtered_ids:
        current_selection = st.session_state.get("inventory_selected_asset_id", "")
    if current_selection not in filtered_ids:
        current_selection = ""
    st.session_state.inventory_asset_select = current_selection
    selected_id = st.selectbox(
        "Select an asset for details or analysis",
        options=[""] + filtered_ids,
        format_func=lambda aid: "— select an asset —" if not aid else _asset_label(next(a for a in filtered if a.get("asset_id") == aid)),
        key="inventory_asset_select",
    )
    st.session_state.inventory_selected_asset_id = selected_id

    if selected_id and st.button("Analyze selected asset", icon=":material/biotech:", type="primary", key="inventory_analyze_action"):
        st.session_state.analysis_asset_id = selected_id
        st.session_state.analysis_asset_preselect = selected_id
        _navigate_to("Vulnerability Analysis")

    _render_asset_detail(assets)


def _render_vulnerability_analysis():
    st.header(":material/biotech: Vulnerability Analysis")

    assets = load_all_assets_metadata()
    asset_by_id = {a.get("asset_id", ""): a for a in assets}

    # A navigation request must override an already-instantiated selectbox
    # state, while normal reruns must preserve the user's manual selection.
    preselect_request = st.session_state.get("analysis_asset_preselect", "")
    if preselect_request:
        if preselect_request in asset_by_id:
            st.session_state.analysis_asset_select = preselect_request
            st.session_state.analysis_asset_id = preselect_request
        else:
            st.session_state.analysis_asset_select = ""
        st.session_state.analysis_asset_preselect = ""

    options = [""] + [a.get("asset_id", "") for a in assets]
    if "analysis_asset_select" not in st.session_state:
        initial_asset_id = st.session_state.analysis_asset_id
        st.session_state.analysis_asset_select = initial_asset_id if initial_asset_id in options else ""
    elif st.session_state.analysis_asset_select not in options:
        st.session_state.analysis_asset_select = ""

    selected_asset_id = st.selectbox(
        "Select asset",
        options=options,
        format_func=lambda aid: "— select an asset —" if not aid else _asset_label(asset_by_id[aid]),
        key="analysis_asset_select",
    )

    with st.form("vulnerability_form"):
        cve_input = st.text_input("CVE ID", placeholder="e.g. CVE-2021-44228", key="cve_input")
        with st.expander("Advanced settings"):
            retrieval_query_override = st.text_input(
                "Asset retrieval query (optional)",
                placeholder="Defaults to the selected asset name",
                key="analysis_retrieval_query",
                help="Used to show semantic retrieval candidates. The selected stable asset ID remains the context source.",
            )
            base_vector_override = st.text_input(
                "Base vector override (optional)",
                placeholder="Defaults to the CVSS v3 vector returned by NVD",
                key="analysis_base_vector_override",
                help="Only use a complete, validated CVSS v3 vector when supplying an explicit override.",
            )
        submitted = st.form_submit_button("Run Analysis", type="primary", icon=":material/play_arrow:")

    st.subheader("Analysis results")
    result_container = st.container()

    if submitted:
        st.session_state.analysis_record = None
        st.session_state.analysis_result = None
        cve_id = cve_input.strip().upper()
        if not selected_asset_id:
            st.error("Select an asset before running the analysis.")
        elif not cve_id:
            st.error("Enter a CVE ID before running the analysis.")
        elif not api_key:
            st.error("Please enter a Gemini API Key in the sidebar.")
        elif not vector_store:
            st.error(f"ChromaDB is not ready. Run `python ingest_assets.py --v2` to populate '{COLLECTION_NAME_V2}'.")
        else:
            with st.spinner("Fetching vulnerability data from NVD..."):
                nvd_data, nvd_message = fetch_cve_from_nvd(cve_id)

            if not nvd_data:
                st.error(f"NVD lookup failed: {nvd_message}")
            else:
                base_vector = base_vector_override.strip() or nvd_data.get("vector_string", "")
                if not base_vector:
                    st.error("NVD did not provide a CVSS v3 vector. Add a complete vector override in Advanced settings to calculate CVSS.")
                else:
                    try:
                        base_cvss_obj = CVSS3(base_vector)
                        calculated_base_score = base_cvss_obj.scores()[0]
                        base_score = (
                            float(nvd_data["base_score"])
                            if not base_vector_override.strip()
                            and nvd_data.get("base_score") is not None
                            else calculated_base_score
                        )
                        base_severity = base_cvss_obj.severities()[0]
                    except Exception as e:
                        st.error(f"Invalid CVSS v3 base vector: {e}")
                        base_cvss_obj = None

                    if base_cvss_obj is not None:
                        asset_source = get_asset_by_id(vector_store, selected_asset_id)
                        if not asset_source or str(asset_source.get("asset_id") or "") != selected_asset_id:
                            st.error(f"Could not load asset `{selected_asset_id}` from ChromaDB by stable asset ID.")
                        else:
                            asset_metadata = filtered_metadata(asset_source)
                            retrieval_query = retrieval_query_override.strip() or str(asset_metadata.get("asset_name") or selected_asset_id)
                            with st.spinner("Loading selected asset context and retrieval evidence..."):
                                retrieval = retrieve_asset_context(
                                    vector_store,
                                    retrieval_query,
                                    top_k=DEFAULT_TOP_K,
                                    confidence_threshold=confidence_threshold,
                                )

                            selected_candidate = next(
                                (candidate for candidate in retrieval.candidates
                                 if str(candidate.metadata.get("asset_id") or "") == selected_asset_id),
                                None,
                            )
                            retrieval_confidence = selected_candidate.confidence if selected_candidate else None
                            if retrieval_confidence is None:
                                st.warning(
                                    "The semantic retrieval did not return the selected asset as a candidate. "
                                    "Analysis continues with the exact asset metadata loaded by stable asset ID; "
                                    "no other asset is substituted."
                                )
                            elif retrieval_confidence < confidence_threshold:
                                st.warning(
                                    f"The selected asset's semantic confidence ({retrieval_confidence:.3f}) is below "
                                    f"the configured threshold ({confidence_threshold:.2f}). Analysis still uses "
                                    "the exact selected asset ID; other candidates are not substituted."
                                )

                            asset_cpes = collect_asset_cpes(asset_metadata)
                            applicability = determine_applicability(
                                asset_cpes, nvd_data.get("configurations")
                            )

                            with st.spinner("Running contextual assessment..."):
                                try:
                                    llm_result = analyze_with_llm(
                                        api_key,
                                        nvd_data.get("description", ""),
                                        base_vector,
                                        asset_metadata,
                                    )
                                    modified_vector_str = build_environmental_vector(base_vector, llm_result)
                                    cvss_obj = CVSS3(modified_vector_str)
                                    _, _, context_score = cvss_obj.scores()
                                    _, _, env_severity = cvss_obj.severities()

                                    # No verified relationships are supplied in the list-based workflow.
                                    graph_dependencies = {}
                                    priority_result = calculate_organizational_priority(
                                        asset_metadata,
                                        base_score=base_score,
                                        context_score=context_score,
                                        applicability=applicability,
                                        graph_dependencies=graph_dependencies,
                                        cve_id=cve_id,
                                    )

                                    with result_container:
                                        st.success("Analysis complete!")
                                        if applicability.status == ApplicabilityStatus.CONFIRMED:
                                            st.badge("CPE applicability: Confirmed", icon=":material/warning:", color="red")
                                        elif applicability.status == ApplicabilityStatus.NOT_APPLICABLE:
                                            st.badge("CPE applicability: Not Applicable", icon=":material/check_circle:", color="green")
                                        else:
                                            st.badge("CPE applicability: Unknown — not confirmed", icon=":material/help:", color="gray")
                                        st.caption(applicability.explanation)

                                        metrics = st.columns(4)
                                        metrics[0].metric("CVSS Base Score", f"{base_score:.1f}", delta="NVD" if not base_vector_override.strip() else "Validated override", delta_color="off", border=True)
                                        delta_score = round(context_score - base_score, 1)
                                        metrics[1].metric(
                                            "Context-Aware CVSS Score",
                                            f"{context_score:.1f}",
                                            delta=f"{delta_score:+.1f}",
                                            delta_color="inverse",
                                            border=True,
                                        )
                                        metrics[2].metric(
                                            "CVSS Severity",
                                            base_severity,
                                            delta=f"Context: {env_severity}",
                                            delta_color="off",
                                            border=True,
                                        )
                                        metrics[3].metric(
                                            "Organizational Priority",
                                            priority_result.priority or "INCOMPLETE",
                                            help=priority_result.explanation,
                                            border=True,
                                        )

                                        with st.expander("Assessment explanation and evidence", expanded=True):
                                            st.markdown("**Contextual requirements assessed by the model**")
                                            st.write(
                                                f"CR: `{llm_result.confidentiality_requirement.value}`  "
                                                f"IR: `{llm_result.integrity_requirement.value}`  "
                                                f"AR: `{llm_result.availability_requirement.value}`"
                                            )
                                            st.markdown("**LLM justification**")
                                            st.info(llm_result.justification)
                                            st.markdown("**Remediation**")
                                            st.write(llm_result.remediation)
                                            st.markdown("**Deterministic organizational-priority rubric**")
                                            st.write(priority_result.explanation)
                                            st.caption(
                                                "LLM priority suggestion (separate from the deterministic result): "
                                                f"{llm_result.organizational_priority.value}"
                                            )
                                            with st.expander("Priority rule contributions"):
                                                st.json([
                                                    {
                                                        "rule": contribution.rule,
                                                        "factor": contribution.factor,
                                                        "value": contribution.evidence_value,
                                                        "points": contribution.points,
                                                        "max_points": contribution.max_points,
                                                        "evidence_source": contribution.evidence_source,
                                                    }
                                                    for contribution in priority_result.rule_contributions
                                                ])
                                            if priority_result.missing_evidence:
                                                st.error("Missing evidence: " + ", ".join(priority_result.missing_evidence))
                                            if priority_result.unknown_factors:
                                                st.caption(
                                                    "Optional evidence not available (counted as 0, not low risk): "
                                                    + ", ".join(priority_result.unknown_factors)
                                                )
                                            st.caption("No verified dependency relationships were supplied; dependency points were not added.")

                                        with st.expander("CPE applicability evidence"):
                                            if applicability.evidence:
                                                st.json([dict(evidence) for evidence in applicability.evidence])
                                            else:
                                                st.write(f"Asset CPEs evaluated: {asset_cpes or 'none'}")

                                        with st.expander("Selected asset context and RAG evidence"):
                                            st.markdown(
                                                f"**Context source:** ChromaDB metadata for selected asset ID `{selected_asset_id}`  \n"
                                                f"**Semantic retrieval query:** {retrieval_query}  \n"
                                                f"**Selected asset confidence:** "
                                                + (f"{retrieval_confidence:.3f} (threshold {confidence_threshold:.2f})" if retrieval_confidence is not None else "not returned")
                                            )
                                            st.json(asset_metadata)
                                            st.markdown("**Top semantic candidates (evidence only; not substituted):**")
                                            if retrieval.candidates:
                                                for candidate in retrieval.candidates:
                                                    marker = "Selected asset" if str(candidate.metadata.get("asset_id") or "") == selected_asset_id else "Not selected"
                                                    st.write(
                                                        f"{marker}: `{candidate.metadata.get('asset_id', '?')}` "
                                                        f"{candidate.metadata.get('asset_name', '?')} — confidence={candidate.confidence:.3f}"
                                                    )
                                            else:
                                                st.caption("No semantic retrieval candidates were returned.")

                                        with st.expander("CVSS vectors"):
                                            st.code(
                                                f"Base: {base_vector}\nEnvironmental: {modified_vector_str}",
                                                language="text",
                                            )

                                    evidence = {
                                        "retrieval_query": retrieval_query,
                                        "retrieval_candidates": [
                                            {
                                                "asset_id": candidate.metadata.get("asset_id"),
                                                "asset_name": candidate.metadata.get("asset_name"),
                                                "confidence": candidate.confidence,
                                            }
                                            for candidate in retrieval.candidates
                                        ],
                                        "selected_asset_context": asset_metadata,
                                        "llm_justification": llm_result.justification,
                                        "llm_remediation": llm_result.remediation,
                                    }
                                    st.session_state.analysis_result = {
                                        "cve_id": cve_id,
                                        "asset_metadata": asset_metadata,
                                        "nvd_description": nvd_data.get("description", ""),
                                        "nvd_base_score": nvd_data.get("base_score"),
                                        "base_score": base_score,
                                        "context_score": context_score,
                                        "priority": priority_result.priority,
                                        "llm_priority": llm_result.organizational_priority.value,
                                        "vector_string": base_vector,
                                        "modified_vector": modified_vector_str,
                                        "retrieval_confidence": retrieval_confidence,
                                        "applicability": applicability.status.value,
                                        "applicability_explanation": applicability.explanation,
                                        "applicability_evidence": [dict(e) for e in applicability.evidence],
                                        "evidence": evidence,
                                    }
                                    st.session_state.analysis_record = record_from_analysis(
                                        cve_id=cve_id,
                                        asset_metadata=asset_metadata,
                                        base_score=base_score,
                                        context_score=context_score,
                                        priority=priority_result.priority,
                                        llm_priority=llm_result.organizational_priority.value,
                                        vector_string=base_vector,
                                        modified_vector=modified_vector_str,
                                        retrieval_confidence=retrieval_confidence,
                                        evidence=evidence,
                                        applicability=applicability.status.value,
                                        applicability_explanation=applicability.explanation,
                                        applicability_evidence=[dict(e) for e in applicability.evidence],
                                        priority_result=priority_result,
                                        graph_dependencies=graph_dependencies,
                                    )
                                except LlmAssessmentError as e:
                                    st.error(f"LLM assessment was rejected/invalid; CVSS calculation stopped: {e}")
                                except Exception as e:
                                    st.error(f"Error processing assessment: {e}")
    else:
        st.info("Select an asset, enter a CVE ID, and run the analysis.")

    # Save UI is rendered outside the analysis run so the button click survives
    # the rerun it triggers.
    if st.session_state.analysis_record:
        if st.button("Save to history", icon=":material/save:"):
            save_assessment(st.session_state.analysis_record)
            st.session_state.analysis_record = None
            st.success("Assessment saved to history.")
            st.rerun()


def _render_analysis_history():
    st.header(":material/history: Analysis History")

    history = load_history()
    if not history:
        st.info("No assessments recorded yet. Results can be saved from the Vulnerability Analysis page after a successful analysis.")
        return

    st.caption(f"{len(history)} assessment(s) recorded")

    for idx, record in enumerate(reversed(history)):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{record.cve_id}** on {record.asset_name or record.asset_id}")
                st.caption(f"{record.timestamp[:19].replace('T', ' ')}")
            with c2:
                st.badge(str(record.priority) if record.priority else "INCOMPLETE", color="blue")
            with c3:
                if st.button("Delete", key=f"del_hist_{idx}"):
                    delete_record(record)
                    st.rerun()

            with st.expander("Details"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Base CVSS", record.base_score, border=True)
                col_b.metric("Context score", record.context_score if record.context_score is not None else "—", border=True)
                col_c.metric("LLM priority", record.llm_priority, border=True)

                st.markdown("**Organizational Priority (deterministic):**")
                if record.priority_status:
                    st.write(f"`{record.priority or 'INCOMPLETE'}` ({record.priority_status}) — score {record.priority_score}/{100}")
                else:
                    st.write(f"`{record.priority or 'INCOMPLETE'}`")
                if record.priority_explanation:
                    st.info(record.priority_explanation)
                if record.priority_missing_evidence:
                    st.error("Missing evidence: " + ", ".join(record.priority_missing_evidence))
                if record.priority_unknown_factors:
                    st.caption("Optional evidence unavailable: " + ", ".join(record.priority_unknown_factors))
                if record.priority_contributions:
                    with st.expander("Rubric contributions"):
                        st.json(record.priority_contributions)
                if record.graph_dependencies:
                    with st.expander("Recorded dependency context"):
                        st.json(record.graph_dependencies)

                if record.applicability:
                    st.markdown("**CVE applicability:**")
                    st.write(f"`{record.applicability}` — {record.applicability_explanation or ''}")
                    if record.applicability_evidence:
                        with st.expander("CPE match evidence"):
                            st.json(record.applicability_evidence)
                st.markdown("**Original vector:**")
                st.code(record.vector_string, language="text")
                st.markdown("**Environmental vector:**")
                st.code(record.modified_vector, language="text")
                st.markdown("**Retrieval confidence:**")
                st.write(record.retrieval_confidence)
                if record.evidence and record.evidence.get("llm_justification"):
                    st.markdown("**LLM justification:**")
                    st.info(record.evidence["llm_justification"])


# ---------------------------------------------------------------------------
# 6. DISPATCH
# ---------------------------------------------------------------------------
if page == "Dashboard":
    _render_dashboard()
elif page == "Asset Inventory":
    _render_asset_inventory()
elif page == "Vulnerability Analysis":
    _render_vulnerability_analysis()
elif page == "Analysis History":
    _render_analysis_history()

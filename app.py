from dataclasses import asdict
from datetime import date
from html import escape
import json
import re
import pandas as pd
import streamlit as st
from cvss import CVSS3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_retrieval import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    filtered_metadata,
    get_asset_by_id,
    list_all_assets,
    retrieve_document_context,
    update_asset_metadata,
)
from llm_assessment import DEFAULT_MODEL, LlmAssessmentError, analyze_with_llm, build_model_context
from nvd_client import fetch_cve_from_nvd
from cvss_engine import build_environmental_vector, validate_base_vector
from organizational_priority import calculate_organizational_priority
from cve_applicability import determine_applicability
from asset_inventory import (
    SOFTWARE_INVENTORY_FIELD,
    collect_asset_cpes,
    inventory_is_verified,
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
from asset_schema_v2 import ALLOWED_VALUES, SCHEMA_VERSION, V2_COLUMNS, validate_row
from ingest_assets import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME_V2,
    EMBEDDING_MODEL_NAME,
    build_v2_document,
)

st.set_page_config(
    page_title="Asset Intelligence & Vulnerability Assessment",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_OPTIONS = [
    "Vulnerability Analysis",
    "Asset Inventory",
    "Analysis History",
]

# Native light/dark colors are controlled by .streamlit/config.toml.
# Keep this CSS layout-only; never override native Streamlit widget colors.
APP_LAYOUT_STYLES = """
<style>
/* A compact local research tool: small navigation, generous content width. */
[data-testid="stSidebar"] { min-width: 0; }
[data-testid="stSidebar"][data-collapsed="false"] { width: 240px !important; max-width: 240px; }
[data-testid="stSidebar"][data-collapsed="true"] {
    width: 0 !important; min-width: 0 !important; max-width: 0 !important;
    flex-basis: 0 !important; border: 0; overflow: hidden;
}
/* Keep the title close to the top; the native sidebar header still holds its collapse control. */
[data-testid="stSidebarUserContent"] { padding: 0 0 .5rem; }
[data-testid="stSidebarHeader"] { height: 2.25rem; min-height: 2.25rem; }
[data-testid="stLogoSpacer"] { height: 1.25rem; }
[data-testid="stSidebar"] h1 { font-size: 1.3rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    display: flex; width: 100%; min-height: 42px; box-sizing: border-box;
    align-items: center; padding: .5rem .75rem; border-radius: 7px; margin: 0;
    color: inherit;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: color-mix(in srgb, #2563eb 17%, transparent);
    color: inherit;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
    color: inherit;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    min-height: calc(100dvh - 3rem);
}
[data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_settings) {
    margin-top: auto;
}
.st-key-sidebar_settings { padding-top: 1.5rem; }
.st-key-sidebar_settings [data-testid="stExpander"] details {
    background: transparent; border: 0; box-shadow: none;
}
.st-key-sidebar_settings summary { min-height: 42px; padding: .5rem .75rem; }
.st-key-nav_page,
.st-key-nav_page [data-testid="stRadio"],
.st-key-nav_page [role="radiogroup"],
.st-key-nav_page [role="radiogroup"] > div {
    width: 100%; align-self: stretch;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div { width: 100%; min-width: 0; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child { display:none; }
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {
    display: flex; width: 100%; min-width: 0; align-items: center; gap: .65rem;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
    margin: 0; line-height: 1.3; white-space: nowrap;
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"]::before {
    font-family: 'Material Symbols Rounded'; font-size: 20px;
    line-height: 1; width: 20px; flex-shrink: 0; text-align: center;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(1) [data-testid="stMarkdownContainer"]::before { content: 'search'; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(2) [data-testid="stMarkdownContainer"]::before { content: 'database'; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child(3) [data-testid="stMarkdownContainer"]::before { content: 'history'; }
/* Streamlit 1.66+ calculates top clearance from theme.paddingTop,
   including the native three-dot toolbar. Do not set a competing CSS padding-top. */
[data-testid="stMainBlockContainer"] {
    width: 100%; max-width: none; margin-inline: 0;
    padding-left: clamp(1.25rem, 2.5vw, 3rem);
    padding-right: clamp(1.25rem, 2.5vw, 3rem);
}
[data-testid="stForm"] { padding: 1.2rem; }
/* Custom result blocks use transparent, inherited colors in both themes. */
.result-card {
    border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    border-radius: 8px; padding: 18px; min-height: 160px;
    background: transparent; color: inherit;
}
.result-card > strong { font-size: 1.45rem; }
.result-card.priority { background: color-mix(in srgb, #d73a49 7%, transparent); }
.result-card.priority-p2 { background: color-mix(in srgb, #e87916 7%, transparent); }
.result-card.priority-p3 { background: color-mix(in srgb, #d4a72c 8%, transparent); }
.result-card.priority-p4 { background: color-mix(in srgb, #2563eb 7%, transparent); }
.result-card.priority strong { font-size: 1.65rem; }
.result-card .card-label { font-size: .85rem; margin-bottom: 7px; }
.result-card .score { font-size: 2rem; line-height: 1.25; font-weight: 700; }
.result-card .card-note { font-size: .78rem; opacity: .76; margin-top: 8px; }
.requirement-card { min-height: 92px; }
.requirement-card strong {
    display: inline-block; margin-top: 9px; border-radius: 5px;
    padding: 3px 10px; font-size: .95rem;
}
.requirement-high strong, .severity-critical, .severity-high { background: #fee2e2; color: #991b1b; }
.requirement-medium strong, .severity-medium { background: #fef3c7; color: #92400e; }
.requirement-low strong, .severity-low { background: #dcfce7; color: #166534; }
.severity-badge {
    display: inline-block; border-radius: 4px; padding: 2px 8px;
    font-size: .8rem; background: color-mix(in srgb, currentColor 9%, transparent);
}
.summary-card {
    border: 1px solid color-mix(in srgb, #2563eb 26%, transparent);
    border-radius: 8px; padding: 18px;
    background: color-mix(in srgb, #2563eb 7%, transparent);
    color: inherit; margin: 10px 0;
}
.summary-card p { color: inherit; white-space: pre-wrap; font-size: .93rem; line-height: 1.7; margin-bottom: 0; }
.db-status { font-size: .85rem; padding: .7rem 0; }
.db-ready { color: #16a34a; }
.db-error { color: #ef4444; }
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Keep touch-friendly native sidebar controls on small screens. */
    [data-testid="stSidebarHeader"] { height: 3rem; min-height: 3rem; }
}
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
        create_collection_if_not_exists=False,
    )


@st.cache_resource
def load_context_db():
    """Document evidence lives separately from stable asset inventory records."""
    return Chroma(
        collection_name="organizational_context",
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
        create_collection_if_not_exists=False,
    )


try:
    vector_store = load_vector_db()
    inventory_count = len(vector_store.get(include=["metadatas"]).get("ids") or [])
    db_status = "Ready" if inventory_count else "Empty inventory"
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


_init_session_state()


# Handle cross-page navigation triggered by buttons rendered later in the app.
# We use a query parameter because Streamlit disallows modifying a widget's
# session-state key after that widget has been instantiated in the same run.
_nav_target = st.query_params.get("nav")
if _nav_target:
    if _nav_target in PAGE_OPTIONS:
        st.session_state.nav_page = _nav_target
    del st.query_params["nav"]
if st.session_state.get("nav_page") not in PAGE_OPTIONS:
    st.session_state.nav_page = "Vulnerability Analysis"


# ---------------------------------------------------------------------------
# 3. SHARED HELPERS
# ---------------------------------------------------------------------------
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
    with st.container(key="sidebar_settings"):
        with st.expander(":material/settings: Settings", expanded=False):
            api_key = st.text_input("Gemini API Key", type="password", key="gemini_api_key")
            model_name = st.text_input("Gemini model", value=DEFAULT_MODEL, key="gemini_model")
            confidence_threshold = st.slider(
                "Retrieval relevance threshold", min_value=0.0, max_value=1.0,
                value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
                help="Only relevant documents for the selected asset may enter model context.",
            )
            st.caption("Light / Dark theme: use the ⋮ menu at the top right.")
            st.caption(f"Inventory collection: {COLLECTION_NAME_V2}")
        status_class = "db-ready" if db_status == "Ready" else "db-error"
        st.html(f'<div class="db-status {status_class}">● ChromaDB {escape(db_status)}</div>')

st.html(APP_LAYOUT_STYLES)


# ---------------------------------------------------------------------------
# 5. PAGE RENDERERS
# ---------------------------------------------------------------------------
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
            if e_cpe != asset.get("cpe", ""):
                updates["inventory_verified"] = "false"
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
        verified_inventory = st.checkbox(
            "I verified this is the complete installed-product inventory for this asset.",
            value=inventory_is_verified(asset), key=f"sw_verified_{selected_id}",
            help="Without explicit completeness verification, an absent product remains Unknown rather than Not Applicable.",
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
                if verified_inventory and not is_valid_cpe(cpe):
                    errors.append(f"Row {idx + 1}: complete verified inventory requires a valid CPE for each software record.")
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
                updates = {SOFTWARE_INVENTORY_FIELD: save_software_inventory(cleaned),
                           "inventory_verified": "true" if verified_inventory else "false"}
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
            st.session_state.analysis_result = None
            st.session_state.analysis_record = None
            _navigate_to("Vulnerability Analysis")


def _render_add_asset():
    """Create only after an explicit submit; never overwrite an existing ID."""
    with st.form("add_asset_form"):
        st.subheader("New asset")
        row = {field: "unknown" for field in V2_COLUMNS}
        row.update(schema_version=SCHEMA_VERSION, software_name="", software_version="", cpe="")
        left, right = st.columns(2)
        for column, fields in (
            (left, ("asset_id", "asset_name", "ip_address", "description", "source", "source_reference")),
            (right, ("owner", "business_function")),
        ):
            for field in fields:
                row[field] = column.text_input(field.replace("_", " ").title(), key=f"new_{field}")
        row["last_updated"] = str(st.date_input("Last updated", value=date.today(), key="new_updated"))
        for field in (
            "asset_type", "business_criticality", "data_sensitivity", "environment",
            "network_zone", "internet_exposed", "confidentiality_requirement",
            "integrity_requirement", "availability_requirement", "is_synthetic",
        ):
            row[field] = st.selectbox(field.replace("_", " ").title(),
                                      ALLOWED_VALUES[field], key=f"new_{field}")
        submitted = st.form_submit_button("Create asset", type="primary")
    if not submitted:
        return
    errors = validate_row(row)
    if row["asset_id"] != row["asset_id"].strip():
        errors.append("Asset ID must not contain surrounding whitespace.")
    if not vector_store:
        errors.append("ChromaDB is not ready.")
    elif get_asset_by_id(vector_store, row["asset_id"]):
        errors.append("Asset ID already exists. Use the editor instead; existing records are never overwritten here.")
    if errors:
        for error in errors:
            st.error(error)
        return
    try:
        document = build_v2_document(row)
        # Chroma collection.add, unlike upsert/add_documents, rejects duplicate IDs.
        vector_store._collection.add(ids=[f"asset:{row['asset_id']}"],
                                     documents=[document.page_content], metadatas=[document.metadata],
                                     embeddings=vector_store.embeddings.embed_documents([document.page_content]))
        stored = get_asset_by_id(vector_store, row["asset_id"])
        if not stored or stored.get("asset_name") != row["asset_name"]:
            st.error("Asset write could not be verified. Refresh inventory before retrying.")
            return
        load_all_assets_metadata.clear()
        st.session_state.show_add_asset = False
        st.session_state.inventory_selected_asset_id = row["asset_id"]
        st.rerun()
    except Exception as exc:
        st.error(f"Asset creation failed: {exc}")


def _render_asset_inventory():
    with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="inventory_header"):
        st.header(":material/database: Asset Inventory")
        if st.button("Add Asset", icon=":material/add:", type="primary", key="add_asset"):
            st.session_state.show_add_asset = not st.session_state.get("show_add_asset", False)
    st.caption("View and manage asset metadata used for contextual analysis.")
    if st.session_state.get("show_add_asset"):
        _render_add_asset()

    assets = load_all_assets_metadata()
    if not assets:
        st.info("No assets found. Run `python ingest_assets.py --v2` to populate the knowledge base.")
        return

    search = st.text_input("Search assets", placeholder="Search by name, ID, or IP…", key="inv_search", label_visibility="collapsed")
    with st.expander("Inventory filters"):
        f1, f2, f3, f4 = st.columns(4)
        type_filter = f1.multiselect("Asset type", options=sorted({a.get("asset_type", "unknown") for a in assets}), key="inv_type")
        crit_filter = f2.multiselect("Business criticality", options=sorted({a.get("business_criticality", "unknown") for a in assets}), key="inv_crit")
        exposure_filter = f3.multiselect("Internet exposure", options=sorted({a.get("internet_exposed", "unknown") for a in assets}), key="inv_exp")
        zone_filter = f4.multiselect("Network zone", options=sorted({a.get("network_zone", "unknown") for a in assets}), key="inv_zone")

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
        })
    table = pd.DataFrame(table_rows, columns=["Asset ID", "Asset Name", "Asset Type", "Business Criticality", "Internet Exposure", "Software/CPE Summary"])
    st.dataframe(table, hide_index=True, width="stretch")

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
        st.session_state.analysis_result = None
        st.session_state.analysis_record = None
        _navigate_to("Vulnerability Analysis")

    _render_asset_detail(assets)


def _new_analysis():
    st.session_state.analysis_result = None
    st.session_state.analysis_record = None
    st.session_state.analysis_saved = False
    st.session_state.cloud_consent = False


def _render_analysis_result():
    """Render exclusively from session state so reruns never erase the results."""
    result = st.session_state.analysis_result
    asset = result["asset_metadata"]
    st.caption(f"{_asset_label(asset)}  |  {result['cve_id']}")
    priority, base, environmental = st.columns([1.4, 1, 1])
    labels = {"P1": "Urgent", "P2": "High", "P3": "Medium", "P4": "Low"}
    remediation_priority = result.get("priority")
    investigation = result.get("investigation_priority")
    if remediation_priority:
        priority_text = f"{remediation_priority} — {labels.get(remediation_priority, '')}"
        priority_note = "Organizational remediation priority"
    elif investigation:
        priority_text = f"{investigation} — Investigate"
        priority_note = "Investigation priority only · applicability not confirmed"
    else:
        priority_text = "Not Applicable" if result.get("applicability") == "Not Applicable" else "Undetermined"
        priority_note = "No organizational remediation priority assigned"
    priority_class = {"P1": " priority", "P2": " priority priority-p2", "P3": " priority priority-p3", "P4": " priority priority-p4"}.get(remediation_priority, "")
    priority.html(
        '<div class="result-card' + priority_class + '">'
        '<div class="card-label">Organizational Priority</div>'
        f'<strong>{escape(priority_text)}</strong><div class="card-note">{escape(priority_note)}</div></div>'
    )
    for column, label, value, severity, source in (
        (base, "Base CVSS", result.get("base_score"), result.get("base_severity", "—"), result.get("base_source", "NVD")),
        (environmental, "Environmental CVSS", result.get("context_score"), result.get("environmental_severity", "—"), "Deterministic · CR / IR / AR"),
    ):
        score_text = f"{value:.1f}" if value is not None else "—"
        severity_class = {"Critical": "severity-critical", "High": "severity-high", "Medium": "severity-medium", "Low": "severity-low"}.get(severity, "")
        column.html(f'<div class="result-card"><div class="card-label">{escape(label)}</div>'
                    f'<div class="score">{escape(score_text)}</div><span class="severity-badge {severity_class}">{escape(str(severity))}</span>'
                    f'<div class="card-note">Technical severity · {escape(str(source))}</div></div>')
    with st.container(border=True):
        status = result.get("applicability", "Unknown")
        with st.container(horizontal=True):
            st.markdown("**Applicability**")
            st.badge(status, color="green" if status == "Confirmed" else "gray")
        if status == "Unknown":
            st.caption("Unknown — not confirmed. Investigation priority is not remediation priority.")
        st.write(result.get("applicability_explanation", ""))
        if status == "Not Applicable":
            st.caption("CVSS describes technical vulnerability severity; it does not imply this asset is affected.")
    st.markdown("**Environmental Requirements**")
    model_output = result.get("model_output", {})
    for column, label, field in zip(st.columns(3), ("Confidentiality (CR)", "Integrity (IR)", "Availability (AR)"),
                                    ("confidentiality_requirement", "integrity_requirement", "availability_requirement")):
        code = model_output.get(field, "X")
        value = {"H": "High", "M": "Medium", "L": "Low", "X": "Not Defined"}.get(code, "Not Defined")
        requirement_class = {"H": "requirement-high", "M": "requirement-medium", "L": "requirement-low"}.get(code, "")
        column.html(f'<div class="result-card requirement-card {requirement_class}"><div class="card-label">{escape(label)}</div>'
                    f'<strong>{escape(value)}</strong></div>')
    st.html('<div class="summary-card"><strong>AI Assessment Summary</strong>'
            f'<p>{escape(str(model_output.get("justification", "No model summary available.")))}</p></div>')
    with st.expander("Justification Details"):
        st.write(result.get("priority_explanation", ""))
        st.caption(f"Priority status: {result.get('priority_status', 'Unknown')}")
        st.markdown("**Independent CR / IR / AR reasons and cited evidence**")
        st.json(model_output.get("metric_reasons", {}))
        st.markdown("**Remediation**")
        st.write(model_output.get("remediation", ""))
        st.caption(f"Model priority suggestion (not the deterministic result): {result.get('llm_priority', '—')}")
        st.json(result.get("priority_result", {}))
    with st.expander("Retrieved Context (RAG)"):
        evidence = result.get("evidence", {})
        retrieval = evidence.get("document_retrieval", {})
        if not retrieval.get("accepted"):
            st.warning("No accepted document evidence; this assessment used exact asset metadata only.")
        if retrieval.get("error"):
            st.warning(retrieval["error"])
        st.caption("Accepted evidence was asset-filtered. Rejected candidates never entered model context.")
        st.json(retrieval)
        st.markdown("**Exact model context (sanitized)**")
        st.json(evidence.get("model_context", {}))
    with st.expander("Technical Details (CVSS, CPE, etc.)"):
        st.code(f"Base: {result.get('vector_string', '')}\nEnvironmental: {result.get('modified_vector', '')}", language="text")
        st.write(result.get("nvd_description", ""))
        st.json(result.get("applicability_evidence", []))
        st.markdown("**NVD source and snapshot provenance**")
        st.json(result.get("evidence", {}).get("nvd", {}))
        st.markdown("**Validated model output**")
        st.json(model_output)
    record = st.session_state.analysis_record
    if st.session_state.get("analysis_saved"):
        st.success("Assessment saved to history.")
    elif record and st.button("Save to history", icon=":material/save:", key="save_analysis"):
        try:
            save_assessment(record)
            if not any(item.id == record.id for item in load_history()):
                st.error("History save could not be verified.")
            else:
                st.session_state.analysis_saved = True
                st.rerun()
        except Exception as exc:
            st.error(f"History save failed: {exc}")


def _render_vulnerability_analysis():
    if st.session_state.analysis_result and not st.session_state.get("analysis_asset_preselect"):
        with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="analysis_header"):
            st.header(":material/biotech: Vulnerability Analysis")
            st.button("New Analysis", icon=":material/arrow_back:", key="new_analysis", on_click=_new_analysis)
        _render_analysis_result()
        return
    st.header(":material/biotech: Vulnerability Analysis")
    st.caption("Select an asset and enter a CVE ID to analyze its impact on your organization.")

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

    with st.form("vulnerability_form"):
        selected_asset_id = st.selectbox(
            "Asset", options=options,
            format_func=lambda aid: "— select an asset —" if not aid else _asset_label(asset_by_id[aid]),
            key="analysis_asset_select",
        )
        cve_input = st.text_input("CVE ID", placeholder="e.g. CVE-2021-44228", key="cve_input")
        with st.expander("Advanced settings"):
            retrieval_query_override = st.text_input(
                "Document retrieval query (optional)",
                placeholder="Defaults to CVE description and selected asset",
                key="analysis_retrieval_query",
                help="Only documents filtered to the selected stable asset ID are eligible as evidence.",
            )
            base_vector_override = st.text_input(
                "Base vector override (optional)",
                placeholder="Defaults to the CVSS v3 vector returned by NVD",
                key="analysis_base_vector_override",
                help="Only use a complete, validated CVSS v3 vector when supplying an explicit override.",
            )
            override_reason = st.text_input(
                "Override reason", key="analysis_override_reason",
                help="Required for a manual vector; overrides are excluded from the main NVD-only test set.",
            )
        cloud_consent = st.checkbox(
            "I consent to send the selected asset context and accepted document evidence to Gemini.",
            value=False, key="cloud_consent",
            help="Organizational context leaves this device. No cloud request is made without explicit consent.",
        )
        submitted = st.form_submit_button("Run Analysis", type="primary", icon=":material/play_arrow:", width="stretch")

    if submitted:
        st.session_state.analysis_record = None
        st.session_state.analysis_result = None
        cve_id = cve_input
        if not selected_asset_id:
            st.error("Select an asset before running the analysis.")
        elif not re.fullmatch(r"CVE-\d{4}-\d{4,}", cve_id):
            st.error("Enter a literal CVE ID in CVE-YYYY-NNNN format (uppercase, no whitespace).")
        elif not api_key:
            st.error("Please enter a Gemini API Key in Settings.")
        elif not model_name.strip():
            st.error("Enter a Gemini model in Settings.")
        elif not cloud_consent:
            st.error("Explicit cloud consent is required before submitting asset context to Gemini.")
        elif base_vector_override and not override_reason.strip():
            st.error("A reason is required for a manual Base vector override.")
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
                        validate_base_vector(base_vector)
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
                            retrieval_query = retrieval_query_override.strip() or (
                                f"{cve_id} {nvd_data.get('description', '')} {asset_metadata.get('asset_name', selected_asset_id)}"
                            )
                            with st.spinner("Loading selected asset context and retrieval evidence..."):
                                try:
                                    context_store = load_context_db()
                                except Exception as exc:
                                    context_store = None
                                    st.warning(f"Organizational documents unavailable: {exc}")
                                retrieval = retrieve_document_context(
                                    context_store, retrieval_query, selected_asset_id,
                                    top_k=DEFAULT_TOP_K, relevance_threshold=confidence_threshold,
                                )
                            retrieval_confidence = max(
                                (item.relevance_score for item in retrieval.accepted if item.relevance_score is not None), default=None,
                            )
                            if not retrieval.accepted:
                                st.warning("No accepted document evidence. The model receives only the exact selected asset metadata; RAG evidence is absent, not substituted.")
                            model_context = build_model_context(
                                asset_metadata, retrieved_evidence=[item.to_dict() for item in retrieval.accepted],
                            )

                            asset_cpes = collect_asset_cpes(asset_metadata)
                            applicability = determine_applicability(
                                asset_cpes, nvd_data.get("configurations"),
                                inventory_verified=inventory_is_verified(asset_source),
                            )
                            model_description = nvd_data.get("description", "")[:20000]

                            with st.spinner("Running contextual assessment..."):
                                try:
                                    llm_result = analyze_with_llm(
                                        api_key,
                                        model_description,
                                        base_vector,
                                        model_context,
                                        model=model_name.strip(),
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

                                    evidence = {
                                        "retrieval_query": retrieval_query,
                                        "document_retrieval": retrieval.to_dict(),
                                        "model_context": model_context,
                                        "selected_asset_context": asset_metadata,
                                        "nvd": nvd_data,
                                        "llm_justification": llm_result.justification,
                                        "llm_remediation": llm_result.remediation,
                                    }
                                    model_output = llm_result.model_dump(mode="json")
                                    provenance = {
                                        "model": model_name.strip(), "cloud_consent": True,
                                        "vector_origin": "manual_override" if base_vector_override else "nvd",
                                        "override_reason": override_reason if base_vector_override else None,
                                        "main_test_set_eligible": not bool(base_vector_override),
                                        "context_mode": "hybrid_rag" if model_context.get("retrieved_evidence") else "direct_context_no_document_evidence",
                                        "nvd": nvd_data, "base_vector_override": base_vector_override.strip(),
                                        "model_input": {"cve_desc": model_description,
                                                        "original_vector": base_vector, "asset_context": model_context},
                                        "model_payload": {"cve_description": model_description,
                                                          "original_nvd_vector": base_vector,
                                                          "organizational_context": model_context},
                                        "retrieval": retrieval.to_dict(),
                                        "rubric_version": getattr(priority_result, "rubric_version", None),
                                    }
                                    st.session_state.analysis_result = {
                                        "cve_id": cve_id,
                                        "asset_metadata": asset_metadata,
                                        "nvd_description": nvd_data.get("description", ""),
                                        "nvd_base_score": nvd_data.get("base_score"),
                                        "base_score": base_score,
                                        "context_score": context_score,
                                        "base_severity": base_severity,
                                        "environmental_severity": env_severity,
                                        "priority": priority_result.priority,
                                        "investigation_priority": getattr(priority_result, "investigation_priority", None),
                                        "priority_status": priority_result.status,
                                        "priority_explanation": priority_result.explanation,
                                        "priority_result": asdict(priority_result),
                                        "model_output": model_output,
                                        "base_source": "Validated override" if base_vector_override.strip() else "NVD",
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
                                        model_output=model_output,
                                        provenance=provenance,
                                    )
                                    st.session_state.analysis_saved = False
                                    st.rerun()
                                except LlmAssessmentError as e:
                                    st.error(f"LLM assessment was rejected/invalid; CVSS calculation stopped: {e}")
                                except Exception as e:
                                    st.error(f"Error processing assessment: {e}")
    else:
        st.info("Select an asset, enter a CVE ID, and run the analysis.")

def _render_analysis_history():
    st.header(":material/history: Analysis History")
    st.caption("View previous vulnerability assessments.")

    history = load_history()
    if not history:
        st.info("No assessments recorded yet. Results can be saved from the Vulnerability Analysis page after a successful analysis.")
        return

    st.caption(f"{len(history)} assessment(s) recorded")

    history_table = pd.DataFrame([
        {"Date": record.timestamp[:19].replace("T", " "), "CVE ID": record.cve_id,
         "Asset": record.asset_id, "Priority": record.priority or "—",
         "Investigation Priority": getattr(record, "investigation_priority", None) or "—",
         "Base CVSS": record.base_score, "Env CVSS": record.context_score,
         "Applicability": record.applicability or "Unknown"}
        for record in reversed(history)
    ])
    st.dataframe(history_table, hide_index=True, width="stretch")
    st.download_button("Export history (JSON)", data=json.dumps([asdict(record) for record in history], indent=2),
                       file_name="assessment_history.json", mime="application/json", key="export_history")
    ordered = list(reversed(history))
    if st.session_state.get("history_selected", 0) not in range(len(ordered)):
        del st.session_state["history_selected"]
    selected = st.selectbox("Assessment details", options=range(len(ordered)), key="history_selected",
                            format_func=lambda index: f"{ordered[index].timestamp[:19]} · {ordered[index].cve_id} · {ordered[index].asset_id}")

    for idx, record in [(selected, ordered[selected])]:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{record.cve_id}** on {record.asset_name or record.asset_id}")
                st.caption(f"{record.timestamp[:19].replace('T', ' ')}")
            with c2:
                st.badge(str(record.priority) if record.priority else "No remediation priority", color="blue")
            with c3:
                if st.button("Delete", key=f"del_hist_{idx}"):
                    delete_record(record)
                    remaining = load_history()
                    still_present = any(item.id == record.id for item in remaining) if record.id else any(
                        asdict(item) == asdict(record) for item in remaining
                    )
                    if still_present:
                        st.error("History deletion could not be verified.")
                    else:
                        st.rerun()

            with st.expander("Details"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Base CVSS", record.base_score, border=True)
                col_b.metric("Environmental CVSS", record.context_score if record.context_score is not None else "—", border=True)
                col_c.metric("LLM priority", record.llm_priority, border=True)

                st.markdown("**Organizational Priority (deterministic):**")
                if getattr(record, "investigation_priority", None):
                    st.write(f"Investigation priority only: {record.investigation_priority}; not a remediation priority.")
                if record.priority_status:
                    st.write(f"{record.priority or 'Not assigned'} ({record.priority_status}) — rubric score {record.priority_score}/100")
                else:
                    st.write(record.priority or "Not assigned")
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
                st.markdown("**Evidence and model input**")
                st.json(record.evidence or {})
                st.markdown("**Validated model output**")
                st.json(getattr(record, "model_output", None) or {})
                st.markdown("**Provenance**")
                st.json(getattr(record, "provenance", None) or {})


# ---------------------------------------------------------------------------
# 6. DISPATCH
# ---------------------------------------------------------------------------
if page == "Asset Inventory":
    _render_asset_inventory()
elif page == "Vulnerability Analysis":
    _render_vulnerability_analysis()
elif page == "Analysis History":
    _render_analysis_history()

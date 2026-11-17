from dataclasses import asdict
from datetime import date
from html import escape
import json
import hashlib
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from skripsi_rag.rag_retrieval import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    get_asset_by_id,
    list_all_assets,
    update_asset_metadata,
)
from skripsi_rag.llm_assessment import DEFAULT_MODEL
from skripsi_rag.va_import import parse_acunetix_csv, findings_preview, results_csv
from skripsi_rag.va_batch import run_va_batch, batch_rows, case_history_record
from skripsi_rag.asset_inventory import (
    SOFTWARE_INVENTORY_FIELD,
    inventory_is_verified,
    is_valid_cpe,
    load_software_inventory,
    save_software_inventory,
)
from skripsi_rag.assessment_history import (
    HistoryReadError,
    delete_record,
    load_history,
    save_assessment,
)
from skripsi_rag.asset_schema_v2 import ALLOWED_VALUES, SCHEMA_VERSION, V2_COLUMNS, validate_row
from skripsi_rag.ingest_assets import (
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


def _init_session_state():
    st.session_state.setdefault("analysis_asset_id", "")
    st.session_state.setdefault("analysis_asset_preselect", "")
    st.session_state.setdefault("inventory_selected_asset_id", "")
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("analysis_record", None)


_init_session_state()


# Query parameters avoid changing a widget's state after Streamlit has rendered it.
_nav_target = st.query_params.get("nav")
if _nav_target:
    if _nav_target in PAGE_OPTIONS:
        st.session_state.nav_page = _nav_target
    del st.query_params["nav"]
if st.session_state.get("nav_page") not in PAGE_OPTIONS:
    st.session_state.nav_page = "Vulnerability Analysis"


def _asset_label(asset: dict) -> str:
    asset_id = asset.get("asset_id", "?")
    name = asset.get("asset_name", asset_id)
    return f"{asset_id} — {name}"


def _navigate_to(page: str):
    st.query_params.nav = page
    st.rerun()


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


def _render_software_inventory(asset: dict, selected_id: str):
    st.markdown("**Installed software records**")
    st.caption("Add software products installed on this asset. Each record may include a valid CPE 2.3 URI for applicability matching.")
    try:
        software = load_software_inventory(asset, strict=True)
        if any(not isinstance(record.get("name"), str) or not record["name"].strip() for record in software):
            raise ValueError("Software inventory contains a record without a software name.")
    except ValueError as exc:
        st.error(f"Software inventory could not be loaded: {exc}")
        st.caption("Repair the inventory source before editing software records. Metadata edits preserve the stored inventory.")
        return
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
            if cpe and cpe != "unknown" and not is_valid_cpe(cpe):
                errors.append(f"Row {idx + 1}: invalid CPE '{cpe}'")
            if verified_inventory and not is_valid_cpe(cpe):
                errors.append(f"Row {idx + 1}: complete verified inventory requires a valid CPE for each software record.")
            original = next((item for item in software if item.get("name") == name
                and item.get("version", "") == str(row.get("version") or "").strip()
                and item.get("cpe", "") == cpe), {})
            cleaned.append({
                **original,
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
    if asset.get("is_synthetic") in (True, "true"):
        st.warning("This asset contains synthetic/demo attributes. Business classifications and derived priorities require organizational review.")

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
            if e_cpe not in ("", "unknown") and not is_valid_cpe(e_cpe):
                st.error(f"Primary CPE is not a valid CPE 2.3 URI: {e_cpe}")
            elif not vector_store:
                st.error("ChromaDB is not ready; cannot save changes.")
            else:
                # Metadata edits must not replace software managed in the other tab.
                if update_asset_metadata(vector_store, selected_id, updates):
                    load_all_assets_metadata.clear()
                    st.success("Asset updated. Changes are reflected immediately in retrieval and analysis.")
                    st.rerun()
                else:
                    st.error("Failed to update asset in ChromaDB.")

        with st.expander("Complete metadata (read only)"):
            st.json(asset)

    with tab_software:
        _render_software_inventory(asset, selected_id)

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
        st.info("No assets found. Run `python scripts/ingest_assets.py --v2` to populate the knowledge base.")
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
    st.session_state.va_batch = None
    st.session_state.va_report = None
    st.session_state.va_report_digest = None
    st.session_state.va_upload_generation = st.session_state.get("va_upload_generation", 0) + 1
    st.session_state.analysis_asset_select = ""
    st.session_state.pop("va_preselected_asset", None)


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
    model_output = result.get("model_output", {})
    if model_output:
        st.markdown("**Environmental Requirements**")
        for column, label, field in zip(st.columns(3), ("Confidentiality (CR)", "Integrity (IR)", "Availability (AR)"),
                                        ("confidentiality_requirement", "integrity_requirement", "availability_requirement")):
            code = model_output.get(field, "X")
            value = {"H": "High", "M": "Medium", "L": "Low", "X": "Not Defined"}.get(code, "Not Defined")
            requirement_class = {"H": "requirement-high", "M": "requirement-medium", "L": "requirement-low"}.get(code, "")
            column.html(f'<div class="result-card requirement-card {requirement_class}"><div class="card-label">{escape(label)}</div>'
                        f'<strong>{escape(value)}</strong></div>')
        st.html('<div class="summary-card"><strong>AI Assessment Summary</strong>'
                f'<p>{escape(str(model_output.get("justification", "No model summary available.")))}</p></div>')
    else:
        st.info("No validated model assessment is available for this case; CR/IR/AR have not been assigned.")
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
        if model_output and not retrieval.get("accepted"):
            st.warning("No accepted document evidence; this assessment used exact asset metadata only.")
        elif not model_output:
            st.info("No validated contextual assessment was completed. See the recorded status and failures.")
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


def _render_va_batch_results():
    batch = st.session_state.va_batch
    rows = batch_rows(batch)
    st.subheader("VA triage results")
    st.caption(f"{len(batch['report']['findings'])} imported findings · {len(rows)} finding/CVE entries. Scanner evidence and CPE applicability are separate.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button("Export results (CSV)", results_csv(rows), file_name="va_triage.csv", mime="text/csv")
    st.download_button("Export evidence (JSON)", json.dumps(batch, ensure_ascii=False, indent=2),
                       file_name="va_triage_evidence.json", mime="application/json")
    counts = pd.Series([row["Status"] for row in rows]).value_counts().to_dict()
    st.write({"Finding/CVE statuses": counts})
    if counts.get("no_cve"):
        st.info("Findings without CVEs are retained with scanner recommendations; no NVD score or organizational remediation priority is invented.")
    if counts.get("scanner_inventory_conflict"):
        st.warning("Some scanner findings conflict with CPE inventory matching. Review both sources; Not Applicable does not dismiss the scanner evidence.")
    selected = st.selectbox("Finding details", list(range(len(batch["cases"]))),
                            format_func=lambda i: f"Row {rows[i]['Row']} · {rows[i]['Finding']} · {rows[i]['CVE'] or 'No CVE'} · {rows[i]['Status']}",
                            key="va_case_select")
    case = batch["cases"][selected]
    finding = next(f for f in batch["report"]["findings"] if f["finding_id"] == case["finding_id"])
    with st.expander("Original scanner finding", expanded=not bool(case["assessment"])):
        st.write(finding["description"])
        st.markdown("**Scanner impact / recommendation**")
        st.write(finding["impact"])
        st.write(finding["recommendation"])
        st.json(finding)
    a = case["assessment"]
    if not a or a.get("base_score") is None:
        st.info(f"Assessment status: {case['status']}")
        if a:
            st.json(a)
        return
    asset = a["inputs"]["asset_metadata"]
    record = case_history_record(batch, case, asset)
    st.session_state.analysis_record = record
    st.session_state.analysis_saved = any(item.id == record.id for item in load_history())
    st.session_state.analysis_result = {
        "cve_id": case["cve_id"], "asset_metadata": asset,
        "base_score": a["base_score"], "context_score": a.get("computed_environmental_score"),
        "base_severity": a.get("technical_severity"), "environmental_severity": a.get("environmental_severity"),
        "priority": a.get("deterministic_priority"), "investigation_priority": a.get("investigation_priority"),
        "applicability": record.applicability or "Unknown", "applicability_explanation": record.applicability_explanation,
        "applicability_evidence": record.applicability_evidence, "model_output": a.get("model_output") or {},
        "priority_explanation": record.priority_explanation, "priority_status": record.priority_status,
        "priority_result": a.get("priority_result") or {}, "llm_priority": a.get("llm_priority"),
        "vector_string": record.vector_string, "modified_vector": record.modified_vector,
        "evidence": record.evidence, "nvd_description": a["inputs"]["description"], "base_source": "NVD",
    }
    if a.get("contextual_score_status") == "provisional_unknown_applicability":
        st.warning("Environmental score is provisional: software applicability is Unknown. Investigation priority is not remediation priority.")
    if a.get("status") == "failed":
        st.error("Contextual assessment failed. Only available baseline and evidence are shown.")
        st.json(a.get("failures", []))
    if a.get("warnings"):
        st.warning("Assessment limitations: " + ", ".join(a["warnings"]))
    _render_analysis_result()


def _render_vulnerability_analysis():
    st.header(":material/biotech: Vulnerability Analysis")
    st.caption("Upload an Acunetix Vulnerabilities CSV, review asset mapping, then assess the reported CVEs using NVD and organizational context.")
    assets = load_all_assets_metadata()
    asset_by_id = {a["asset_id"]: a for a in assets}
    preselect = st.session_state.get("analysis_asset_preselect", "")
    if preselect:
        st.session_state.analysis_asset_select = preselect if preselect in asset_by_id else ""
        st.session_state.va_preselected_asset = st.session_state.analysis_asset_select
        st.session_state.analysis_asset_preselect = ""
        st.session_state.va_batch = None
    st.button("New Analysis", key="new_analysis", on_click=_new_analysis)
    uploaded = st.file_uploader("Acunetix Vulnerabilities CSV", type=["csv"],
                                key=f"va_upload_{st.session_state.get('va_upload_generation', 0)}",
                                help="Use the Vulnerabilities export. Locations exports do not contain vulnerability findings.")
    if uploaded is None:
        st.info("Upload the CSV exported from Acunetix to begin. Asset context remains in the existing inventory and organizational documents.")
        return
    content = uploaded.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    if st.session_state.get("va_report_digest") != digest:
        st.session_state.analysis_asset_select = st.session_state.pop("va_preselected_asset", "")
        st.session_state.va_report_digest = digest
        st.session_state.va_batch = None
        st.session_state.va_report = None
        st.session_state.cloud_consent = False
        st.session_state.analysis_result = None
        st.session_state.analysis_record = None
        try:
            st.session_state.va_report = parse_acunetix_csv(content, uploaded.name)
        except ValueError as exc:
            st.session_state.va_import_error = str(exc)
        else:
            st.session_state.va_import_error = None
    else:
        st.session_state.pop("va_preselected_asset", None)
    report = st.session_state.get("va_report")
    if not report:
        st.error(st.session_state.get("va_import_error") or "CSV could not be read.")
        return
    st.subheader("Review imported findings")
    st.caption(f"{len(report['findings'])} findings · {sum(bool(f['cve_ids']) for f in report['findings'])} rows with explicit CVEs · source SHA-256: {report['report_id'][:16]}")
    st.dataframe(pd.DataFrame(findings_preview(report)), hide_index=True, width="stretch")
    if report["warnings"]:
        with st.expander(f"Import diagnostics ({len(report['warnings'])})"):
            for warning in report["warnings"]:
                st.warning(warning)
    st.subheader("Map report targets to organizational assets")
    st.caption("Confirm each mapping. Leave a target unmapped to retain its findings for review without contextual assessment.")
    targets = list(dict.fromkeys(f["target"] for f in report["findings"]))
    mapping = {}
    with st.form("va_analysis_form"):
        for index, target in enumerate(targets):
            key = "analysis_asset_select" if len(targets) == 1 else f"va_map_{digest[:16]}_{index}"
            options = [""] + list(asset_by_id)
            if key not in st.session_state or st.session_state[key] not in options:
                default = st.session_state.get("analysis_asset_select", "")
                st.session_state[key] = default if len(targets) == 1 and default in options else ""
            mapping[target] = st.selectbox(f"Asset for {target or '(target missing in CSV)'}", options,
                format_func=lambda aid: "— unmapped —" if not aid else _asset_label(asset_by_id[aid]), key=key)
        consent = st.checkbox("I consent to send mapped asset context and accepted document evidence to Gemini.",
                              value=False, key="cloud_consent")
        submitted = st.form_submit_button("Run Analysis", type="primary", width="stretch")
    if any(asset_by_id.get(aid, {}).get("is_synthetic") in (True, "true") for aid in mapping.values()):
        st.warning("Mapped assets include synthetic/demo attributes. Their contextual assessment is a research/demo result, not a verified organizational classification.")
    if submitted:
        eligible = any(f["import_status"] == "ready" and f["cve_ids"] and mapping.get(f["target"])
                       for f in report["findings"])
        if eligible and not consent:
            st.error("Explicit cloud consent is required before submitting organizational context to Gemini.")
        elif eligible and (not api_key or not model_name.strip()):
            st.error("Enter a Gemini API key and model in Settings.")
        else:
            # Re-read exact IDs before analysis; do not trust a stale inventory preview.
            exact_assets = {aid: get_asset_by_id(vector_store, aid) for aid in set(mapping.values()) if aid}
            exact_assets = {aid: asset for aid, asset in exact_assets.items() if asset}
            context_store = None
            if eligible:
                try:
                    context_store = load_context_db()
                except Exception:
                    st.warning("Organizational documents are unavailable; affected assessments will record retrieval failure.")
            progress = st.progress(0.0, text="Processing imported findings...")
            try:
                st.session_state.analysis_result = None
                st.session_state.analysis_record = None
                st.session_state.va_batch = run_va_batch(report, mapping, exact_assets,
                    api_key=api_key, model=model_name.strip(), context_store=context_store,
                    cloud_consent=consent, relevance_threshold=confidence_threshold, top_k=DEFAULT_TOP_K,
                    on_progress=lambda done, total: progress.progress(done / total, text=f"Processed {done}/{total} findings"))
                st.session_state.pop("va_case_select", None)
            except ValueError as exc:
                st.error(str(exc))
            finally:
                progress.empty()
    if st.session_state.get("va_batch"):
        if st.session_state.va_batch["target_assets"] != mapping:
            st.warning("Asset mapping changed. These results retain the previous mappings; run analysis again to apply the new mappings.")
        _render_va_batch_results()


def _render_analysis_history():
    st.header(":material/history: Analysis History")
    st.caption("View previous vulnerability assessments.")

    try:
        history = load_history(strict=True)
    except HistoryReadError as exc:
        st.error(f"History could not be loaded: {exc}")
        return
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
                    try:
                        delete_record(record)
                        remaining = load_history(strict=True)
                        still_present = any(item.id == record.id for item in remaining) if record.id else any(
                            asdict(item) == asdict(record) for item in remaining
                        )
                        if still_present:
                            st.error("History deletion could not be verified.")
                        else:
                            st.rerun()
                    except (HistoryReadError, OSError) as exc:
                        st.error(f"History delete failed: {exc}")

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
                st.markdown("**Retrieval relevance:**")
                st.write(record.retrieval_confidence)
                st.caption("Similarity score, not a calibrated probability.")
                if record.evidence and record.evidence.get("llm_justification"):
                    st.markdown("**LLM justification:**")
                    st.info(record.evidence["llm_justification"])
                st.markdown("**Evidence and model input**")
                st.json(record.evidence or {})
                st.markdown("**Validated model output**")
                st.json(getattr(record, "model_output", None) or {})
                st.markdown("**Provenance**")
                st.json(getattr(record, "provenance", None) or {})


if page == "Asset Inventory":
    _render_asset_inventory()
elif page == "Vulnerability Analysis":
    _render_vulnerability_analysis()
elif page == "Analysis History":
    _render_analysis_history()

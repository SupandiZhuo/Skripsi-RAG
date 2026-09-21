import streamlit as st
import requests
import re
from cvss import CVSS3
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_retrieval import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TOP_K,
    filtered_metadata,
    retrieve_asset_context,
)
from llm_assessment import LlmAssessmentError, analyze_with_llm

# ==========================================
# 1. SETUP & CACHING (Agar aplikasi cepat)
# ==========================================
st.set_page_config(page_title="Context-Aware Vuln Assessor", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_vector_db():
    """Memuat ChromaDB hanya sekali saat aplikasi dijalankan."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(collection_name="corporate_assets", persist_directory="./chroma_db", embedding_function=embeddings)

try:
    vector_store = load_vector_db()
    db_status = "Ready"
except Exception as e:
    vector_store = None
    db_status = f"Error: {e}"

# State memori untuk NVD
if "cve_data" not in st.session_state:
    st.session_state.cve_data = {"base_score": 0.0, "vector_string": "", "description": ""}

# ==========================================
# 2. FUNGSI LOGIKA INTI
# ==========================================
def fetch_cve_from_nvd(cve_id):
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data.get("vulnerabilities"): return None, "CVE tidak ditemukan."
        
        cve_data = data["vulnerabilities"][0]["cve"]
        descriptions = cve_data.get("descriptions", [])
        desc = next((d["value"] for d in descriptions if d["lang"] == "en"), "Tidak ada deskripsi.")
        
        metrics = cve_data.get("metrics", {})
        cvss_metrics = metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30") or []
        cvss_v3 = cvss_metrics[0].get("cvssData", {}) if cvss_metrics else {}
        
        return {
            "cve_id": cve_id,
            "description": desc,
            "base_score": cvss_v3.get("baseScore", 0.0),
            "vector_string": cvss_v3.get("vectorString", "")
        }, "Sukses"
    except Exception as e:
        return None, str(e)

def _norm(s):
    return str(s or "").strip().lower()

def build_modified_vector(original_vector, asset_metadata):
    """Deterministic CR/IR/AR/MAV mapping. Base metrics from NVD are never altered."""
    if not original_vector or not original_vector.strip().startswith("CVSS:3."):
        raise ValueError("Base Vector NVD tidak valid.")
    # Validate base vector parses
    CVSS3(original_vector.strip())
    asset_metadata = asset_metadata or {}
    sensitivity = _norm(asset_metadata.get("data_sensitivity"))
    criticality = _norm(asset_metadata.get("business_criticality"))
    zone = _norm(asset_metadata.get("network_zone"))

    if sensitivity in ("restricted", "confidential"):
        cr_ir = "CR:H/IR:H"
    elif sensitivity == "internal":
        cr_ir = "CR:M/IR:M"
    elif sensitivity == "public":
        cr_ir = "CR:L/IR:L"
    else:
        cr_ir = "CR:X/IR:X"

    if criticality in ("mission critical", "high"):
        ar = "AR:H"
    elif criticality == "medium":
        ar = "AR:M"
    elif criticality == "low":
        ar = "AR:L"
    else:
        ar = "AR:X"

    # Strip any pre-existing environmental overrides to keep idempotent
    parts = original_vector.strip().split("/")
    base_parts = [p for p in parts if not p.startswith(("CR:", "IR:", "AR:", "MAV:"))]
    modified = "/".join(base_parts) + f"/{cr_ir}/{ar}"

    # Deterministic MAV: AV:N exposed to internet (DMZ) stays N; isolated zones become A
    m = re.search(r"/AV:([NALP])", "/" + "/".join(base_parts))
    av = m.group(1) if m else ""
    if av == "N" and zone in ("restricted", "internal"):
        modified += "/MAV:A"
    return modified

# ==========================================
# 3. ANTARMUKA STREAMLIT
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    api_key = st.text_input("Gemini API Key", type="password", help="Dapatkan dari Google AI Studio")
    confidence_threshold = st.slider(
        "Ambang Kepercayaan Retrieval (RAG)",
        min_value=0.0, max_value=1.0, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05,
        help="Skor relevansi ternormalisasi (lebih tinggi = lebih mirip). "
             "Kandidat aset di bawah ambang ini tidak akan dipakai sebagai konteks."
    )
    st.metric(label="Status ChromaDB", value=db_status)
    if db_status == "Ready":
        st.success(f"Database aktif. Siap melakukan RAG.")

st.title("🛡️ Context-Aware Vuln Assessor")
st.markdown("Menggabungkan NVD Threat Intel & RAG Business Context.")

col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("1. Threat Intelligence (NVD)")
    col_cve, col_btn = st.columns([2, 1])
    with col_cve:
        cve_input = st.text_input("CVE ID", placeholder="Contoh: CVE-2021-44228", label_visibility="collapsed")
    with col_btn:
        if st.button("Tarik NVD API", use_container_width=True) and cve_input:
            with st.spinner("Menghubungi NVD..."):
                data, msg = fetch_cve_from_nvd(cve_input.strip())
                if data:
                    st.session_state.cve_data = data
                    st.success("Sukses!")
                else:
                    st.error(msg)
                    
    st.subheader("2. Penilaian Kontekstual")
    with st.form("vulnerability_form"):
        target_asset = st.text_input("Aset Internal Terdampak", placeholder="Contoh: Database Server Utama")
        c1, c2 = st.columns(2)
        with c1: base_score = st.number_input("Base Score", value=float(st.session_state.cve_data['base_score']))
        with c2: base_vector = st.text_input("Base Vector", value=st.session_state.cve_data['vector_string'])
        cve_description = st.text_area("Deskripsi", value=st.session_state.cve_data['description'], height=100)
        submitted = st.form_submit_button("Jalankan Analisis", type="primary")

with col_output:
    st.subheader("📊 Hasil Analisis")
    
    if submitted:
        if not api_key:
            st.error("Silakan masukkan API Key di sidebar terlebih dahulu.")
        elif not target_asset.strip():
            st.error("Silakan masukkan nama aset yang terdampak.")
        elif not base_vector.strip():
            st.error("Base Vector belum tersedia. Tarik data CVE atau masukkan vector yang valid.")
        elif not vector_store:
            st.error("ChromaDB belum siap. Jalankan skrip ingest_assets.py dulu.")
        else:
            with st.spinner("Mencari konteks aset di ChromaDB..."):
                # 1. RAG RETRIEVAL (top-3 candidates + confidence score)
                retrieval = retrieve_asset_context(
                    vector_store, target_asset,
                    top_k=DEFAULT_TOP_K, confidence_threshold=confidence_threshold
                )

            if not retrieval.passed_threshold:
                st.warning(
                    "Tidak ada aset di RAG yang cukup relevan dengan input "
                    f"'{target_asset}' (skor kandidat terbaik: "
                    f"{retrieval.best.confidence:.3f} < ambang {confidence_threshold:.2f}). "
                    "Analisis kontekstual dihentikan agar tidak menggunakan konteks aset yang salah."
                    if retrieval.best else
                    "Tidak ada kandidat aset yang ditemukan di ChromaDB untuk input ini."
                )
                with st.expander("Kandidat teratas yang ditemukan (tidak dipakai)"):
                    for c in retrieval.candidates:
                        st.write(f"`{c.metadata.get('asset_id', '?')}` "
                                 f"{c.metadata.get('asset_name', '?')} — confidence={c.confidence:.3f}")
                st.stop()

            best = retrieval.best
            asset_metadata = filtered_metadata(best.metadata)

            with st.spinner("Menganalisis dengan LLM..."):
                try:
                    # 2. DETERMINISTIC VECTOR (LLM tidak boleh mengubah vektor)
                    modified_vector_str = build_modified_vector(base_vector, asset_metadata)

                    # 3. LLM GENERATION (hanya justifikasi/remediasi)
                    llm_result = analyze_with_llm(api_key, cve_description, base_vector, modified_vector_str, asset_metadata)

                    # 4. DETERMINISTIC MATH CALCULATION
                    # Kita gunakan library python CVSS untuk menghitung skor akhir dari string deterministik
                    cvss_obj = CVSS3(modified_vector_str)
                    base_cvss, temporal_cvss, environmental_cvss = cvss_obj.scores()
                    context_score = environmental_cvss
                    
                    # UI Rendering
                    st.success("Analisis Selesai!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Base CVSS (NVD)", f"{base_score}", delta_color="off")
                    
                    # Hitung selisih untuk visualisasi
                    delta_score = round(context_score - base_score, 1)
                    m2.metric("Context-Aware Score", f"{context_score}", delta=f"{delta_score} (Env)", delta_color="inverse")
                    m3.metric("Business Impact", llm_result.nist_impact_level.value, delta_color="off")
                    
                    tab1, tab2, tab3 = st.tabs(["📝 Justifikasi Risiko", "🧮 Vektor Final", "🗄️ Konteks Aset (RAG)"])
                    with tab1:
                        st.markdown("**Analisis LLM:**")
                        st.info(llm_result.justification)
                        st.markdown("**Rekomendasi Mitigasi:**")
                        st.warning(llm_result.remediation)
                    with tab2:
                        st.code(f"Original : {base_vector}\nModified : {modified_vector_str}", language="text")
                    with tab3:
                        st.markdown(
                            f"**Aset terpilih:** `{best.metadata.get('asset_id', '?')}` "
                            f"{best.metadata.get('asset_name', '?')}  \n"
                            f"**Skor confidence:** {best.confidence:.3f} "
                            f"(ambang: {confidence_threshold:.2f})"
                        )
                        st.json(asset_metadata)
                        st.markdown("**Kandidat retrieval (top-3):**")
                        for c in retrieval.candidates:
                            marker = "✅" if c is best else "▫️"
                            st.write(f"{marker} `{c.metadata.get('asset_id', '?')}` "
                                     f"{c.metadata.get('asset_name', '?')} — confidence={c.confidence:.3f}")
                
                except LlmAssessmentError as e:
                    st.error(f"Analisis LLM ditolak/tidak valid, kalkulasi CVSS dihentikan: {str(e)}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses data: {str(e)}")
    else:
        st.info("👈 Masukkan data CVE, target aset, dan klik 'Jalankan Analisis' untuk memulai.")
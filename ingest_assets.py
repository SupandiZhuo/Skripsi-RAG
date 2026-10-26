import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys
from typing import Any, Mapping

from asset_schema_v2 import UNKNOWN, V2_COLUMNS, load_assets, validate_csv

# 1. Konfigurasi
CSV_FILE = "asset_data.csv"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "corporate_assets"

# Asset Schema v2 (Phase 2A): ingested into a separate collection so the
# original corporate_assets collection and the v1 workflow stay untouched.
CSV_FILE_V2 = os.path.join("data", "asset_data_v2.csv")
COLLECTION_NAME_V2 = "corporate_assets_v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def ingest_to_chromadb():
    print("⏳ Memulai proses data ingestion ke ChromaDB...")
    
    # 2. Baca CSV menggunakan Pandas
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ Berhasil membaca {len(df)} baris data dari {CSV_FILE}")
    except FileNotFoundError:
        print(f"❌ Error: File {CSV_FILE} tidak ditemukan. Pastikan file ada di folder yang sama.")
        return

    # 3. Konversi baris CSV menjadi LangChain Documents
    documents = []
    for index, row in df.iterrows():
        # Text yang akan di-vektorisasi untuk keperluan pencarian semantik (Similarity Search)
        page_content = f"Asset Name: {row['asset_name']}. Description: {row['description']}"
        
        # Metadata murni untuk diumpankan ke LLM sebagai konteks aturan lingkungan (Environmental Metrics)
        metadata = {
            "asset_id": row["asset_id"],
            "asset_name": row["asset_name"],
            "network_zone": row["network_zone"],
            "business_criticality": row["business_criticality"],
            "data_sensitivity": row["data_sensitivity"],
            "ip_address": row["ip_address"]
        }
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    # 4. Inisialisasi Model Embedding (Lokal & Gratis)
    print("⏳ Mengunduh/Memuat model embedding (HuggingFace all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Masukkan ke ChromaDB dan simpan di penyimpanan lokal (Persistent)
    print(f"⏳ Menyimpan vektor ke direktori {CHROMA_PERSIST_DIR}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    print("✅ Ingestion selesai! Database RAG Anda sudah siap.")
    return vector_store

# ==========================================
# INGESTION ASSET SCHEMA v2 (Phase 2A)
# ==========================================
def _clean_v2_value(value: Any) -> str:
    """Normalize a schema v2 cell into a safe, non-empty metadata string.

    Missing/blank values become the schema's explicit literal 'unknown';
    no information is invented.
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, float) and pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


def document_id(asset_id: Any) -> str:
    """Deterministic ChromaDB document ID derived from asset_id (idempotency key)."""
    return f"asset:{_clean_v2_value(asset_id)}"


def build_v2_document(row: Mapping[str, Any]) -> Document:
    """Build one structured LangChain Document from a Schema v2 asset row.

    page_content covers asset identity, business function/criticality,
    software and version, network exposure, data classification, CIA
    requirements, source provenance, and the free-text description.
    Metadata preserves every Schema v2 attribute for retrieval evidence.
    """
    value = {field: _clean_v2_value(row.get(field)) for field in V2_COLUMNS}

    page_content = "\n".join([
        f"Asset identity: {value['asset_name']} ({value['asset_id']}), "
        f"IP {value['ip_address']}, type {value['asset_type']}, "
        f"owner {value['owner']} (contact {value['owner_contact']}).",
        f"Business function: {value['business_function']}; "
        f"criticality {value['business_criticality']}; "
        f"environment {value['environment']}.",
        f"Software: {value['vendor']} {value['software_name']} "
        f"version {value['software_version']}; CPE {value['cpe']}.",
        f"Network exposure: zone {value['network_zone']}; "
        f"internet exposed {value['internet_exposed']}.",
        f"Data classification: sensitivity {value['data_sensitivity']}; "
        f"classification {value['data_classification']}.",
        f"CIA requirements: confidentiality {value['confidentiality_requirement']}, "
        f"integrity {value['integrity_requirement']}, "
        f"availability {value['availability_requirement']}.",
        f"Source provenance: source {value['source']} "
        f"(reference {value['source_reference']}); "
        f"last updated {value['last_updated']}; "
        f"synthetic {value['is_synthetic']}; schema {value['schema_version']}.",
        f"Description: {value['description']}",
    ])

    metadata = {field: value[field] for field in V2_COLUMNS}
    return Document(page_content=page_content, metadata=metadata)


def ingest_v2_to_chromadb(
    csv_file: str = CSV_FILE_V2,
    collection_name: str = COLLECTION_NAME_V2,
    persist_directory: str = CHROMA_PERSIST_DIR,
    embeddings: Any = None,
):
    """Ingest an Asset Schema v2 CSV into a NEW ChromaDB collection.

    Validates the CSV against Schema v2 first (raises ValueError on any
    schema error, writing nothing), then upserts documents with
    deterministic asset_id-based IDs, so reruns never create duplicates
    and the original corporate_assets collection is never touched.
    """
    print(f"⏳ Memulai ingestion Asset Schema v2 dari {csv_file}...")

    errors = validate_csv(csv_file)
    if errors:
        raise ValueError(
            f"Asset Schema v2 validation failed for {csv_file} "
            f"({len(errors)} error(s)):\n  - " + "\n  - ".join(errors)
        )
    print(f"✅ Skema v2 valid: {len(load_assets(csv_file))} baris lolos validasi")

    rows = load_assets(csv_file)
    documents = [build_v2_document(row) for row in rows]
    ids = [document_id(row.get("asset_id")) for row in rows]

    if embeddings is None:
        print(f"⏳ Mengunduh/Memuat model embedding (HuggingFace {EMBEDDING_MODEL_NAME})...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"⏳ Menyimpan vektor ke {persist_directory} koleksi '{collection_name}'...")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(documents=documents, ids=ids)

    print(
        f"✅ Ingestion v2 selesai: {len(documents)} aset di koleksi "
        f"'{collection_name}' (upsert, ID deterministik)."
    )
    return vector_store


# ==========================================
# FUNGSI TESTING (Untuk Vibe Coding)
# ==========================================
def test_retrieval():
    print("\n🔍 Mari kita tes mesin pencari RAG kita...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    
    # Simulasi input dari user di Streamlit
    query = "Server database untuk transaksi keuangan kartu kredit"
    print(f"Query User: '{query}'")
    
    # Mencari 1 aset paling mirip (k=1)
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        print("\n🎯 HASIL PENCARIAN TERATAS:")
        print(f"ID Aset     : {results[0].metadata['asset_id']}")
        print(f"Nama Aset   : {results[0].metadata['asset_name']}")
        print(f"Konteks RAG : {results[0].metadata}")
    else:
        print("❌ Tidak ada aset yang cocok.")

if __name__ == "__main__":
    if "--v2" in sys.argv:
        ingest_v2_to_chromadb()
    else:
        # Jalankan ingestion
        ingest_to_chromadb()

        # Jalankan tes pencarian
        test_retrieval()

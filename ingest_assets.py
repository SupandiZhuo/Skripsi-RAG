import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import argparse
import hashlib
import json
import os
from typing import Any, Mapping

from rag_retrieval import (
    CONTEXT_COLLECTION_NAME, context_chunk_id, document_provenance_error,
)

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

    inventory = row.get("software_inventory_json")
    if inventory:
        page_content += f"\nInstalled software inventory: {inventory}"
    metadata: dict[str, Any] = {field: value[field] for field in V2_COLUMNS}
    if inventory:
        metadata["software_inventory_json"] = str(inventory)
    if "inventory_verified" in row:
        flag = row["inventory_verified"]
        if type(flag) is not bool and (not isinstance(flag, str) or flag not in ("", "unknown", "true", "false")):
            raise ValueError("inventory_verified must be an explicit boolean, true, false, unknown, or empty")
        metadata["inventory_verified"] = flag
        page_content += f"\nInventory completeness verified: {flag}"
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

def build_context_documents(
    record: Mapping[str, Any], chunk_size: int = 1600, chunk_overlap: int = 200,
) -> list[Document]:
    """Build attributed source chunks without inventing content or provenance.

    JSONL fields: document_id, source_id, asset_id, source, source_reference,
    last_updated (ISO date), is_synthetic (boolean), content; optional title.
    Source/document IDs are assigned by the operator, not guessed from text.
    Hashes validate content integrity, not the truth/authenticity of the source.
    """
    if not isinstance(record, Mapping):
        raise ValueError("Context record must be an object")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (chunk_size, chunk_overlap)) or chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    content = record.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Context content must be non-empty text")
    if not isinstance(record.get("is_synthetic"), bool):
        raise ValueError("is_synthetic must be an explicit JSON boolean")
    fields = ("asset_id", "document_id", "source_id", "source", "source_reference", "last_updated", "is_synthetic")
    metadata: dict[str, Any] = {key: record.get(key) for key in fields}
    for key in ("source_type", "source_version", "collected_at", "verified_by_role", "verified_at"):
        value = record.get(key, "unknown")
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"{key} must be explicit text or 'unknown'")
        metadata[key] = value
    for key in fields[:-1]:
        if not isinstance(metadata[key], str) or not metadata[key].strip() or metadata[key] != metadata[key].strip():
            raise ValueError(f"{key} must be explicit non-empty text without surrounding whitespace")
    if "title" in record:
        if not isinstance(record["title"], str):
            raise ValueError("title must be text")
        metadata["title"] = record["title"]
    documents = []
    offset = 0
    while offset < len(content):
        chunk = content[offset:offset + chunk_size]
        if chunk.strip():
            index = len(documents)
            chunk_metadata = dict(metadata)
            chunk_metadata.update({
                "chunk_index": index,
                "chunk_id": context_chunk_id(metadata["asset_id"], metadata["source_id"], metadata["document_id"], index),
                "content_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            })
            error = document_provenance_error(chunk_metadata, chunk)
            if error:
                raise ValueError(f"Context record has {error}; complete provenance is required")
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        if offset + chunk_size >= len(content):
            break
        offset += chunk_size - chunk_overlap
    return documents


def load_context_jsonl(jsonl_file: str) -> list[dict[str, Any]]:
    """Validate the entire input before opening a database or loading embeddings."""
    records = []
    seen = set()
    with open(jsonl_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                build_context_documents(record)
                key = (record["asset_id"], record["source_id"], record["document_id"])
                if key in seen:
                    raise ValueError("duplicate asset/source/document key")
                seen.add(key)
                records.append(record)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{jsonl_file}: line {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Context JSONL contains no document records")
    return records


def ingest_context_to_chromadb(
    jsonl_file: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
    embeddings: Any = None,
):
    """Upsert genuine caller-provided documents into the dedicated collection.

    Validates every record first, fixes the scoring space to cosine, removes
    obsolete chunks only for documents present in this batch, then reads back
    exact content/provenance. Never writes either asset collection.
    """
    records = load_context_jsonl(jsonl_file)
    documents = [document for record in records for document in build_context_documents(record)]
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    store = Chroma(
        collection_name=CONTEXT_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if (store._collection.metadata or {}).get("hnsw:space") != "cosine":
        raise ValueError("Existing organizational context collection is not cosine; refusing incompatible scores")
    stale_ids = []
    for record in records:
        existing = store.get(where={"$and": [
            {"asset_id": record["asset_id"]}, {"source_id": record["source_id"]},
            {"document_id": record["document_id"]},
        ]}, include=["metadatas"])
        current_ids = {doc.metadata["chunk_id"] for doc in documents
                       if doc.metadata["asset_id"] == record["asset_id"]
                       and doc.metadata["source_id"] == record["source_id"]
                       and doc.metadata["document_id"] == record["document_id"]}
        stale_ids.extend(doc_id for doc_id in existing.get("ids", []) if doc_id not in current_ids)
    ids = [doc.metadata["chunk_id"] for doc in documents]
    store.add_documents(documents=documents, ids=ids)
    if stale_ids:
        store.delete(ids=stale_ids)
    saved = store.get(ids=ids, include=["documents", "metadatas"])
    actual = {doc_id: (text, metadata) for doc_id, text, metadata in zip(
        saved.get("ids", []), saved.get("documents", []), saved.get("metadatas", [])
    )}
    if len(actual) != len(documents) or any(
        actual.get(doc.metadata["chunk_id"]) != (doc.page_content, doc.metadata) for doc in documents
    ):
        raise RuntimeError("Context ingestion read-back verification failed")
    if stale_ids and store.get(ids=stale_ids).get("ids"):
        raise RuntimeError("Obsolete document chunks remain after ingestion")
    return store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest asset CSVs or attributed organizational context JSONL")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--v2", action="store_true", help="Ingest the Schema v2 asset CSV")
    mode.add_argument("--context-jsonl", metavar="PATH", help="Ingest organizational documents, not asset rows")
    parser.add_argument("--validate-only", action="store_true", help="Validate context JSONL without database writes or model downloads")
    parser.add_argument("--persist-directory", default=CHROMA_PERSIST_DIR)
    args = parser.parse_args(argv)
    if args.validate_only and not args.context_jsonl:
        parser.error("--validate-only requires --context-jsonl")
    if args.context_jsonl:
        if args.validate_only:
            records = load_context_jsonl(args.context_jsonl)
            print(f"Validated {len(records)} organizational documents; no database writes.")
        else:
            store = ingest_context_to_chromadb(args.context_jsonl, persist_directory=args.persist_directory)
            print(f"Organizational context ingestion verified in '{CONTEXT_COLLECTION_NAME}' ({store._collection.count()} chunks total).")
    elif args.v2:
        ingest_v2_to_chromadb(persist_directory=args.persist_directory)
    else:
        if args.persist_directory != CHROMA_PERSIST_DIR:
            parser.error("Custom --persist-directory requires --v2 or --context-jsonl")
        ingest_to_chromadb()


if __name__ == "__main__":
    main()

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1. Konfigurasi
CSV_FILE = "asset_data.csv"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "corporate_assets"

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
    # Jalankan ingestion
    ingest_to_chromadb()
    
    # Jalankan tes pencarian
    test_retrieval()
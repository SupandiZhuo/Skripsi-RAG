"""Targeted checks for the Phase 2B pipeline switch to Asset Schema v2.

Covers: the Streamlit app loads the corporate_assets_v2 collection with the
ingested v2 assets, retrieval context exposes the full v2 evidence (while
staying backward-compatible with v1 documents), the LLM prompt carries the
schema-v2 baseline rules, and collection constants stay consistent with
ingestion. No live LLM/NVD calls are made.

The app loader is inspected via AST, never imported in bare Streamlit mode.
Integration stores are ingested under tmp_path with dummy embeddings. Run:
python -m pytest test_pipeline_v2.py -q
"""

import ast
from pathlib import Path

from langchain_chroma import Chroma

from asset_schema_v2 import load_assets
from ingest_assets import (
    COLLECTION_NAME,
    COLLECTION_NAME_V2,
    build_v2_document,
    document_id,
    ingest_v2_to_chromadb,
)
from llm_assessment import TEMPLATE
from rag_retrieval import CONTEXT_METADATA_FIELDS, filtered_metadata, retrieve_asset_context
from test_ingest_assets_v2 import DummyEmbeddings

ROOT = Path(__file__).resolve().parents[1]
V2_CSV = ROOT / "data" / "asset_data_v2.csv"

V1_ONLY_METADATA = {
    "asset_id": "SRV-100",
    "asset_name": "Legacy Server",
    "ip_address": "10.0.0.1",
    "network_zone": "DMZ",
    "business_criticality": "High",
    "data_sensitivity": "Public",
}


# ---------------------------------------------------------------------------
# 1. App loads the v2 collection (not the original one)
# ---------------------------------------------------------------------------
def test_collection_constants_are_distinct():
    assert COLLECTION_NAME_V2 == "corporate_assets_v2"
    assert COLLECTION_NAME == "corporate_assets"
    assert COLLECTION_NAME_V2 != COLLECTION_NAME


def test_app_vector_store_targets_v2_collection():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_vector_db")
    returns = [node.value for node in ast.walk(loader) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    call = returns[0]
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == "Chroma"
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert isinstance(keywords["collection_name"], ast.Name)
    assert keywords["collection_name"].id == "COLLECTION_NAME_V2"
    assert ast.literal_eval(keywords["create_collection_if_not_exists"]) is False
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "ingest_assets"
        and any(alias.name == "COLLECTION_NAME_V2" and alias.asname is None for alias in node.names)
        for node in tree.body
    )


def test_temporary_v2_collection_contains_ingested_assets(tmp_path):
    store = ingest_v2_to_chromadb(
        csv_file=str(V2_CSV), collection_name=COLLECTION_NAME_V2,
        persist_directory=str(tmp_path), embeddings=DummyEmbeddings(),
    )
    ids = set(store.get()["ids"])
    assert "asset:SRV-001" in ids
    assert {document_id(row["asset_id"]) for row in load_assets(str(V2_CSV))} == ids
    assert all(doc_id.startswith("asset:") for doc_id in ids)


# ---------------------------------------------------------------------------
# 2. Retrieval context exposes full v2 evidence
# ---------------------------------------------------------------------------
def test_filtered_metadata_exposes_v2_evidence():
    rows = {row["asset_id"]: row for row in load_assets(str(V2_CSV))}
    metadata = build_v2_document(rows["SRV-001"]).metadata

    filtered = filtered_metadata(metadata)
    assert filtered["asset_type"] == "database"
    assert filtered["owner"] == "unknown"
    assert filtered["business_function"] == "unknown"
    assert filtered["software_name"] == "unknown"
    assert filtered["cpe"] == "unknown"
    assert filtered["data_classification"] == "PCI-DSS"
    assert filtered["confidentiality_requirement"] == "X"
    assert filtered["integrity_requirement"] == "X"
    assert filtered["availability_requirement"] == "X"
    assert filtered["source"] == "synthetic_sample"
    assert filtered["last_updated"] == rows["SRV-001"]["last_updated"]
    assert filtered["is_synthetic"] == "true"
    assert filtered["schema_version"] == "2.0"


def test_filtered_metadata_order_is_stable():
    rows = load_assets(str(V2_CSV))
    metadata = build_v2_document(rows[0]).metadata
    filtered = filtered_metadata(metadata)
    assert list(filtered.keys()) == [
        field for field in CONTEXT_METADATA_FIELDS if field in metadata
    ]


def test_filtered_metadata_backward_compatible_with_v1_documents():
    filtered = filtered_metadata(V1_ONLY_METADATA)
    assert filtered == V1_ONLY_METADATA
    assert "environment" not in filtered
    assert "confidentiality_requirement" not in filtered


# ---------------------------------------------------------------------------
# 3. Retrieval against a v2 store returns v2 context
# ---------------------------------------------------------------------------
def test_retrieval_returns_v2_candidates_with_full_context(tmp_path):
    rows = load_assets(str(V2_CSV))
    documents = [build_v2_document(row) for row in rows]
    ids = [document_id(row["asset_id"]) for row in rows]

    store = Chroma(
        collection_name=COLLECTION_NAME_V2,
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=DummyEmbeddings(),
        persist_directory=str(tmp_path),
    )
    store.add_documents(documents=documents, ids=ids)

    result = retrieve_asset_context(
        store, "Payment Gateway Database", top_k=3, confidence_threshold=0.0
    )
    assert len(result.candidates) == 3
    expected_asset_ids = {row["asset_id"] for row in rows}
    for candidate in result.candidates:
        assert 0.0 <= candidate.confidence <= 1.0
        assert candidate.metadata["asset_id"] in expected_asset_ids
        context = filtered_metadata(candidate.metadata)
        assert "confidentiality_requirement" in context
        assert "asset_type" in context
        assert "schema_version" in context


# ---------------------------------------------------------------------------
# 4. LLM prompt carries schema-v2 baseline rules
# ---------------------------------------------------------------------------
def test_prompt_excludes_baseline_labels_and_requires_independent_evidence():
    assert "excludes stored baseline and expert CR/IR/AR labels" in TEMPLATE
    assert "Infer CR, IR, and AR independently" in TEMPLATE
    assert "Use X for each unsupported requirement" in TEMPLATE
    assert "Every non-X metric requires evidence" in TEMPLATE
    assert "unknown" in TEMPLATE
    assert "MAV" in TEMPLATE
    assert "is_synthetic" in TEMPLATE

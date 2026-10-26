"""Targeted checks for Asset Schema v2 ChromaDB ingestion (Phase 2A).

Covers: successful ingestion of the seven v2 assets, deterministic
asset_id-based document IDs, no duplicates after repeated (upsert)
ingestion, full metadata preservation, safety of unknown/blank values,
rejection of invalid schema input before anything is written, and that
the original corporate_assets collection is never modified.

Uses a lightweight deterministic embedding stub so no model download is
needed and the production embedding model stays unchanged. Run:
python -m pytest test_ingest_assets_v2.py -q
"""

import csv
from pathlib import Path

import pytest
from langchain_chroma import Chroma

import ingest_assets
from asset_schema_v2 import UNKNOWN, V2_COLUMNS, load_assets
from ingest_assets import (
    COLLECTION_NAME,
    COLLECTION_NAME_V2,
    build_v2_document,
    document_id,
    ingest_v2_to_chromadb,
)

ROOT = Path(__file__).resolve().parents[1]
V1_CSV = ROOT / "asset_data.csv"
V2_CSV = ROOT / "data" / "asset_data_v2.csv"


class DummyEmbeddings:
    """Deterministic fixed-dimension embeddings (no network / no HF model)."""

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        vector = [0.0] * 8
        for index, char in enumerate(text):
            vector[index % 8] += (ord(char) % 97) / 97.0
        return vector


def expected_ids():
    return [document_id(row["asset_id"]) for row in load_assets(str(V2_CSV))]


def ingest(tmp_path, csv_file=V2_CSV, collection=COLLECTION_NAME_V2):
    return ingest_v2_to_chromadb(
        csv_file=str(csv_file),
        collection_name=collection,
        persist_directory=str(tmp_path),
        embeddings=DummyEmbeddings(),
    )


def open_store(persist_directory, collection=COLLECTION_NAME_V2):
    return Chroma(
        collection_name=collection,
        embedding_function=DummyEmbeddings(),
        persist_directory=str(persist_directory),
    )


def snapshot(store):
    data = store.get(include=["metadatas", "documents"])
    return {
        doc_id: (document_text, metadata)
        for doc_id, document_text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    }


# ---------------------------------------------------------------------------
# 1. Successful ingestion of the seven v2 assets
# ---------------------------------------------------------------------------
def test_ingests_seven_assets(tmp_path):
    store = ingest(tmp_path)
    ids = store.get()["ids"]
    assert len(ids) == 7
    assert set(ids) == set(expected_ids())

    reopened = open_store(tmp_path)
    assert set(reopened.get()["ids"]) == set(expected_ids())


def test_document_content_covers_required_sections(tmp_path):
    store = ingest(tmp_path)
    rows = {row["asset_id"]: row for row in load_assets(str(V2_CSV))}
    doc_id = document_id("SRV-001")
    content = store.get(ids=[doc_id])["documents"][0]

    for marker in (
        "Asset identity:",
        "Business function:",
        "Software:",
        "Network exposure:",
        "Data classification:",
        "CIA requirements:",
        "Source provenance:",
        "Description:",
    ):
        assert marker in content, f"missing section marker: {marker}"
    row = rows["SRV-001"]
    assert row["asset_name"] in content
    assert row["ip_address"] in content
    assert row["business_criticality"] in content
    assert row["data_classification"] in content
    assert row["confidentiality_requirement"] in content
    assert row["source_reference"] in content
    assert row["description"] in content


# ---------------------------------------------------------------------------
# 2. Stable, deterministic document IDs
# ---------------------------------------------------------------------------
def test_document_id_is_deterministic():
    assert document_id("SRV-001") == document_id("SRV-001")
    assert document_id("SRV-001") != document_id("SRV-002")
    assert document_id(" SRV-001 ") == document_id("SRV-001")


def test_ingested_ids_match_deterministic_ids(tmp_path):
    store = ingest(tmp_path)
    assert set(store.get()["ids"]) == set(expected_ids())

    again = ingest(tmp_path)
    assert set(again.get()["ids"]) == set(expected_ids())


# ---------------------------------------------------------------------------
# 3. Repeated ingestion creates no duplicates (upsert)
# ---------------------------------------------------------------------------
def test_rerun_creates_no_duplicates(tmp_path):
    first = ingest(tmp_path)
    first_snapshot = snapshot(first)

    second = ingest(tmp_path)
    second_snapshot = snapshot(second)

    assert len(first_snapshot) == 7
    assert len(second_snapshot) == 7
    assert set(second_snapshot) == set(expected_ids())
    assert second_snapshot == first_snapshot

    reopened = open_store(tmp_path)
    assert len(reopened.get()["ids"]) == 7


# ---------------------------------------------------------------------------
# 4. Metadata preservation
# ---------------------------------------------------------------------------
def test_metadata_preserves_all_schema_v2_fields(tmp_path):
    store = ingest(tmp_path)
    rows = {row["asset_id"]: row for row in load_assets(str(V2_CSV))}

    for asset_id, row in rows.items():
        data = store.get(ids=[document_id(asset_id)])
        assert data["metadatas"], f"no document stored for {asset_id}"
        metadata = data["metadatas"][0]
        assert set(metadata.keys()) == set(V2_COLUMNS)
        for field in V2_COLUMNS:
            expected = str(row[field]).strip() or UNKNOWN
            assert metadata[field] == expected, f"{asset_id}: {field} mismatch"


def test_metadata_contains_assessment_evidence(tmp_path):
    store = ingest(tmp_path)
    metadata = store.get(ids=[document_id("SRV-001")])["metadatas"][0]

    assert metadata["asset_id"] == "SRV-001"
    assert metadata["business_criticality"] == "Mission Critical"
    assert metadata["data_sensitivity"] == "Restricted"
    assert metadata["network_zone"] == "Restricted"
    assert metadata["environment"] == "unknown"
    assert metadata["internet_exposed"] == "no"
    assert metadata["asset_type"] == "database"
    assert metadata["data_classification"] == "PCI-DSS"
    assert metadata["confidentiality_requirement"] == "X"
    assert metadata["integrity_requirement"] == "X"
    assert metadata["availability_requirement"] == "X"
    assert metadata["source"] == "synthetic_sample"
    assert metadata["source_reference"] == "asset_data.csv v1 migration"
    assert metadata["is_synthetic"] == "true"
    assert metadata["schema_version"] == "2.0"


def test_blank_optional_values_become_unknown(tmp_path):
    row = load_assets(str(V2_CSV))[0]
    row["owner"] = ""
    row["business_function"] = "   "
    row["data_classification"] = ""

    document = build_v2_document(row)
    assert document.metadata["owner"] == UNKNOWN
    assert document.metadata["business_function"] == UNKNOWN
    assert document.metadata["data_classification"] == UNKNOWN
    assert document.metadata["asset_id"] == row["asset_id"]
    assert f"owner {UNKNOWN}" in document.page_content


# ---------------------------------------------------------------------------
# 5. Original corporate_assets collection is not modified
# ---------------------------------------------------------------------------
def test_original_collection_not_modified(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_assets, "CSV_FILE", str(V1_CSV))
    monkeypatch.setattr(ingest_assets, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(
        ingest_assets,
        "HuggingFaceEmbeddings",
        lambda model_name: DummyEmbeddings(),
    )

    original_store = ingest_assets.ingest_to_chromadb()
    before = snapshot(original_store)
    assert len(before) == 7

    ingest(tmp_path)

    after = snapshot(open_store(tmp_path, collection=COLLECTION_NAME))
    assert after == before
    assert all(not doc_id.startswith("asset:") for doc_id in after)

    v2_ids = set(open_store(tmp_path, collection=COLLECTION_NAME_V2).get()["ids"])
    assert v2_ids == set(expected_ids())
    assert set(before).isdisjoint(v2_ids)


# ---------------------------------------------------------------------------
# 6. Invalid schema handling
# ---------------------------------------------------------------------------
def write_invalid_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])


def test_invalid_schema_rejected_before_any_write(tmp_path):
    rows = load_assets(str(V2_CSV))
    bad_csv = tmp_path / "invalid.csv"
    write_invalid_csv(bad_csv, [c for c in V2_COLUMNS if c != "vendor"], rows)

    store_dir = tmp_path / "store"
    with pytest.raises(ValueError) as excinfo:
        ingest_v2_to_chromadb(
            csv_file=str(bad_csv),
            persist_directory=str(store_dir),
            embeddings=DummyEmbeddings(),
        )

    message = str(excinfo.value)
    assert "validation failed" in message
    assert "missing column(s): vendor" in message
    assert not store_dir.exists()


def test_missing_required_field_rejected(tmp_path):
    rows = load_assets(str(V2_CSV))
    rows[0] = dict(rows[0])
    rows[0]["asset_id"] = ""
    bad_csv = tmp_path / "missing_id.csv"
    write_invalid_csv(bad_csv, V2_COLUMNS, rows)

    with pytest.raises(ValueError) as excinfo:
        ingest(tmp_path, csv_file=bad_csv)
    assert "required field 'asset_id'" in str(excinfo.value)
    assert not (tmp_path / "chroma.sqlite3").exists()


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        ingest_v2_to_chromadb(
            csv_file=str(tmp_path / "does_not_exist.csv"),
            persist_directory=str(tmp_path / "store"),
            embeddings=DummyEmbeddings(),
        )
    assert "file not found" in str(excinfo.value)

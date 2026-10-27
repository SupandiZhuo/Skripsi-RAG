"""Targeted checks for the editable Asset Inventory (Phase B1).

Covers: software inventory serialization, CPE validation, CPE collection from
legacy + inventory fields, and ChromaDB metadata update/persistence.
Integration tests ingest fresh temporary collections with dummy embeddings.
Run: python -m pytest test_asset_inventory.py -q
"""

from pathlib import Path

import pytest
from langchain_chroma import Chroma

from ingest_assets import COLLECTION_NAME, COLLECTION_NAME_V2, ingest_v2_to_chromadb
from test_ingest_assets_v2 import DummyEmbeddings
from asset_inventory import (
    SOFTWARE_INVENTORY_FIELD,
    collect_asset_cpes,
    is_valid_cpe,
    load_software_inventory,
    save_software_inventory,
)
from rag_retrieval import (
    get_asset_by_id,
    list_all_assets,
    update_asset_metadata,
)


# ---------------------------------------------------------------------------
# Software inventory serialization
# ---------------------------------------------------------------------------
def test_load_software_inventory_empty_when_missing():
    assert load_software_inventory({}) == []
    assert load_software_inventory({SOFTWARE_INVENTORY_FIELD: ""}) == []
    assert load_software_inventory({SOFTWARE_INVENTORY_FIELD: "unknown"}) == []


def test_save_and_load_software_inventory_round_trip():
    software = [
        {"name": "Apache Log4j", "version": "2.14.1", "cpe": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"},
        {"name": "WordPress", "version": "6.0", "cpe": ""},
    ]
    serialized = save_software_inventory(software)
    loaded = load_software_inventory({SOFTWARE_INVENTORY_FIELD: serialized})
    assert loaded == software


def test_load_software_inventory_skips_malformed_json():
    assert load_software_inventory({SOFTWARE_INVENTORY_FIELD: "not-json"}) == []


# ---------------------------------------------------------------------------
# CPE validation and collection
# ---------------------------------------------------------------------------
def test_is_valid_cpe_accepts_cpe_2_3():
    assert is_valid_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
    assert not is_valid_cpe("not-a-cpe")
    assert not is_valid_cpe("")


def test_collect_asset_cpes_includes_legacy_and_inventory():
    asset = {
        "cpe": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
        SOFTWARE_INVENTORY_FIELD: save_software_inventory([
            {"name": "OpenSSL", "version": "1.1.1", "cpe": "cpe:2.3:a:openssl:openssl:1.1.1:*:*:*:*:*:*:*"},
        ]),
    }
    cpes = collect_asset_cpes(asset)
    assert "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*" in cpes
    assert "cpe:2.3:a:openssl:openssl:1.1.1:*:*:*:*:*:*:*" in cpes
    assert len(cpes) == 2


def test_collect_asset_cpes_deduplicates():
    asset = {
        "cpe": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
        SOFTWARE_INVENTORY_FIELD: save_software_inventory([
            {"name": "Log4j", "version": "2.14.1", "cpe": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"},
        ]),
    }
    assert len(collect_asset_cpes(asset)) == 1


def test_collect_asset_cpes_skips_invalid_and_unknown():
    asset = {
        "cpe": "unknown",
        SOFTWARE_INVENTORY_FIELD: save_software_inventory([
            {"name": "Bad", "version": "1", "cpe": "not-a-cpe"},
            {"name": "Good", "version": "2", "cpe": "cpe:2.3:a:vendor:product:2:*:*:*:*:*:*:*"},
        ]),
    }
    cpes = collect_asset_cpes(asset)
    assert cpes == ["cpe:2.3:a:vendor:product:2:*:*:*:*:*:*:*"]


# ---------------------------------------------------------------------------
# ChromaDB integration
# ---------------------------------------------------------------------------
@pytest.fixture
def vector_store(tmp_path):
    return ingest_v2_to_chromadb(
        csv_file=str(Path(__file__).resolve().parents[1] / "data" / "asset_data_v2.csv"),
        collection_name=COLLECTION_NAME_V2,
        persist_directory=str(tmp_path),
        embeddings=DummyEmbeddings(),
    )


def reopen_store(tmp_path, collection=COLLECTION_NAME_V2):
    return Chroma(
        collection_name=collection, persist_directory=str(tmp_path),
        embedding_function=DummyEmbeddings(), create_collection_if_not_exists=False,
    )


def test_get_asset_by_id_returns_metadata(vector_store):
    asset = get_asset_by_id(vector_store, "SRV-001")
    assert asset is not None
    assert asset["asset_id"] == "SRV-001"


def test_get_asset_by_id_returns_none_for_missing(vector_store):
    assert get_asset_by_id(vector_store, "SRV-99999") is None


def test_update_asset_metadata_persists_and_reflected_after_reopen(vector_store, tmp_path):
    original = get_asset_by_id(vector_store, "SRV-003")
    assert original is not None
    original_desc = original.get("description", "")

    new_desc = original_desc + " [test-mark]"
    updates = {"description": new_desc}
    assert update_asset_metadata(vector_store, "SRV-003", updates)

    # Direct fetch should see the change.
    fetched = get_asset_by_id(vector_store, "SRV-003")
    assert fetched["description"] == new_desc

    # A separately opened wrapper must see the persisted update too.
    all_assets = {a["asset_id"]: a for a in list_all_assets(reopen_store(tmp_path))}
    assert all_assets["SRV-003"]["description"] == new_desc


def test_software_inventory_persists_in_chroma(vector_store, tmp_path):
    software = [
        {"name": "TestApp", "version": "1.0.0", "cpe": "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"},
    ]
    updates = {SOFTWARE_INVENTORY_FIELD: save_software_inventory(software)}
    assert update_asset_metadata(vector_store, "SRV-003", updates)

    fetched = get_asset_by_id(reopen_store(tmp_path), "SRV-003")
    loaded = load_software_inventory(fetched)
    assert len(loaded) == 1
    assert loaded[0]["cpe"] == "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"


def test_collect_asset_cpes_from_temporary_asset_with_inventory(vector_store):
    software = [
        {"name": "TestApp", "version": "1.0.0", "cpe": "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"},
    ]
    assert update_asset_metadata(vector_store, "SRV-003", {SOFTWARE_INVENTORY_FIELD: save_software_inventory(software)})

    fetched = get_asset_by_id(vector_store, "SRV-003")
    cpes = collect_asset_cpes(fetched)
    assert "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*" in cpes


# ---------------------------------------------------------------------------
# Legacy collection preservation
# ---------------------------------------------------------------------------
def test_legacy_corporate_assets_collection_untouched(tmp_path):
    legacy = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(tmp_path),
        embedding_function=DummyEmbeddings(),
    )
    legacy.add_texts(["Legacy fixture asset"], metadatas=[{"asset_id": "LEGACY-001"}], ids=["legacy:001"])
    before = legacy.get(include=["metadatas", "documents", "embeddings"])
    current = ingest_v2_to_chromadb(
        csv_file=str(Path(__file__).resolve().parents[1] / "data" / "asset_data_v2.csv"),
        persist_directory=str(tmp_path), collection_name=COLLECTION_NAME_V2,
        embeddings=DummyEmbeddings(),
    )
    assert update_asset_metadata(current, "SRV-003", {"description": "Temporary test update"})
    after = reopen_store(tmp_path, COLLECTION_NAME).get(include=["metadatas", "documents", "embeddings"])
    assert after["ids"] == before["ids"] == ["legacy:001"]
    assert after["documents"] == before["documents"]
    assert after["metadatas"] == before["metadatas"]
    assert (after["embeddings"] == before["embeddings"]).all()

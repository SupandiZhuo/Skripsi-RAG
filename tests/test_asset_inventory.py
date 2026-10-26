"""Targeted checks for the editable Asset Inventory (Phase B1).

Covers: software inventory serialization, CPE validation, CPE collection from
legacy + inventory fields, and ChromaDB metadata update/persistence.
Integration tests use the real `corporate_assets_v2` collection, which must
have been ingested via `python ingest_assets.py --v2`.
Run: python -m pytest test_asset_inventory.py -q
"""

import pytest

from app import load_all_assets_metadata, load_vector_db
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
@pytest.fixture(scope="module")
def vector_store():
    return load_vector_db()


def test_get_asset_by_id_returns_metadata(vector_store):
    asset = get_asset_by_id(vector_store, "SRV-001")
    assert asset is not None
    assert asset["asset_id"] == "SRV-001"


def test_get_asset_by_id_returns_none_for_missing(vector_store):
    assert get_asset_by_id(vector_store, "SRV-99999") is None


def test_update_asset_metadata_persists_and_reflected_after_cache_clear(vector_store):
    # Use SRV-003 (low-criticality internal app) to avoid altering critical sample data.
    original = get_asset_by_id(vector_store, "SRV-003")
    assert original is not None
    original_desc = original.get("description", "")

    new_desc = original_desc + " [test-mark]"
    updates = {"description": new_desc}
    assert update_asset_metadata(vector_store, "SRV-003", updates)

    # Direct fetch should see the change.
    fetched = get_asset_by_id(vector_store, "SRV-003")
    assert fetched["description"] == new_desc

    # list_all_assets / cached loader should also see it after clearing cache.
    load_all_assets_metadata.clear()
    all_assets = {a["asset_id"]: a for a in list_all_assets(vector_store)}
    assert all_assets["SRV-003"]["description"] == new_desc

    # Restore original value.
    update_asset_metadata(vector_store, "SRV-003", {"description": original_desc})
    load_all_assets_metadata.clear()


def test_software_inventory_persists_in_chroma(vector_store):
    original = get_asset_by_id(vector_store, "SRV-003")
    original_sw = original.get(SOFTWARE_INVENTORY_FIELD, "")

    software = [
        {"name": "TestApp", "version": "1.0.0", "cpe": "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"},
    ]
    updates = {SOFTWARE_INVENTORY_FIELD: save_software_inventory(software)}
    assert update_asset_metadata(vector_store, "SRV-003", updates)

    fetched = get_asset_by_id(vector_store, "SRV-003")
    loaded = load_software_inventory(fetched)
    assert len(loaded) == 1
    assert loaded[0]["cpe"] == "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"

    # Restore.
    update_asset_metadata(vector_store, "SRV-003", {SOFTWARE_INVENTORY_FIELD: original_sw})
    load_all_assets_metadata.clear()


def test_collect_asset_cpes_from_live_asset_with_inventory(vector_store):
    # Set inventory on SRV-003, collect CPEs, then restore.
    original = get_asset_by_id(vector_store, "SRV-003")
    original_sw = original.get(SOFTWARE_INVENTORY_FIELD, "")

    software = [
        {"name": "TestApp", "version": "1.0.0", "cpe": "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*"},
    ]
    update_asset_metadata(vector_store, "SRV-003", {SOFTWARE_INVENTORY_FIELD: save_software_inventory(software)})

    fetched = get_asset_by_id(vector_store, "SRV-003")
    cpes = collect_asset_cpes(fetched)
    assert "cpe:2.3:a:testvendor:testapp:1.0.0:*:*:*:*:*:*:*" in cpes

    # Restore.
    update_asset_metadata(vector_store, "SRV-003", {SOFTWARE_INVENTORY_FIELD: original_sw})
    load_all_assets_metadata.clear()


# ---------------------------------------------------------------------------
# Legacy collection preservation
# ---------------------------------------------------------------------------
def test_legacy_corporate_assets_collection_untouched(vector_store):
    from ingest_assets import COLLECTION_NAME
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from ingest_assets import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

    legacy = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
    )
    ids = set(legacy.get()["ids"])
    # Legacy collection should still contain its original assets and not v2 IDs.
    assert "asset:SRV-001" not in ids
    assert len(ids) > 0

"""Package integration uses temporary synthetic fixtures, never source edits."""
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skripsi_rag.asset_inventory import collect_asset_cpes, inventory_is_verified, load_software_inventory
from skripsi_rag.asset_package import load_orion_package
from skripsi_rag.asset_schema_v2 import V2_COLUMNS, V2_EXTENSION_COLUMNS, load_assets
from skripsi_rag.ingest_assets import build_context_documents, ingest_orion_package, main
from skripsi_rag.llm_assessment import build_model_context
from skripsi_rag.rag_retrieval import retrieve_document_context
from test_ingest_assets_v2 import DummyEmbeddings


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    source = load_assets(str(Path(__file__).resolve().parents[1] / "data/examples/asset_data_v2.csv"))[:2]
    assets, software, documents = [], [], []
    for i, row in enumerate(source, start=1):
        asset = dict(row, asset_id=f"ORI-TEST-{i}", cpe="unknown", is_synthetic="true",
                     software_inventory_json="", inventory_verified="false")
        assets.append({k: asset[k] for k in V2_COLUMNS + list(V2_EXTENSION_COLUMNS)})
        software.append({"asset_id": asset["asset_id"], "software_name": "PostgreSQL",
            "software_version": "16", "vendor": "PostgreSQL", "cpe": "unknown",
            "inventory_verified": "false", "source_reference": "test questionnaire"})
        documents.append({"document_id": f"DOC-{i}", "asset_id": asset["asset_id"],
            "source_id": "TEST-SOURCE", "source": "Test-only questionnaire summary",
            "source_reference": "test questionnaire", "last_updated": "2026-10-30",
            "is_synthetic": True, "content": f"Operational context for asset {i}: daily service availability supports business operations."})
    write_csv(root / "01_asset_inventory/02_asset_data_v2_ORION_DEMO.csv", assets)
    write_csv(root / "01_asset_inventory/03_software_inventory_ORION.csv", software)
    context = root / "02_dokumen_rag/04_organizational_context_BASELINE.jsonl"
    context.parent.mkdir(parents=True)
    context.write_text("\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8")
    catalog = [{"document_id": d["document_id"], "primary_asset_id": d["asset_id"],
        "source_id": d["source_id"], "is_synthetic": "true",
        "content_sha256": hashlib.sha256(d["content"].encode()).hexdigest()} for d in documents]
    trace = [{"document_id": d["document_id"], "primary_asset_id": d["asset_id"],
        "source_id": d["source_id"], "is_synthetic": "true", "source_reference": d["source_reference"]} for d in documents]
    write_csv(root / "02_dokumen_rag/document_catalog.csv", catalog)
    write_csv(root / "02_dokumen_rag/document_source_traceability.csv", trace)
    mapping = [{"asset_id": d["asset_id"], "document_id": d["document_id"], "mapping_status": "baseline"} for d in documents]
    mapping.append({"asset_id": assets[0]["asset_id"], "document_id": documents[1]["document_id"], "mapping_status": "proposed_review"})
    write_csv(root / "03_pemetaan_aset_dokumen/asset_document_mapping.csv", mapping)
    evaluation = root / "04_evaluasi_eksperimen"
    evaluation.mkdir()
    (evaluation / "05_organizational_context_SYNTHETIC.jsonl").write_text('{"expert_label":"DO NOT EMBED"}', encoding="utf-8")
    return root


def test_package_preserves_unknowns_and_excludes_proposed_links_and_labels(package):
    data = load_orion_package(package)
    assert data["summary"]["assets"] == 2
    assert data["summary"]["software_records"] == 2
    assert data["summary"]["documents"] == 2
    assert data["summary"]["excluded_proposed_links"] == 1
    assert len(data["summary"]["source_sha256"]) == 6
    assert "DO NOT EMBED" not in json.dumps(data)
    for asset in data["assets"]:
        assert asset["is_synthetic"] == "true"
        assert not inventory_is_verified(asset)
        assert collect_asset_cpes(asset) == []
        software = load_software_inventory(asset)
        assert software[0]["version"] == "16"  # Never invent a patch version.
        assert software[0]["cpe"] == "unknown"
        model_context = build_model_context(asset)
        assert model_context["asset"]["software_inventory"][0]["software_name"] == "PostgreSQL"
        assert "confidentiality_requirement" not in model_context["asset"]


@pytest.mark.parametrize("problem", ["hash", "source", "cross_asset", "unknown_software_asset", "verified_software"])
def test_invalid_package_fails_before_model_or_database_access(package, problem):
    if problem in ("hash", "source"):
        path = package / "02_dokumen_rag/04_organizational_context_BASELINE.jsonl"
        docs = [json.loads(line) for line in path.read_text().splitlines()]
        docs[0]["content" if problem == "hash" else "source_reference"] = "changed"
        path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    elif problem == "cross_asset":
        path = package / "03_pemetaan_aset_dokumen/asset_document_mapping.csv"
        rows = list(csv.DictReader(path.open()))
        rows[-1]["mapping_status"] = "baseline"
        write_csv(path, rows)
    else:
        path = package / "01_asset_inventory/03_software_inventory_ORION.csv"
        rows = list(csv.DictReader(path.open()))
        rows[0]["asset_id" if problem == "unknown_software_asset" else "inventory_verified"] = "OTHER" if problem == "unknown_software_asset" else "true"
        write_csv(path, rows)
    with patch("skripsi_rag.ingest_assets.HuggingFaceEmbeddings") as embeddings, patch("skripsi_rag.ingest_assets.Chroma") as chroma:
        with pytest.raises(ValueError):
            ingest_orion_package(package)
    embeddings.assert_not_called()
    chroma.assert_not_called()


def test_package_ingestion_is_idempotent_scoped_and_preserves_unrelated_data(package, tmp_path):
    directory = str(tmp_path / "chroma")
    assets, contexts, summary = ingest_orion_package(package, directory, DummyEmbeddings())
    other = build_context_documents({"asset_id": "OTHER", "document_id": "OTHER-DOC", "source_id": "TEST",
        "source": "Unit test", "source_reference": "unit-test", "last_updated": "2026-01-01",
        "is_synthetic": True, "content": "Unrelated context must remain unchanged."})
    contexts.add_documents(other, ids=[d.metadata["chunk_id"] for d in other])
    assets._collection.add(ids=["asset:OTHER"], documents=["Unrelated asset"], metadatas=[{"asset_id": "OTHER"}], embeddings=[[0.1] * 8])
    before = contexts.get(ids=[other[0].metadata["chunk_id"]])
    again_assets, again_contexts, _ = ingest_orion_package(package, directory, DummyEmbeddings())
    assert len(again_assets.get()["ids"]) == 3
    assert len(again_contexts.get()["ids"]) == 3
    assert again_contexts.get(ids=[other[0].metadata["chunk_id"]]) == before
    result = retrieve_document_context(again_contexts, "daily business service availability", "ORI-TEST-1", relevance_threshold=0)
    assert result.accepted and not result.error
    assert {d.metadata["document_id"] for d in result.accepted} == {"DOC-1"}
    assert summary["baseline_links"] == 2


def test_validate_only_does_not_download_or_write(package, capsys):
    with patch("skripsi_rag.ingest_assets.HuggingFaceEmbeddings") as embeddings, patch("skripsi_rag.ingest_assets.Chroma") as chroma:
        main(["--orion-package", str(package), "--validate-only"])
    embeddings.assert_not_called()
    chroma.assert_not_called()
    assert "no database writes" in capsys.readouterr().out


def test_custom_asset_csv_is_forwarded_without_replacing_source_file(package):
    path = str(package / "01_asset_inventory/02_asset_data_v2_ORION_DEMO.csv")
    with patch("skripsi_rag.ingest_assets.ingest_v2_to_chromadb") as ingest:
        main(["--v2", "--asset-csv", path])
    assert ingest.call_args.kwargs["csv_file"] == path

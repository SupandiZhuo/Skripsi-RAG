"""Validate the supplied Orion asset/document package without database writes.

Only the Schema v2 demo inventory and baseline documents are imported. Review
tables remain source material; evaluation labels, synthetic scenarios and
proposed cross-asset links never enter the baseline knowledge base.
"""
import csv
import hashlib
from pathlib import Path

from skripsi_rag.asset_inventory import is_valid_cpe, save_software_inventory
from skripsi_rag.asset_schema_v2 import load_assets, validate_csv


def _csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if (not required <= set(reader.fieldnames or [])
                or len(reader.fieldnames) != len(set(reader.fieldnames))):
            raise ValueError(f"{path.name}: missing required columns")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: empty file or invalid row width")
    return rows


def load_orion_package(directory):
    from skripsi_rag.ingest_assets import load_context_jsonl

    root = Path(directory)
    paths = {
        "assets": root / "01_asset_inventory/02_asset_data_v2_ORION_DEMO.csv",
        "software": root / "01_asset_inventory/03_software_inventory_ORION.csv",
        "context": root / "02_dokumen_rag/04_organizational_context_BASELINE.jsonl",
        "catalog": root / "02_dokumen_rag/document_catalog.csv",
        "traceability": root / "02_dokumen_rag/document_source_traceability.csv",
        "mapping": root / "03_pemetaan_aset_dokumen/asset_document_mapping.csv",
    }
    errors = validate_csv(str(paths["assets"]))
    if errors:
        raise ValueError("Orion asset schema validation failed: " + "; ".join(errors))
    assets = load_assets(str(paths["assets"]))
    asset_by_id = {a["asset_id"]: a for a in assets}
    if not asset_by_id:
        raise ValueError("Package contains no assets")
    software = _csv(paths["software"], {"asset_id", "software_name", "software_version", "cpe", "inventory_verified", "source_reference"})
    grouped = {aid: [] for aid in asset_by_id}
    for item in software:
        if item["asset_id"] not in grouped or not item["software_name"].strip():
            raise ValueError("Software inventory has unknown asset_id or empty software name")
        if item["inventory_verified"] != "false":
            raise ValueError("Orion software is unverified; verification requires a separate inventory review")
        cpe = item["cpe"]
        if cpe not in ("", "unknown") and not is_valid_cpe(cpe):
            raise ValueError("Software inventory contains an invalid CPE")
        grouped[item["asset_id"]].append({
            **{k: v for k, v in item.items() if k != "asset_id"},
            "name": item["software_name"], "version": item["software_version"],
        })
    for asset in assets:
        asset["software_inventory_json"] = save_software_inventory(grouped[asset["asset_id"]])
        asset["inventory_verified"] = "false"

    documents = load_context_jsonl(str(paths["context"]))
    docs = {d["document_id"]: d for d in documents}
    if len(docs) != len(documents) or any(d["asset_id"] not in asset_by_id for d in documents):
        raise ValueError("Baseline documents have duplicate IDs or unknown assets")
    catalog = _csv(paths["catalog"], {"document_id", "primary_asset_id", "source_id", "is_synthetic", "content_sha256"})
    traceability = _csv(paths["traceability"], {"document_id", "primary_asset_id", "source_id", "source_reference", "is_synthetic"})
    for name, records in (("catalog", catalog), ("traceability", traceability)):
        if len(records) != len(docs) or {r["document_id"] for r in records} != set(docs):
            raise ValueError(f"Document {name} does not match the baseline documents")
        for record in records:
            doc = docs[record["document_id"]]
            if (record["primary_asset_id"] != doc["asset_id"] or record["source_id"] != doc["source_id"]
                    or record["is_synthetic"] != str(doc["is_synthetic"]).lower()):
                raise ValueError(f"Document {name} provenance mismatch for {doc['document_id']}")
            if name == "catalog" and record["content_sha256"] != hashlib.sha256(doc["content"].encode()).hexdigest():
                raise ValueError(f"Catalog content hash mismatch for {doc['document_id']}")
            if name == "traceability" and record["source_reference"] != doc["source_reference"]:
                raise ValueError(f"Document source reference mismatch for {doc['document_id']}")

    mapping = _csv(paths["mapping"], {"asset_id", "document_id", "mapping_status"})
    seen, baseline = set(), set()
    for link in mapping:
        key = (link["asset_id"], link["document_id"])
        if key in seen or key[0] not in asset_by_id or key[1] not in docs:
            raise ValueError("Asset/document mapping has duplicate or unknown IDs")
        seen.add(key)
        if link["mapping_status"] == "baseline":
            if docs[key[1]]["asset_id"] != key[0]:
                raise ValueError("Cross-asset baseline mapping requires separate approval and retrieval support")
            baseline.add(key)
        elif link["mapping_status"] != "proposed_review":
            raise ValueError("Unknown asset/document mapping status")
    if baseline != {(d["asset_id"], d["document_id"]) for d in documents}:
        raise ValueError("Each baseline document must have its primary asset mapping")
    return {
        "assets": assets, "documents": documents, "context_path": str(paths["context"]),
        "summary": {"assets": len(assets), "software_records": len(software), "documents": len(documents),
                    "baseline_links": len(baseline), "excluded_proposed_links": len(mapping) - len(baseline),
                    "synthetic_assets": sum(a["is_synthetic"] == "true" for a in assets),
                    "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}},
    }

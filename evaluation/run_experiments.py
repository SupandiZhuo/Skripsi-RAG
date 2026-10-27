"""Offline NVD snapshot E0-E3 runner; never fetches CVEs or invents labels.

Usage (no model call):
  python -m evaluation.run_experiments --scenarios scenarios.jsonl \
      --assets assets.jsonl --snapshots-dir snapshots --output-dir runs/run-001

E1-E3 require --allow-cloud and GOOGLE_API_KEY/GEMINI_API_KEY. E3 also requires
an existing document collection and a locally cached embedding model; the
runner refuses implicit corpus creation or embedding downloads. Labels, if
provided for summary, are read only after all predictions have been written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import uuid
from pathlib import Path

from evaluation.config import ExperimentConfig, freeze_manifest
from evaluation.metrics import summarize_runs, validate_splits

_SCENARIO_FIELDS = {"scenario_id", "cve_id", "asset_id", "retrieval_query", "split"}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {number} must be an object.")
                rows.append(row)
    return rows


def load_assets(path):
    target = Path(path)
    if target.suffix.lower() == ".csv":
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = read_jsonl(target)
    result = {}
    for row in rows:
        asset_id = row.get("asset_id")
        if not isinstance(asset_id, str) or not asset_id or asset_id in result:
            raise ValueError("Asset IDs must be nonempty and unique.")
        result[asset_id] = row
    return result


def _validate_scenarios(scenarios, assets):
    seen = set()
    for row in scenarios:
        unknown = set(row) - _SCENARIO_FIELDS
        if unknown:
            raise ValueError(f"Unexpected scenario fields (labels forbidden): {sorted(unknown)}")
        for key in ("scenario_id", "cve_id", "asset_id"):
            value = row.get(key)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"Scenario {key} must be a literal nonempty string.")
        if not re.fullmatch(r"CVE-\d{4}-\d{4,}", row["cve_id"]):
            raise ValueError("Invalid literal CVE ID in scenario.")
        if row["scenario_id"] in seen:
            raise ValueError("Scenario IDs must be unique; use config.repeats for repeats.")
        if row["asset_id"] not in assets:
            raise ValueError(f"Unknown selected asset_id: {row['asset_id']}")
        if assets[row["asset_id"]].get("asset_id") != row["asset_id"]:
            raise ValueError("Selected asset key must match the exact asset_metadata.asset_id.")
        if "retrieval_query" in row and not isinstance(row["retrieval_query"], str):
            raise ValueError("retrieval_query must be text.")
        if row.get("split", "unspecified") not in {"train", "validation", "test", "unspecified"}:
            raise ValueError("split must be train, validation, test, or unspecified.")
        seen.add(row["scenario_id"])
    validate_splits(scenarios)


def _content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def open_document_store(persist_directory, config):
    """Open existing cosine context collection only; no model downloads/ingest."""
    from chromadb import PersistentClient
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from rag_retrieval import CONTEXT_COLLECTION_NAME
    if not persist_directory or not Path(persist_directory).is_dir():
        raise ValueError("E3 requires an existing document-store directory.")
    client = PersistentClient(path=str(persist_directory))
    collection = client.get_collection(CONTEXT_COLLECTION_NAME)
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        raise ValueError("Document collection must use cosine distance.")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model,
                                       model_kwargs={"local_files_only": True})
    return Chroma(client=client, collection_name=CONTEXT_COLLECTION_NAME,
                  embedding_function=embeddings, create_collection_if_not_exists=False)


def run_batch(scenarios, assets, snapshots_dir, output_dir, config: ExperimentConfig,
              *, allow_cloud=False, api_key="", context_store=None,
              input_paths=(), analyzer=None):
    """Return records after writing frozen manifest, JSONL, CSV, and summary.

    No NVD network fallback. Snapshot/input errors become explicit per-scenario
    failures. A new output directory is required; previous runs are untouched.
    Empty batches are rejected rather than presented as completed research.
    """
    _validate_scenarios(scenarios, assets)
    needs_model = any(experiment != "E0" for experiment in config.experiments)
    if needs_model and allow_cloud is not True:
        raise PermissionError("E1-E3 require explicit --allow-cloud consent for cloud processing.")
    if needs_model and analyzer is None and not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY required; never supply it in config.")
    if not scenarios:
        raise ValueError("No scenarios supplied. No research data will be synthesized.")
    if "E3" in config.experiments and context_store is None:
        raise ValueError("E3 requires an existing organizational document store.")
    from nvd_client import load_nvd_snapshot
    from pipeline import run_assessment
    random.seed(config.seed)
    batch_id = str(uuid.uuid4())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = [Path(snapshots_dir) / (cve + ".json") for cve in sorted({s["cve_id"] for s in scenarios})]
    extra = {"batch_id": batch_id, "cloud_consent": bool(allow_cloud),
             "scenario_sha256": _content_hash(scenarios), "asset_input_sha256": _content_hash(assets),
             "nvd_policy": "local-snapshots-only-no-network-fallback",
             "missing_snapshots": [str(path) for path in snapshots if not path.exists()],
             "purpose": "infrastructure; empirical validity requires independent expert labels"}
    if context_store is not None:
        corpus = context_store.get(include=["documents", "metadatas"])
        entries = sorted(zip(corpus.get("ids", []), corpus.get("documents", []), corpus.get("metadatas", [])), key=lambda item: item[0])
        extra["document_corpus_sha256"] = _content_hash(entries)
        extra["document_corpus_count"] = len(entries)
        collection = getattr(context_store, "_collection", None)
        extra["document_collection_metadata"] = dict(getattr(collection, "metadata", None) or {})
    manifest = freeze_manifest(config, output / "manifest.json",
                               input_paths=[*input_paths, *[p for p in snapshots if p.exists()]], extra=extra)
    records = []
    with (output / "experiments.jsonl").open("x", encoding="utf-8") as handle:
        for scenario in scenarios:
            snapshot_error = None
            try:
                nvd = load_nvd_snapshot(Path(snapshots_dir) / (scenario["cve_id"] + ".json"), scenario["cve_id"])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                nvd = {"cve_id": scenario["cve_id"]}
                snapshot_error = type(exc).__name__
            for repeat in range(config.repeats):
                for experiment in config.experiments:
                    record = run_assessment(nvd, assets[scenario["asset_id"]], api_key=api_key,
                        experiment_id=experiment, context_store=context_store, model=config.model,
                        top_k=config.top_k, relevance_threshold=config.relevance_threshold,
                        cloud_consent=allow_cloud, scenario_id=scenario["scenario_id"], analyzer=analyzer,
                        retrieval_query=scenario.get("retrieval_query"), embedding_model=config.embedding_model,
                        config_hash=manifest["config_sha256"])
                    record.update(batch_id=batch_id, repeat=repeat, split=scenario.get("split", "unspecified"))
                    if snapshot_error:
                        record["failures"].insert(0, {"error_type": "nvd_snapshot_failure", "stage": "nvd",
                                                      "exception_class": snapshot_error})
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    records.append(record)
    # CSV preserves all fields; nested evidence/provenance remain JSON cells.
    fields = sorted({key for row in records for key in row})
    with (output / "experiments.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in record.items()})
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summarize_runs(records), handle, ensure_ascii=False, indent=2, allow_nan=False)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--snapshots-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-cloud", action="store_true", help="Explicit consent to send minimized context to cloud LLM")
    parser.add_argument("--document-store", help="E3 existing Chroma persistence directory")
    parser.add_argument("--labels", help="Optional external adjudicated JSONL, read AFTER predictions only")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_json(args.config)
    if any(e != "E0" for e in config.experiments) and not args.allow_cloud:
        parser.error("E1-E3 require --allow-cloud; no model or data access attempted.")
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if any(e != "E0" for e in config.experiments) and not key:
        parser.error("GOOGLE_API_KEY or GEMINI_API_KEY is required.")
    store = open_document_store(args.document_store, config) if "E3" in config.experiments else None
    records = run_batch(read_jsonl(args.scenarios), load_assets(args.assets), args.snapshots_dir,
                        args.output_dir, config, allow_cloud=args.allow_cloud, api_key=key, context_store=store,
                        input_paths=[args.config, args.scenarios, args.assets])
    if args.labels:
        labels = {}
        for row in read_jsonl(args.labels):
            if row["scenario_id"] in labels:
                raise ValueError("Duplicate expert label scenario; adjudicate labels separately first.")
            labels[row["scenario_id"]] = row
        with (Path(args.output_dir) / "summary_with_reference.json").open("x", encoding="utf-8") as handle:
            json.dump({"reference_sha256": hashlib.sha256(Path(args.labels).read_bytes()).hexdigest(),
                       "metrics": summarize_runs(records, labels)}, handle, indent=2, allow_nan=False)
    print(json.dumps({"output_dir": args.output_dir, "n_records": len(records),
                      "scope": "Infrastructure execution; not a research-validity claim."}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

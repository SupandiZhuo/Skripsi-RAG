"""Evaluation unit fixtures only; no expert labels or empirical claims."""
import json

import pytest

from evaluation.config import ExperimentConfig, freeze_manifest
from evaluation.metrics import classification_metrics, retrieval_metrics, summarize_runs, validate_splits
from evaluation.run_experiments import run_batch


def test_unknown_predictions_are_not_removed_from_accuracy_or_f1():
    metrics = classification_metrics(["Confirmed", "Not Applicable", "Unknown"],
                                     ["Confirmed", "Unknown", "Unknown"],
                                     ["Confirmed", "Not Applicable"])
    assert metrics["n_labeled"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["coverage"] == 0.5
    assert metrics["excluded_unknown_reference"] == 1
    assert metrics["confusion_matrix"]["Not Applicable"]["__abstain__"] == 1


def test_empty_evaluation_is_na_not_perfect():
    assert classification_metrics([], [], ["P1", "P2"])["macro_f1"] is None


def test_retrieval_duplicates_do_not_inflate_recall():
    result = retrieval_metrics(["a", "a", "b"], ["b", "c"], 3)
    assert result["recall_at_k"] == 0.5
    assert result["mrr"] == 0.5
    assert retrieval_metrics([], [], 3)["recall_at_k"] is None


def test_e0_crirar_is_not_applicable_even_with_labels():
    records = [{"scenario_id": "u1", "experiment_id": "E0", "status": "complete"}]
    labels = {"u1": {"cr": "H", "ir": "H", "ar": "H"}}
    report = summarize_runs(records, labels)
    assert report["E0"]["crirar"]["status"] == "not_applicable"


def test_splits_reject_scenario_and_pair_leakage():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits([{"scenario_id": "a", "cve_id": "CVE-2024-1234", "asset_id": "u", "split": "validation"},
                         {"scenario_id": "b", "cve_id": "CVE-2024-1234", "asset_id": "u", "split": "test"}])


def test_frozen_manifest_hashes_configuration(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = freeze_manifest(ExperimentConfig(), path, input_paths=[])
    assert manifest["config"]["experiments"] == ["E0"]
    assert len(manifest["config_sha256"]) == 64
    with pytest.raises(FileExistsError):
        freeze_manifest(ExperimentConfig(), path, input_paths=[])


def test_runner_requires_cloud_consent_before_writing(tmp_path):
    with pytest.raises(PermissionError):
        run_batch([], {}, tmp_path, tmp_path / "out", ExperimentConfig(experiments=("E1",)))
    assert not (tmp_path / "out").exists()


def test_runner_never_passes_hidden_labels(tmp_path):
    with pytest.raises(ValueError, match="Unexpected scenario fields"):
        run_batch([{"scenario_id": "u", "asset_id": "a", "cve_id": "CVE-2024-1234",
                    "expert_labels": {"cr": "H"}}], {"a": {}}, tmp_path, tmp_path / "out", ExperimentConfig())


def test_offline_runner_writes_auditable_outputs_from_public_snapshot(tmp_path):
    from pathlib import Path
    fixture_dir = Path(__file__).parent / "fixtures" / "nvd"
    config = ExperimentConfig(experiments=("E0",), repeats=2)
    records = run_batch([{"scenario_id": "unit-public-replay", "asset_id": "UNIT", "cve_id": "CVE-2021-44228"}],
                        {"UNIT": {"asset_id": "UNIT", "is_synthetic": True}}, fixture_dir,
                        tmp_path / "run", config)
    assert len(records) == 2
    assert all(row["status"] == "complete" for row in records)
    assert all(row["model_input"] is None for row in records)
    logged = [json.loads(line) for line in (tmp_path / "run" / "experiments.jsonl").read_text().splitlines()]
    assert len(logged) == 2
    assert logged[0]["nvd_snapshot_id"] == logged[1]["nvd_snapshot_id"]
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["extra"]["nvd_policy"] == "local-snapshots-only-no-network-fallback"
    assert (tmp_path / "run" / "experiments.csv").is_file()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["E0"]["crirar"]["status"] == "not_applicable"


def test_real_public_fixtures_are_attributed_and_distinct():
    from pathlib import Path
    from nvd_client import load_nvd_snapshot
    fixture_dir = Path(__file__).parent / "fixtures" / "nvd"
    attribution = json.loads((fixture_dir / "source_metadata.json").read_text())
    assert len(attribution) == 5
    assert len({row["cve_id"] for row in attribution}) == 5
    for entry in attribution:
        snapshot = load_nvd_snapshot(fixture_dir / entry["file"], entry["cve_id"])
        assert snapshot["cve_id"] == entry["cve_id"]
        assert snapshot["nvd_snapshot_sha256"] == entry["sha256"]
        assert entry["source_url"].startswith("https://services.nvd.nist.gov/")

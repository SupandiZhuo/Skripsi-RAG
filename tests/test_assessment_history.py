"""Targeted checks for assessment history persistence (Phase 5).

Covers: save/load round-trip, record creation, and deletion. Pure Python/JSON,
no Streamlit/ChromaDB/LLM dependencies.
Run: python -m pytest test_assessment_history.py -q
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from assessment_history import (
    AssessmentRecord,
    delete_record,
    load_history,
    record_from_analysis,
    save_assessment,
)


def test_save_and_load_history_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        record = AssessmentRecord(
            timestamp="2026-10-23T10:00:00+00:00",
            cve_id="CVE-2021-44228",
            asset_id="SRV-001",
            asset_name="Payment Gateway Database",
            base_score=9.8,
            context_score=10.0,
            priority="P1",
            llm_priority="P1",
            vector_string="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            modified_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H/CR:H/IR:H/AR:H",
            retrieval_confidence=0.85,
            evidence={"asset_context": {"asset_id": "SRV-001"}},
        )
        save_assessment(record, path)
        loaded = load_history(path)
        assert len(loaded) == 1
        assert loaded[0].cve_id == "CVE-2021-44228"
        assert loaded[0].priority == "P1"


def test_record_from_analysis_maps_fields():
    asset_metadata = {"asset_id": "SRV-002", "asset_name": "Web Portal"}
    record = record_from_analysis(
        cve_id="CVE-2021-45046",
        asset_metadata=asset_metadata,
        base_score=9.0,
        context_score=9.5,
        priority="P2",
        llm_priority="P1",
        vector_string="CVSS:3.1/...",
        modified_vector="CVSS:3.1/.../CR:H",
        retrieval_confidence=0.75,
        evidence={"justification": "test"},
    )
    assert record.asset_id == "SRV-002"
    assert record.asset_name == "Web Portal"
    assert record.cve_id == "CVE-2021-45046"
    assert record.timestamp.endswith("+00:00")


def test_record_from_analysis_assigns_unique_id():
    asset_metadata = {"asset_id": "SRV-002", "asset_name": "Web Portal"}
    record = record_from_analysis(
        cve_id="CVE-2021-45046",
        asset_metadata=asset_metadata,
        base_score=9.0,
        context_score=9.5,
        priority="P2",
        llm_priority="P1",
        vector_string="CVSS:3.1/...",
        modified_vector="CVSS:3.1/.../CR:H",
        retrieval_confidence=0.75,
    )
    assert record.id is not None
    assert len(record.id) > 0


def test_investigation_and_provenance_survive_history_round_trip(tmp_path):
    result = SimpleNamespace(
        status="needs_verification", score=80,
        investigation_priority="P1", rubric_version="research-test",
        explanation="Verify installed software before remediation ranking.",
        missing_evidence=(), unknown_factors=(), rule_contributions=(),
    )
    record = record_from_analysis(
        cve_id="CVE-2021-44228", asset_metadata={"asset_id": "A"},
        base_score=9.8, context_score=9.8, priority=None,
        llm_priority="P1", vector_string="base", modified_vector="env",
        retrieval_confidence=None, applicability="Unknown",
        priority_result=result,
        model_output={"confidentiality_requirement": "H"},
        provenance={"vector_origin": "nvd", "model_input": {"asset_id": "A"}},
    )
    path = str(tmp_path / "history.json")
    save_assessment(record, path)
    loaded = load_history(path)[0]
    assert loaded.priority is None
    assert loaded.investigation_priority == "P1"
    assert loaded.priority_status == "needs_verification"
    assert loaded.rubric_version == "research-test"
    assert loaded.model_output == {"confidentiality_requirement": "H"}
    assert loaded.provenance is not None
    assert loaded.provenance["model_input"] == {"asset_id": "A"}


def test_delete_record_removes_one_entry():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        r1 = AssessmentRecord(
            timestamp="2026-10-23T10:00:00+00:00",
            cve_id="CVE-2021-44228",
            asset_id="SRV-001",
            asset_name="A",
            base_score=9.8,
            context_score=10.0,
            priority="P1",
            llm_priority="P1",
            vector_string="",
            modified_vector="",
            retrieval_confidence=0.8,
            evidence={},
        )
        r2 = AssessmentRecord(
            timestamp="2026-10-23T11:00:00+00:00",
            cve_id="CVE-2021-45046",
            asset_id="SRV-002",
            asset_name="B",
            base_score=9.0,
            context_score=9.0,
            priority="P2",
            llm_priority="P2",
            vector_string="",
            modified_vector="",
            retrieval_confidence=0.7,
            evidence={},
        )
        save_assessment(r1, path)
        save_assessment(r2, path)
        loaded = load_history(path)
        assert len(loaded) == 2

        delete_record(r1, path)
        remaining = load_history(path)
        assert len(remaining) == 1
        assert remaining[0].cve_id == "CVE-2021-45046"


def test_delete_record_by_id_does_not_affect_similar_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        r1 = AssessmentRecord(
            id="id-1",
            timestamp="2026-10-23T10:00:00+00:00",
            cve_id="CVE-2021-44228",
            asset_id="SRV-001",
            asset_name="A",
            base_score=9.8,
            context_score=10.0,
            priority="P1",
            llm_priority="P1",
            vector_string="",
            modified_vector="",
            retrieval_confidence=0.8,
            evidence={},
        )
        r2 = AssessmentRecord(
            id="id-2",
            timestamp="2026-10-23T10:00:00+00:00",
            cve_id="CVE-2021-44228",
            asset_id="SRV-001",
            asset_name="A",
            base_score=9.8,
            context_score=10.0,
            priority="P1",
            llm_priority="P1",
            vector_string="",
            modified_vector="",
            retrieval_confidence=0.8,
            evidence={},
        )
        save_assessment(r1, path)
        save_assessment(r2, path)
        loaded = load_history(path)
        assert len(loaded) == 2

        delete_record(r1, path)
        remaining = load_history(path)
        assert len(remaining) == 1
        assert remaining[0].id == "id-2"


def test_load_history_returns_empty_for_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent.json")
        assert load_history(path) == []


def test_load_history_returns_empty_for_malformed_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not valid json")
        assert load_history(path) == []


def test_load_history_returns_empty_for_non_list_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"records": []}, f)
        assert load_history(path) == []


def test_load_history_handles_null_evidence():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        record = AssessmentRecord(
            timestamp="2026-10-23T10:00:00+00:00",
            cve_id="CVE-2021-44228",
            asset_id="SRV-001",
            asset_name="A",
            base_score=9.8,
            context_score=10.0,
            priority="P1",
            llm_priority="P1",
            vector_string="",
            modified_vector="",
            retrieval_confidence=0.8,
            evidence=None,
        )
        save_assessment(record, path)
        loaded = load_history(path)
        assert len(loaded) == 1
        assert loaded[0].evidence is None


def test_load_history_skips_malformed_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"cve_id": "ok", "asset_id": "A", "asset_name": "A", "base_score": 1.0,
                        "context_score": 1.0, "priority": "P4", "llm_priority": "P4",
                        "vector_string": "", "modified_vector": "", "retrieval_confidence": 0.1,
                        "evidence": {}, "timestamp": "2026-10-23T10:00:00+00:00"},
                       {"invalid": "record"}], f)
        loaded = load_history(path)
        assert len(loaded) == 1
        assert loaded[0].cve_id == "ok"

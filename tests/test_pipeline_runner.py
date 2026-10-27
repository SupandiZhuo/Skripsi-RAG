"""No live model calls. Scenario records here are unit fixtures only."""
import json
from unittest.mock import Mock

import pytest

from pipeline import run_assessment
from llm_assessment import ContextualAssessment

BASE = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
NVD = {"cve_id": "CVE-2024-1234", "description": "Unit description", "base_score": 9.8,
       "vector_string": BASE, "configurations": [], "vector_origin": "nvd"}
ASSET = {"asset_id": "UNIT", "business_criticality": "High", "expert_labels": {"cr": "L"},
         "ground_truth": "H", "owner": "Private Person", "ip_address": "10.0.0.1"}


def prediction(no_context=False):
    values = ("X", "X", "X") if no_context else ("H", "M", "L")
    return ContextualAssessment.model_validate({
        "confidentiality_requirement": values[0], "integrity_requirement": values[1],
        "availability_requirement": values[2], "organizational_priority": "P2",
        "metric_reasons": {metric: {"reason": "Unit evidence or missing organizational context.",
                                    "evidence_ids": [] if no_context else ["asset:UNIT"]}
                           for metric in ("CR", "IR", "AR")},
        "justification": "Unit fixture evidence.", "remediation": "Unit patch advice."})


def test_e0_has_no_llm_context_or_environmental_predictions():
    analyzer = Mock()
    result = run_assessment(NVD, ASSET, experiment_id="E0", analyzer=analyzer)
    analyzer.assert_not_called()
    assert result["status"] == "complete"
    assert result["predicted_cr"] is None
    assert result["computed_environmental_score"] is None
    assert result["deterministic_priority"] is None
    assert result["model_input"] is None
    json.dumps(result, allow_nan=False)


def test_cloud_consent_is_required_before_model_call():
    analyzer = Mock()
    result = run_assessment(NVD, ASSET, experiment_id="E1", analyzer=analyzer)
    assert result["status"] == "blocked"
    assert result["failures"][0]["error_type"] == "cloud_consent_required"
    analyzer.assert_not_called()


def test_e1_does_not_receive_asset_context():
    analyzer = Mock(return_value=prediction(no_context=True))
    result = run_assessment(NVD, ASSET, experiment_id="E1", cloud_consent=True, analyzer=analyzer)
    assert analyzer.call_args.kwargs["asset_context"] == {}
    assert result["deterministic_priority"] is None
    assert result["computed_environmental_vector"].startswith(BASE + "/CR:X")


def test_e2_sanitizes_hidden_labels_and_personal_fields():
    analyzer = Mock(return_value=prediction())
    result = run_assessment(NVD, ASSET, experiment_id="E2", cloud_consent=True, analyzer=analyzer)
    context = json.dumps(analyzer.call_args.kwargs["asset_context"])
    assert "ground_truth" not in context
    assert "expert_labels" not in context
    assert "Private Person" not in context
    assert "10.0.0.1" not in context
    assert result["deterministic_priority"] is None  # missing applicability stays Unknown
    assert result["status"] == "needs_verification"


def test_e3_missing_document_store_is_not_mislabeled_direct_context():
    analyzer = Mock()
    result = run_assessment(NVD, ASSET, experiment_id="E3", cloud_consent=True, analyzer=analyzer)
    assert result["status"] == "failed"
    assert any(f["error_type"] == "retrieval_unavailable" for f in result["failures"])
    analyzer.assert_not_called()


def test_manual_vector_requires_reason_and_is_outside_main_test_set():
    result = run_assessment(NVD, ASSET, experiment_id="E0", vector_override=BASE)
    assert result["status"] == "failed"
    result = run_assessment(NVD, ASSET, experiment_id="E0", vector_override=BASE,
                            override_reason="Unit fixture alternate source")
    assert result["vector_origin"] == "manual_override"
    assert result["main_test_set_eligible"] is False


def test_schema_error_retains_baseline_and_failure_without_key_leak():
    analyzer = Mock(return_value={"bad": "output"})
    result = run_assessment(NVD, ASSET, experiment_id="E1", cloud_consent=True,
                            api_key="secret-unit-key", analyzer=analyzer)
    assert result["status"] == "failed"
    assert result["base_score"] == 9.8
    assert result["failures"][-1]["error_type"] == "schema_failure"
    assert "secret-unit-key" not in json.dumps(result)


def test_rejects_unsupported_experiment():
    with pytest.raises(ValueError):
        run_assessment(NVD, ASSET, experiment_id="E4")


@pytest.mark.parametrize("has_invented_citation", [False, True])
def test_e1_real_analyze_boundary_accepts_no_context_but_rejects_fake_evidence(monkeypatch, has_invented_citation):
    import llm_assessment
    from langchain_core.runnables import RunnableLambda
    seen = []

    class OfflineStructuredProvider:
        def with_structured_output(self, schema):
            def respond(prompt):
                seen.append(prompt)
                return prediction(no_context=not has_invented_citation)
            return RunnableLambda(respond)

    monkeypatch.setattr(llm_assessment, "_build_llm", lambda api_key, model: OfflineStructuredProvider())
    result = run_assessment(NVD, ASSET, experiment_id="E1", cloud_consent=True,
                            api_key="offline-fixture-not-a-real-key")
    assert seen  # Exercises analyze_with_llm + actual prompt boundary, not injected analyzer.
    rendered = str(seen[0])
    assert "Private Person" not in rendered
    assert "asset:UNIT" not in rendered
    assert result["model_input"]["asset_context"] == {}
    if has_invented_citation:
        assert result["status"] == "failed"
        assert result["computed_environmental_score"] is None
        assert result["failures"][-1]["error_type"] == "schema_failure"
    else:
        assert result["status"] == "complete"
        assert result["predicted_cr"] == "X"


def test_e3_records_verified_scoped_document_input():
    import hashlib
    from langchain_core.documents import Document
    from rag_retrieval import context_chunk_id
    content = "Unit service requires transaction integrity and documented recovery."
    metadata = {"asset_id": "UNIT", "source_id": "unit-source", "document_id": "unit-doc",
                "chunk_index": 0, "source": "unit fixture", "source_reference": "unit:fixture",
                "last_updated": "2026-01-01", "is_synthetic": True,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest()}
    metadata["chunk_id"] = context_chunk_id("UNIT", "unit-source", "unit-doc", 0)
    store = Mock(spec=["similarity_search_with_relevance_scores"])
    store.similarity_search_with_relevance_scores.return_value = [(Document(page_content=content, metadata=metadata), 0.8)]
    analyzer = Mock(return_value=prediction())
    result = run_assessment(NVD, ASSET, experiment_id="E3", context_store=store,
                            cloud_consent=True, analyzer=analyzer)
    assert result["retrieved_document_ids"] == ["unit-doc"]
    assert result["retrieval_scores"] == [0.8]
    assert content in json.dumps(result["model_input"])
    assert store.similarity_search_with_relevance_scores.call_args.kwargs["filter"] == {"asset_id": "UNIT"}
    assert result["status"] == "needs_verification"


def test_model_cannot_cite_unsupplied_evidence_even_through_injected_adapter():
    analyzer = Mock(return_value=prediction())
    result = run_assessment(NVD, ASSET, experiment_id="E1", cloud_consent=True, analyzer=analyzer)
    assert result["status"] == "failed"
    assert result["failures"][-1]["error_type"] == "schema_failure"
    assert result["computed_environmental_score"] is None

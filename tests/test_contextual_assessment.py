"""Targeted checks for the revised contextual assessment architecture.

Covers: schema validation (valid/invalid CR/IR/AR, valid/invalid P1-P4),
Base metric preservation, absence of MAV, deterministic Environmental
vector construction, and CVSS3 parsing of the generated vector.

No live API calls are made - analyze_with_llm's LLM call is mocked.
Run: python -m pytest test_contextual_assessment.py -q
"""

from unittest.mock import patch

import pytest
from cvss import CVSS3
from pydantic import ValidationError

from cvss_engine import build_environmental_vector
from llm_assessment import (
    CiaRequirement,
    ContextualAssessment,
    LlmAssessmentError,
    OrganizationalPriority,
    analyze_with_llm,
)

BASE_VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def make_assessment(cr="H", ir="M", ar="L", priority="P1"):
    return ContextualAssessment(
        confidentiality_requirement=cr,
        integrity_requirement=ir,
        availability_requirement=ar,
        metric_reasons={
            metric: {"reason": f"Independent {metric} requirement from asset registry evidence.",
                     "evidence_ids": ["asset:A1"]}
            for metric in ("CR", "IR", "AR")
        },
        organizational_priority=priority,
        justification="Asset holds regulated customer data per RAG context.",
        remediation="Patch immediately and restrict network access.",
    )


# ---------------------------------------------------------------------------
# 1. Valid CR/IR/AR output
# ---------------------------------------------------------------------------
def test_valid_cia_requirement_values():
    for v in ("X", "L", "M", "H"):
        assessment = make_assessment(cr=v, ir=v, ar=v)
        assert assessment.confidentiality_requirement.value == v
        assert assessment.integrity_requirement.value == v
        assert assessment.availability_requirement.value == v


# ---------------------------------------------------------------------------
# 2. Invalid metric rejection
# ---------------------------------------------------------------------------
def test_invalid_cia_requirement_rejected():
    with pytest.raises(ValidationError):
        make_assessment(cr="CRITICAL")
    with pytest.raises(ValidationError):
        make_assessment(ir="Medium")  # must be short code, not full word
    with pytest.raises(ValidationError):
        make_assessment(ar="Z")


# ---------------------------------------------------------------------------
# 3. Valid / invalid P1-P4 output
# ---------------------------------------------------------------------------
def test_valid_organizational_priority():
    for p in ("P1", "P2", "P3", "P4"):
        assessment = make_assessment(priority=p)
        assert assessment.organizational_priority.value == p


def test_invalid_organizational_priority_rejected():
    with pytest.raises(ValidationError):
        make_assessment(priority="P0")
    with pytest.raises(ValidationError):
        make_assessment(priority="Critical")


def test_blank_justification_rejected():
    payload = make_assessment().model_dump()
    payload["justification"] = "   "
    with pytest.raises(ValidationError):
        ContextualAssessment(**payload)


def test_metric_reasons_are_required():
    payload = make_assessment().model_dump()
    del payload["metric_reasons"]
    with pytest.raises(ValidationError, match="metric_reasons"):
        ContextualAssessment(**payload)


@pytest.mark.parametrize("metric", ["CR", "IR", "AR"])
def test_defined_metric_requires_cited_evidence(metric):
    payload = make_assessment().model_dump()
    payload["metric_reasons"][metric]["evidence_ids"] = []
    with pytest.raises(ValidationError, match=f"{metric} requires cited evidence"):
        ContextualAssessment(**payload)


def test_undefined_metrics_allow_explicit_insufficient_evidence_reasons():
    payload = make_assessment(cr="X", ir="X", ar="X").model_dump()
    payload["metric_reasons"] = {
        metric: {"reason": "Insufficient organizational evidence.", "evidence_ids": []}
        for metric in ("CR", "IR", "AR")
    }
    assessment = ContextualAssessment(**payload)
    assert all(not getattr(assessment.metric_reasons, metric).evidence_ids for metric in ("CR", "IR", "AR"))


# ---------------------------------------------------------------------------
# 4. Preservation of original Base metrics + 5. absence of MAV modification
# ---------------------------------------------------------------------------
def test_base_metrics_preserved_and_no_mav():
    assessment = make_assessment(cr="H", ir="M", ar="L")
    modified = build_environmental_vector(BASE_VECTOR, assessment)

    for base_metric in ("AV:N", "AC:L", "PR:N", "UI:N", "S:U", "C:H", "I:H", "A:H"):
        assert base_metric in modified

    assert "MAV:" not in modified
    assert "CR:H" in modified
    assert "IR:M" in modified
    assert "AR:L" in modified


@pytest.mark.parametrize("suffix", ["/CR:X", "/IR:X", "/AR:X", "/MAV:A", "/CR:X/IR:X/AR:X/MAV:A"])
def test_preexisting_environmental_metrics_rejected(suffix):
    # NVD input must be Base-only; never silently strip contextual metrics.
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_environmental_vector(BASE_VECTOR + suffix, make_assessment())


def test_invalid_base_vector_rejected():
    with pytest.raises(ValueError):
        build_environmental_vector("not-a-vector", make_assessment())


# ---------------------------------------------------------------------------
# 6. Deterministic Environmental vector construction + 8. CVSS3 parsing
# ---------------------------------------------------------------------------
def test_cvss3_parses_generated_vector_and_scores_deterministically():
    assessment = make_assessment(cr="H", ir="H", ar="H")
    modified = build_environmental_vector(BASE_VECTOR, assessment)

    cvss_obj = CVSS3(modified)
    base_score, temporal_score, env_score = cvss_obj.scores()

    assert base_score == pytest.approx(9.8)
    assert env_score >= base_score  # higher CR/IR/AR should not lower env score
    severities = cvss_obj.severities()
    assert severities[0] == "Critical"


# ---------------------------------------------------------------------------
# 7. Malformed/refused model output handling (mocked, no live API calls)
# ---------------------------------------------------------------------------
def test_analyze_with_llm_raises_on_malformed_response():
    with patch("llm_assessment._build_llm"):
        # Simulate `prompt | structured_llm` chain whose .invoke() returns
        # something that is not a ContextualAssessment (malformed output).
        class FakeChain:
            def invoke(self, _):
                return {"not": "a ContextualAssessment"}

        with patch("llm_assessment.ChatPromptTemplate.__or__", return_value=FakeChain()):
            with pytest.raises(LlmAssessmentError, match="Unrecognized structured model response"):
                analyze_with_llm(
                    api_key="fake-key",
                    cve_desc="Some CVE description",
                    original_vector=BASE_VECTOR,
                    asset_context={"asset_id": "A1"},
                )


def test_analyze_with_llm_raises_on_exception_during_invoke():
    class RaisingChain:
        def invoke(self, _):
            raise RuntimeError("simulated provider failure")

    with patch("llm_assessment._build_llm"), patch("llm_assessment.ChatPromptTemplate.__or__", return_value=RaisingChain()):
        with pytest.raises(LlmAssessmentError, match="simulated provider failure"):
            analyze_with_llm(
                api_key="fake-key",
                cve_desc="Some CVE description",
                original_vector=BASE_VECTOR,
                asset_context={"asset_id": "A1"},
            )


def test_analyze_with_llm_returns_valid_assessment_when_mocked():
    expected = make_assessment(cr="M", ir="M", ar="L", priority="P2")

    class FakeChain:
        def invoke(self, _):
            return expected

    with patch("llm_assessment._build_llm"), patch("llm_assessment.ChatPromptTemplate.__or__", return_value=FakeChain()):
        result = analyze_with_llm(
            api_key="fake-key",
            cve_desc="Some CVE description",
            original_vector=BASE_VECTOR,
            asset_context={"asset_id": "A1"},
        )

    assert result == expected  # Provider output is revalidated, not returned unchecked.
    assert result.confidentiality_requirement == CiaRequirement.MEDIUM
    assert result.organizational_priority == OrganizationalPriority.P2


def test_analyze_with_llm_rejects_citations_absent_from_supplied_context():
    class FakeChain:
        def invoke(self, _):
            return make_assessment()  # Cites asset:A1, not the supplied asset:A2.

    with patch("llm_assessment._build_llm"), patch("llm_assessment.ChatPromptTemplate.__or__", return_value=FakeChain()):
        with pytest.raises(LlmAssessmentError, match="evidence IDs absent from model input"):
            analyze_with_llm(
                api_key="fake-key", cve_desc="Some CVE description",
                original_vector=BASE_VECTOR, asset_context={"asset_id": "A2"},
            )

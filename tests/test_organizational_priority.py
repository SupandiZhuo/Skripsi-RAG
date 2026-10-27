"""Targeted checks for the deterministic Organizational Priority engine (Phase C).

Covers: severity bands, business criticality, exposure, CPE applicability,
inactive graph inputs, missing evidence handling, applicability gating,
reproducibility, and separation from CVSS severity.
Run: python -m pytest test_organizational_priority.py -q
"""

from dataclasses import asdict

import pytest

from cve_applicability import ApplicabilityStatus, ApplicabilityResult
from organizational_priority import (
    FACTOR_MAX_POINTS,
    MAX_SCORE,
    OrganizationalPriorityResult,
    calculate_organizational_priority,
    _level_for_score,
)


def _applicability(status: ApplicabilityStatus) -> ApplicabilityResult:
    return ApplicabilityResult(status=status, evidence=(), explanation="test")


def test_zero_environmental_score_is_not_replaced_with_base_score():
    result = calculate_organizational_priority(
        {"asset_id": "A", "business_criticality": "Low"},
        base_score=9.8, context_score=0.0,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    severity = next(c for c in result.rule_contributions if c.factor == "vulnerability severity")
    assert severity.points == 0
    assert severity.evidence_source == "contextual.cvss_environmental_score"
    assert not any("vulnerability severity" in value for value in result.missing_evidence)
    assert result.rubric_version == "organizational-priority-v2.0"


def test_not_applicable_has_no_remediation_or_investigation_priority():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "Mission Critical",
        "internet_exposed": "yes",
        "network_zone": "DMZ",
        "data_sensitivity": "Restricted",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=9.8,
        applicability=_applicability(ApplicabilityStatus.NOT_APPLICABLE),
    )
    assert result.status == "not_applicable"
    assert result.priority is None
    assert result.level is None
    assert result.score is None
    assert result.investigation_priority is None


def test_confirmed_mission_critical_exposed_is_p1():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "Mission Critical",
        "internet_exposed": "yes",
        "network_zone": "DMZ",
        "data_sensitivity": "Restricted",
        "environment": "production",
        "confidentiality_requirement": "H",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=9.8,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    assert result.priority == "P1"
    assert result.level == "Urgent"
    assert any(c.factor == "CPE applicability" and c.points == 15 for c in result.rule_contributions)


def test_unknown_applicability_and_severity_gives_incomplete():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "no",
        "network_zone": "Internal",
    }
    result = calculate_organizational_priority(asset)
    assert result.status == "incomplete"
    assert result.priority is None
    assert any("vulnerability severity" in m for m in result.missing_evidence)
    assert any("CPE applicability" in m for m in result.missing_evidence)


def test_missing_criticality_gives_incomplete():
    asset = {
        "asset_id": "SRV-001",
        "internet_exposed": "yes",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=9.8,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "incomplete"
    assert "business criticality" in result.missing_evidence


def test_exposure_inferred_from_zone():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "unknown",
        "network_zone": "Internal",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    contrib = next(c for c in result.rule_contributions if c.factor == "internet exposure")
    assert contrib.points == 0
    assert "Internal" in contrib.evidence_value


def test_dmz_gives_partial_exposure_points():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "unknown",
        "network_zone": "DMZ",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    contrib = next(c for c in result.rule_contributions if c.factor == "internet exposure")
    assert contrib.points == 10


@pytest.mark.parametrize("deps", [
    {"incoming_count": 2, "outgoing_count": 1},
    {"incoming_count": "not-a-count", "outgoing_count": -1},
])
def test_graph_dependencies_are_accepted_but_inactive(deps):
    asset = {"asset_id": "SRV-001", "business_criticality": "High"}
    kwargs = {"context_score": 7.5, "applicability": _applicability(ApplicabilityStatus.CONFIRMED)}
    baseline = calculate_organizational_priority(asset, **kwargs)
    result = calculate_organizational_priority(asset, graph_dependencies=deps, **kwargs)
    assert result == baseline
    assert not any(c.factor == "business dependencies" for c in result.rule_contributions)
    assert "business dependencies" not in result.unknown_factors


def test_priority_without_verified_relationships_uses_no_dependency_points():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "yes",
        "network_zone": "DMZ",
    }
    kwargs = {
        "context_score": 7.5,
        "applicability": _applicability(ApplicabilityStatus.CONFIRMED),
    }

    without_graph = calculate_organizational_priority(asset, **kwargs)
    with_empty_graph = calculate_organizational_priority(asset, graph_dependencies={}, **kwargs)

    assert without_graph == with_empty_graph
    assert without_graph.priority == "P2"
    assert not any(c.factor == "business dependencies" for c in without_graph.rule_contributions)
    assert "business dependencies" not in without_graph.unknown_factors


def test_low_criticality_internal_low_severity_is_p4():
    asset = {
        "asset_id": "SRV-003",
        "business_criticality": "Low",
        "internet_exposed": "no",
        "network_zone": "Internal",
        "data_sensitivity": "Internal",
    }
    result = calculate_organizational_priority(
        asset,
        base_score=2.0,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    assert result.priority == "P4"
    assert result.level == "Low"


def test_unknown_optional_factors_are_listed():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "yes",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert "data sensitivity" in result.unknown_factors
    assert "environment" in result.unknown_factors
    assert "business dependencies" not in result.unknown_factors


def test_unknown_exposure_does_not_block_classification():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "unknown",
        "network_zone": "unknown",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    assert result.priority is not None
    assert "internet exposure" in result.unknown_factors
    exposure = next(c for c in result.rule_contributions if c.factor == "internet exposure")
    assert exposure.points == 0


def test_reproducible_for_same_inputs():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "Mission Critical",
        "internet_exposed": "yes",
    }
    args = (asset,)
    kwargs = {"context_score": 9.8, "applicability": _applicability(ApplicabilityStatus.CONFIRMED)}
    r1 = calculate_organizational_priority(*args, **kwargs)
    r2 = calculate_organizational_priority(*args, **kwargs)
    assert r1 == r2


def test_context_score_preferred_over_base_score():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "yes",
    }
    result = calculate_organizational_priority(
        asset,
        base_score=9.8,
        context_score=5.0,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    severity = next(c for c in result.rule_contributions if c.factor == "vulnerability severity")
    assert "5.0" in severity.evidence_value or "Medium" in severity.evidence_value


def test_result_contains_cve_and_asset_identifiers():
    asset = {"asset_id": "SRV-001", "business_criticality": "High", "internet_exposed": "yes"}
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
        cve_id="CVE-2021-44228",
    )
    assert result.asset_id == "SRV-001"
    assert result.cve_id == "CVE-2021-44228"


def test_priority_is_separate_from_cvss_severity():
    """A Critical CVSS alone on a Low internal asset does not reach P1."""
    asset = {
        "asset_id": "SRV-003",
        "business_criticality": "Low",
        "internet_exposed": "no",
        "network_zone": "Internal",
        "data_sensitivity": "Public",
    }
    result = calculate_organizational_priority(
        asset,
        context_score=9.8,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.priority in ("P3", "P4")
    assert result.priority != "P1"



def _maximum_asset():
    return {
        "asset_id": "SRV-MAX", "business_criticality": "Mission Critical",
        "internet_exposed": "yes", "network_zone": "DMZ",
        "data_sensitivity": "Restricted", "environment": "production",
        "confidentiality_requirement": "H",
    }


def test_active_factor_maximum_is_exactly_100_and_attainable():
    result = calculate_organizational_priority(
        _maximum_asset(), base_score=10.0,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
        graph_dependencies={"incoming_count": 100, "outgoing_count": 100},
    )
    assert MAX_SCORE == sum(FACTOR_MAX_POINTS.values()) == 100
    assert result.score == result.max_score == 100
    assert sum(c.points for c in result.rule_contributions) == result.score
    assert sum(c.max_points for c in result.rule_contributions) == 100
    assert {c.factor for c in result.rule_contributions} == set(FACTOR_MAX_POINTS)
    assert result.investigation_priority is None


@pytest.mark.parametrize("applicability", [
    _applicability(ApplicabilityStatus.UNKNOWN), {"status": "Unknown"},
])
def test_unknown_is_investigation_not_remediation_even_on_maximum_asset(applicability):
    result = calculate_organizational_priority(
        _maximum_asset(), base_score=10.0, applicability=applicability,
    )
    assert result.status == "needs_verification"
    assert result.priority is None
    assert result.level is None
    assert result.investigation_priority == "P1"
    assert result.score == sum(c.points for c in result.rule_contributions)
    assert "investigation" in result.explanation.lower()
    assert "not a remediation" in result.explanation.lower()


def test_unknown_with_missing_criticality_still_needs_verification():
    result = calculate_organizational_priority(
        {}, base_score=9.8, applicability=_applicability(ApplicabilityStatus.UNKNOWN),
    )
    assert result.status == "needs_verification"
    assert result.priority is None
    assert result.investigation_priority is None
    assert result.score is None
    assert "business criticality" in result.missing_evidence


@pytest.mark.parametrize("applicability", [None, {"status": "invalid"}, {}])
def test_missing_or_invalid_applicability_never_assigns_remediation_priority(applicability):
    result = calculate_organizational_priority(
        _maximum_asset(), base_score=9.8, applicability=applicability,
    )
    assert result.status == "incomplete"
    assert result.priority is None
    assert result.investigation_priority is None
    assert "CPE applicability" in result.missing_evidence


@pytest.mark.parametrize("severity,criticality,exposure,expected", [
    (9.8, "High", "yes", "P2"),
    (2.0, "Low", "no", "P4"),
])
def test_unknown_investigation_uses_same_thresholds(severity, criticality, exposure, expected):
    result = calculate_organizational_priority(
        {"business_criticality": criticality, "internet_exposed": exposure},
        base_score=severity, applicability=_applicability(ApplicabilityStatus.UNKNOWN),
    )
    assert result.status == "needs_verification"
    assert result.priority is None
    assert result.investigation_priority == expected


@pytest.mark.parametrize("severity,criticality,exposure,sensitivity,environment,cia", [
    (9.8, "Mission Critical", "yes", "Restricted", "production", "H"),
    (7.5, "High", "unknown", "Confidential", "staging", "M"),
    (5.0, "Medium", "no", "Internal", "development", "L"),
    (2.0, "Low", "no", "Public", "unknown", "X"),
])
def test_scores_stay_bounded_and_equal_contributions(severity, criticality, exposure, sensitivity, environment, cia):
    result = calculate_organizational_priority(
        {"business_criticality": criticality, "internet_exposed": exposure,
         "data_sensitivity": sensitivity, "environment": environment,
         "confidentiality_requirement": cia},
        base_score=severity, applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    assert 0 <= result.score <= result.max_score == 100
    assert result.score == sum(c.points for c in result.rule_contributions)



@pytest.mark.parametrize("score,priority,level", [
    (0, "P4", "Low"), (39, "P4", "Low"),
    (40, "P3", "Normal"), (59, "P3", "Normal"),
    (60, "P2", "High"), (79, "P2", "High"),
    (80, "P1", "Urgent"), (100, "P1", "Urgent"),
])
def test_priority_threshold_boundaries(score, priority, level):
    assert _level_for_score(score) == (priority, level)


def test_unknown_without_severity_discloses_missing_evidence():
    result = calculate_organizational_priority(
        {"business_criticality": "High"},
        applicability=_applicability(ApplicabilityStatus.UNKNOWN),
    )
    assert result.status == "needs_verification"
    assert result.priority is None
    assert result.investigation_priority is not None
    assert any("vulnerability severity" in item for item in result.missing_evidence)


def test_confirmed_without_severity_retains_partial_evidence_policy():
    result = calculate_organizational_priority(
        {"business_criticality": "High"},
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
    )
    assert result.status == "complete"
    assert result.priority is not None
    assert result.investigation_priority is None
    assert any("vulnerability severity" in item for item in result.missing_evidence)


def test_investigation_serialization_preserves_nullable_remediation_fields():
    result = calculate_organizational_priority(
        _maximum_asset(), base_score=9.8,
        applicability=_applicability(ApplicabilityStatus.UNKNOWN), cve_id="CVE-2021-44228",
    )
    record = asdict(result)
    assert record["status"] == "needs_verification"
    assert record["priority"] is None
    assert record["level"] is None
    assert record["investigation_priority"] == "P1"
    assert record["asset_id"] == "SRV-MAX"
    assert record["cve_id"] == "CVE-2021-44228"

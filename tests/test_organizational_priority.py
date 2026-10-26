"""Targeted checks for the deterministic Organizational Priority engine (Phase C).

Covers: severity bands, business criticality, exposure, CPE applicability,
graph dependencies, missing evidence handling, Not Applicable capping,
reproducibility, and separation from CVSS severity.
Run: python -m pytest test_organizational_priority.py -q
"""

import pytest

from cve_applicability import ApplicabilityStatus, ApplicabilityResult
from organizational_priority import (
    MAX_SCORE,
    OrganizationalPriorityResult,
    calculate_organizational_priority,
)


def _applicability(status: ApplicabilityStatus) -> ApplicabilityResult:
    return ApplicabilityResult(status=status, evidence=(), explanation="test")


def test_not_applicable_caps_at_p4():
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
    assert result.priority == "P4"
    assert result.level == "Low"


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


def test_graph_dependencies_add_points():
    asset = {
        "asset_id": "SRV-001",
        "business_criticality": "High",
        "internet_exposed": "no",
        "network_zone": "Internal",
    }
    deps = {"incoming_count": 2, "outgoing_count": 1}
    result = calculate_organizational_priority(
        asset,
        context_score=7.5,
        applicability=_applicability(ApplicabilityStatus.CONFIRMED),
        graph_dependencies=deps,
    )
    contrib = next(c for c in result.rule_contributions if c.factor == "business dependencies")
    assert contrib.points == 15


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
    assert "business dependencies" in without_graph.unknown_factors


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
    assert "business dependencies" in result.unknown_factors


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

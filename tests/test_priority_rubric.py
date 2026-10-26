"""Targeted checks for the deterministic Organizational Priority rubric (Phase 2C).

Covers: high-criticality assets, internet-exposed assets, low-criticality
internal assets, missing/unknown metadata (explicit incomplete state),
conflicting evidence, reproducibility, backward compatibility with v1-style
metadata, and rubric design invariants (severity cap, score consistency).

No live API calls, no LLM, no ChromaDB - the rubric is a pure function.
Run: python -m pytest test_priority_rubric.py -q
"""

import pytest

from asset_schema_v2 import load_assets
from priority_rubric import (
    MAX_SCORE,
    PRIORITY_LEVELS,
    calculate_organizational_priority,
)

# Asset contexts mirroring data/asset_data_v2.csv (Asset Schema v2).
SRV_001 = {  # Payment Gateway Database: Restricted zone, Mission Critical
    "asset_id": "SRV-001",
    "network_zone": "Restricted",
    "business_criticality": "Mission Critical",
    "data_sensitivity": "Restricted",
    "environment": "unknown",
    "internet_exposed": "no",
    "confidentiality_requirement": "X",
    "integrity_requirement": "X",
    "availability_requirement": "X",
}
SRV_002 = {  # Corporate Web Portal: DMZ, High, Public, internet-facing
    "asset_id": "SRV-002",
    "network_zone": "DMZ",
    "business_criticality": "High",
    "data_sensitivity": "Public",
    "environment": "unknown",
    "internet_exposed": "yes",
    "confidentiality_requirement": "X",
    "integrity_requirement": "X",
    "availability_requirement": "X",
}
SRV_003 = {  # Cafeteria menu: Internal, Low, exposure unknown
    "asset_id": "SRV-003",
    "network_zone": "Internal",
    "business_criticality": "Low",
    "data_sensitivity": "Internal",
    "environment": "unknown",
    "internet_exposed": "unknown",
}
SRV_004 = {  # Active Directory DC: Restricted zone, Mission Critical
    "asset_id": "SRV-004",
    "network_zone": "Restricted",
    "business_criticality": "Mission Critical",
    "data_sensitivity": "Restricted",
    "environment": "unknown",
    "internet_exposed": "unknown",
}


def _points(result, factor):
    return next(c.points for c in result.rule_contributions if c.factor == factor)


# ---------------------------------------------------------------------------
# 1. High-criticality assets
# ---------------------------------------------------------------------------
def test_mission_critical_with_critical_vuln_is_p1():
    result = calculate_organizational_priority(SRV_001, 9.8)
    assert result.status == "complete"
    assert result.priority == "P1"
    # 35 (Critical) + 30 (Mission Critical) + 0 (not exposed) + 8 (Restricted)
    assert result.score == 73
    assert _points(result, "vulnerability_severity") == 35
    assert _points(result, "business_criticality") == 30


def test_high_criticality_internet_exposed_with_high_vuln_is_p2():
    result = calculate_organizational_priority(SRV_002, 8.0)
    assert result.status == "complete"
    assert result.priority == "P2"
    # 27 (High) + 22 (High crit) + 20 (exposed) + 0 (Public) = 69 < 70
    assert result.score == 69
    assert _points(result, "exposure") == 20


# ---------------------------------------------------------------------------
# 2. Internet-exposed assets
# ---------------------------------------------------------------------------
def test_internet_exposure_increases_priority_score():
    exposed = dict(SRV_002, internet_exposed="yes")
    unexposed = dict(SRV_002, internet_exposed="no")
    r_exposed = calculate_organizational_priority(exposed, 8.0)
    r_unexposed = calculate_organizational_priority(unexposed, 8.0)
    assert _points(r_exposed, "exposure") == 20
    assert _points(r_unexposed, "exposure") == 0
    assert r_exposed.score > r_unexposed.score
    assert PRIORITY_LEVELS.index(r_exposed.priority) < PRIORITY_LEVELS.index(
        r_unexposed.priority
    )


def test_dmzone_zone_derives_partial_exposure_when_flag_unknown():
    result = calculate_organizational_priority(dict(SRV_002, internet_exposed="unknown"), 8.0)
    assert result.status == "complete"
    assert _points(result, "exposure") == 12  # DMZ-derived, below explicit=yes
    exposure = next(c for c in result.rule_contributions if c.factor == "exposure")
    assert "tersirat" in exposure.evidence_source


# ---------------------------------------------------------------------------
# 3. Low-criticality internal assets
# ---------------------------------------------------------------------------
def test_low_criticality_internal_medium_vuln_is_p4():
    result = calculate_organizational_priority(SRV_003, 5.0)
    assert result.status == "complete"
    assert result.priority == "P4"
    # 14 (Medium) + 3 (Low crit) + 0 (internal zone) + 2 (Internal data) = 19
    assert result.score == 19


def test_low_criticality_internal_high_vuln_is_p3():
    result = calculate_organizational_priority(SRV_003, 8.0)
    assert result.priority == "P3"
    assert result.score == 32


def test_low_severity_never_exceeds_p2_even_on_max_stakes_asset():
    max_ctx = dict(
        SRV_001,
        internet_exposed="yes",
        environment="production",
        confidentiality_requirement="H",
    )
    result = calculate_organizational_priority(max_ctx, 3.0)  # Low severity = 4 pts
    assert result.status == "complete"
    # 4 + 30 + 20 + 8 + 5 + 2 = 69 < 70 -> at most P2 by design
    assert result.score == 69
    assert result.priority == "P2"


# ---------------------------------------------------------------------------
# 4. Missing / unknown metadata -> explicit incomplete state
# ---------------------------------------------------------------------------
def test_empty_asset_context_is_incomplete_not_low_risk():
    result = calculate_organizational_priority({}, 9.8)
    assert result.status == "incomplete"
    assert result.priority is None
    assert result.score is None
    assert "business_criticality" in result.unknown_factors
    assert "exposure" in result.unknown_factors
    assert "risiko rendah" in result.explanation


def test_missing_nvd_base_score_is_incomplete():
    for bad in (None, 0.0, "not-a-number", -1.0, 11.0):
        result = calculate_organizational_priority(SRV_001, bad)
        assert result.status == "incomplete", bad
        assert result.priority is None
        assert "vulnerability_severity" in result.unknown_factors


def test_unknown_optional_factors_do_not_block_classification():
    result = calculate_organizational_priority(SRV_003, 5.0)
    # environment + cia fields unknown/absent, but required factors known
    assert result.status == "complete"
    assert "environment" in result.unknown_factors
    assert "cia_baseline" in result.unknown_factors
    assert result.priority == "P4"


# ---------------------------------------------------------------------------
# 5. Conflicting evidence
# ---------------------------------------------------------------------------
def test_conflicting_exposure_evidence_uses_higher_risk_and_records_conflict():
    conflicting = dict(SRV_003, internet_exposed="yes")  # Internal zone + exposed=yes
    result = calculate_organizational_priority(conflicting, 8.0)
    assert result.status == "complete"
    assert _points(result, "exposure") == 20  # fail-safe: explicit yes wins
    assert any("bertentangan" in c for c in result.conflicts)
    assert "bertentangan" in result.explanation


# ---------------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------------
def test_calculation_is_reproducible():
    first = calculate_organizational_priority(SRV_001, 9.8)
    second = calculate_organizational_priority(dict(SRV_001), 9.8)
    assert first == second  # frozen dataclasses: full structural equality


def test_contributions_sum_to_score():
    result = calculate_organizational_priority(SRV_001, 9.8)
    assert sum(c.points for c in result.rule_contributions) == result.score
    assert result.max_score == MAX_SCORE


def test_priority_always_in_enum_or_none():
    for ctx, score in ((SRV_001, 9.8), (SRV_002, 5.0), (SRV_003, 1.0), ({}, 9.8)):
        result = calculate_organizational_priority(ctx, score)
        assert result.priority is None or result.priority in PRIORITY_LEVELS


# ---------------------------------------------------------------------------
# 7. Backward compatibility
# ---------------------------------------------------------------------------
def test_v1_style_metadata_without_v2_fields_still_classifies():
    v1_context = {  # v1 asset_data.csv columns only
        "asset_id": "SRV-005",
        "asset_name": "HR Document Storage",
        "ip_address": "10.0.20.15",
        "network_zone": "Internal",
        "business_criticality": "High",
        "data_sensitivity": "Confidential",
        "description": "File server dengan data PII karyawan.",
    }
    result = calculate_organizational_priority(v1_context, 8.0)
    assert result.status == "complete"
    # 27 + 22 + 0 (zone-derived internal) + 5 = 54 -> P2
    assert result.priority == "P2"
    assert result.score == 54
    assert "environment" in result.unknown_factors


def test_full_v2_dataset_classifies_deterministically():
    assets = load_assets("data/asset_data_v2.csv")
    expected = {
        "SRV-001": "P2",  # 27+30+0+8 = 65
        "SRV-002": "P2",  # 27+22+20+0 = 69
        "SRV-003": "P3",  # 27+3+0+2 = 32
        "SRV-004": "P2",  # 27+30+0+8 = 65 (exposure zone-derived)
        "SRV-005": "P2",  # 27+22+0+5 = 54
        "SRV-006": "P2",  # 27+22+12+0 = 61 (DMZ-derived exposure)
        "SRV-007": "P3",  # 27+11+0+2 = 40
    }
    for asset in assets:
        result = calculate_organizational_priority(asset, 7.5)
        assert result.status == "complete", asset["asset_id"]
        assert result.priority == expected[asset["asset_id"]], asset["asset_id"]
        # Reproducible on the same row
        assert result == calculate_organizational_priority(asset, 7.5)

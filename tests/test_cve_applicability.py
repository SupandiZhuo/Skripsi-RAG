"""Hardened tests for CVE applicability matching (Phase A.1).

Covers: version-range boundaries (inclusive/exclusive), mixed AND/OR logic,
nested child nodes, negate flags, multiple installed CPEs, missing/unknown
inventory, unsupported version schemes, false-positive prevention, and
backward compatibility.
Run: python -m pytest test_cve_applicability.py -q
"""

import pytest

from cve_applicability import (
    ApplicabilityStatus,
    cpe_matches,
    determine_applicability,
    parse_cpe,
)


def _cpe_match(criteria: str, vulnerable: bool = True, **ranges) -> dict:
    m = {"vulnerable": vulnerable, "criteria": criteria, "matchCriteriaId": "mc-001"}
    m.update(ranges)
    return m


def _node(operator: str = "OR", negate: bool = False, matches=None, children=None):
    node = {"operator": operator, "negate": negate}
    if matches:
        node["cpeMatch"] = matches
    if children:
        node["children"] = children
    return node


def _config(*nodes) -> dict:
    return {"nodes": list(nodes)}


def _configs(*configs) -> dict:
    return {"configurations": list(configs)}


# ---------------------------------------------------------------------------
# Basic parsing and exact matching
# ---------------------------------------------------------------------------
def test_parse_cpe_extracts_parts():
    parts = parse_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
    assert parts["part"] == "a"
    assert parts["vendor"] == "apache"
    assert parts["product"] == "log4j"
    assert parts["version"] == "2.14.1"


def test_cpe_matches_exact_and_wildcard():
    assert cpe_matches(
        "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
        "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
    )
    assert cpe_matches(
        "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
        "cpe:2.3:a:*:log4j:2.14.1:*:*:*:*:*:*:*",
    )
    assert not cpe_matches(
        "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
        "cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*",
    )


# ---------------------------------------------------------------------------
# Version-range boundaries
# ---------------------------------------------------------------------------
def test_version_start_inclusive():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.0.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED
    assert determine_applicability("cpe:2.3:a:apache:log4j:1.9.9:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_version_start_exclusive_and_end_exclusive():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartExcluding="2.0.0", versionEndExcluding="2.17.0"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.0.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.0.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Mixed AND/OR
# ---------------------------------------------------------------------------
def test_and_requires_all_conditions():
    cfg = _configs(_config(_node(operator="AND", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
        _cpe_match("cpe:2.3:o:microsoft:windows:10:*:*:*:*:*:*:*"),
    ])))
    # Both match
    assert determine_applicability(
        ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", "cpe:2.3:o:microsoft:windows:10:*:*:*:*:*:*:*"],
        cfg,
    ).status == ApplicabilityStatus.CONFIRMED
    # Only one matches
    assert determine_applicability(
        ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", "cpe:2.3:o:linux:linux_kernel:5.0:*:*:*:*:*:*:*"],
        cfg,
    ).status == ApplicabilityStatus.NOT_APPLICABLE


def test_or_any_condition():
    cfg = _configs(_config(_node(operator="OR", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
        _cpe_match("cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.16.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Nested child nodes
# ---------------------------------------------------------------------------
def test_nested_children_or_inside_and():
    cfg = _configs(_config(_node(operator="AND", children=[
        _node(operator="OR", matches=[
            _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
            _cpe_match("cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*"),
        ]),
        _node(operator="OR", matches=[
            _cpe_match("cpe:2.3:o:microsoft:windows:10:*:*:*:*:*:*:*"),
            _cpe_match("cpe:2.3:o:microsoft:windows:11:*:*:*:*:*:*:*"),
        ]),
    ])))
    assert determine_applicability(
        ["cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*", "cpe:2.3:o:microsoft:windows:11:*:*:*:*:*:*:*"],
        cfg,
    ).status == ApplicabilityStatus.CONFIRMED
    assert determine_applicability(
        ["cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*", "cpe:2.3:o:linux:linux_kernel:5.0:*:*:*:*:*:*:*"],
        cfg,
    ).status == ApplicabilityStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Negate and vulnerable:false semantics
# ---------------------------------------------------------------------------
def test_vulnerable_false_does_not_override_vulnerable_true_in_or():
    # OR: vulnerable:true range covering 2.17.0, vulnerable:false for 2.17.0.
    # The vulnerable:true match should still yield Confirmed.
    cfg = _configs(_config(_node(operator="OR", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
        _cpe_match("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", vulnerable=False),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED


def test_vulnerable_false_contextualizes_within_and():
    # AND: vulnerable:true range, vulnerable:false for 2.17.0.
    # 2.17.0 should be Not Applicable because the AND is broken by the negation.
    cfg = _configs(_config(_node(operator="AND", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
        _cpe_match("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", vulnerable=False),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_vulnerable_false_alone_marks_not_applicable():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", vulnerable=False),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_node_negate():
    cfg = _configs(_config(_node(operator="OR", negate=True, matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Multiple installed CPEs
# ---------------------------------------------------------------------------
def test_multiple_asset_cpes_string_delimited():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    cpes = "cpe:2.3:a:apache:log4j:1.2.0:*:*:*:*:*:*:*; cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
    assert determine_applicability(cpes, cfg).status == ApplicabilityStatus.CONFIRMED


def test_multiple_asset_cpes_list():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    cpes = [
        "cpe:2.3:a:apache:log4j:1.2.0:*:*:*:*:*:*:*",
        "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
    ]
    assert determine_applicability(cpes, cfg).status == ApplicabilityStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Missing / unknown inventory
# ---------------------------------------------------------------------------
def test_unknown_asset_cpe_returns_unknown():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    for value in ("unknown", "N/A", None, ""):
        assert determine_applicability(value, cfg).status == ApplicabilityStatus.UNKNOWN


def test_missing_version_returns_unknown_when_range_used():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0"),
    ])))
    # version "*" is not a concrete version
    assert determine_applicability("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Unsupported version schemes
# ---------------------------------------------------------------------------
def test_alphanumeric_version_returns_unknown():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14a:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN


def test_unsupported_version_scheme_in_range_returns_unknown():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:beta:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN


# ---------------------------------------------------------------------------
# False-positive prevention
# ---------------------------------------------------------------------------
def test_different_product_is_not_applicable():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    assert determine_applicability("cpe:2.3:a:microsoft:windows:10:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_different_vendor_same_product_is_not_applicable():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
    ])))
    assert determine_applicability("cpe:2.3:a:other:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_version_outside_range_is_not_applicable():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndExcluding="2.17.0"),
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
def test_backward_compatible_direct_node_container():
    """Old callers passed {"nodes": [...]} instead of {"configurations": [...]}."""
    cfg = {"nodes": [_node(matches=[_cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")])]}
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED


def test_traceable_evidence_includes_path_and_range():
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
    ])))
    result = determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg)
    assert result.status == ApplicabilityStatus.CONFIRMED
    assert len(result.evidence) >= 1
    ev = result.evidence[0]
    assert "path" in ev
    assert "versionRange" in ev
    assert ev["asset_cpe"] == "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"


def test_multiple_configurations_or_combined():
    cfg = _configs(
        _config(_node(matches=[_cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")])),
        _config(_node(matches=[_cpe_match("cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*")])),
    )
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED


def test_unknown_when_no_overlap_and_no_version():
    # Asset has a concrete product but the CVE config is unparseable.
    cfg = _configs(_config(_node(matches=[{"vulnerable": True, "criteria": "not-a-cpe"}])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN


def test_explicit_non_match_distinguished_from_unknown():
    """Same product, version explicitly outside range -> Not Applicable (not Unknown)."""
    cfg = _configs(_config(_node(matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", versionStartIncluding="2.0.0", versionEndIncluding="2.17.0"),
    ])))
    result = determine_applicability("cpe:2.3:a:apache:log4j:9.9.9:*:*:*:*:*:*:*", cfg)
    assert result.status == ApplicabilityStatus.NOT_APPLICABLE
    assert any(ev.get("reason") == "version_explicitly_outside_range" for ev in result.evidence)


# ---------------------------------------------------------------------------
# Three-valued Kleene logic: Unknown must not be silently dropped
# ---------------------------------------------------------------------------
def test_or_with_true_and_unknown_returns_true():
    cfg = _configs(_config(_node(operator="OR", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
        {"vulnerable": True, "criteria": "not-a-cpe", "matchCriteriaId": "mc-bad"},
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.CONFIRMED


def test_and_with_false_and_unknown_returns_false():
    cfg = _configs(_config(_node(operator="AND", matches=[
        _cpe_match("cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*"),
        {"vulnerable": True, "criteria": "not-a-cpe", "matchCriteriaId": "mc-bad"},
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.NOT_APPLICABLE


def test_and_with_true_and_unknown_returns_unknown():
    cfg = _configs(_config(_node(operator="AND", matches=[
        _cpe_match("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"),
        {"vulnerable": True, "criteria": "not-a-cpe", "matchCriteriaId": "mc-bad"},
    ])))
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN


def test_or_across_configs_with_false_and_unknown_returns_unknown():
    cfg = _configs(
        _config(_node(matches=[_cpe_match("cpe:2.3:a:apache:http_server:2.4.51:*:*:*:*:*:*:*")])),
        _config(_node(matches=[{"vulnerable": True, "criteria": "not-a-cpe", "matchCriteriaId": "mc-bad"}])),
    )
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == ApplicabilityStatus.UNKNOWN

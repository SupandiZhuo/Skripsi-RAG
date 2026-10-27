"""P0 CPE oracle cases. Synthetic expressions are unit cases, not NVD fixtures."""

import json
from pathlib import Path

import pytest

from asset_inventory import collect_asset_cpes, inventory_is_verified, is_valid_cpe
from cve_applicability import ApplicabilityStatus as Status
from cve_applicability import cpe_matches, determine_applicability, parse_cpe


APP = "cpe:2.3:a:example:app:1.2:*:*:*:*:*:*:*"
OS = "cpe:2.3:o:example:os:10:*:*:*:*:*:*:*"
OTHER = "cpe:2.3:a:other:app:1.2:*:*:*:*:*:*:*"


def leaf(cpe=APP, vulnerable=True, **bounds):
    return {"criteria": cpe, "vulnerable": vulnerable, **bounds}


def node(*matches, operator="OR", **extra):
    return {"operator": operator, "cpeMatch": list(matches), **extra}


def configurations(*nodes, **extra):
    return [{"nodes": list(nodes), **extra}]


@pytest.mark.parametrize("bad", [
    "cpe:2.3:a:example:app:1.2", APP + ":extra", APP.replace(":example:", "::"),
    APP.replace(":a:", ":x:"), APP.replace("example", "ex ample"),
    APP.replace("example", r"ex\qample"), APP.replace("example", "ex*ample"),
    APP[:-1] + "\\", APP.replace(":*:*:*:*:*:*:*", ":*:*:bad_language:*:*:*:*"),
    None, 7,
])
def test_rejects_malformed_formatted_strings(bad):
    assert parse_cpe(bad) is None
    assert not is_valid_cpe(bad)


def test_escaped_delimiters_and_literals_are_preserved():
    cpe = APP.replace("example", r"ex\:ample").replace("app", r"app\*\?\\")
    parsed = parse_cpe(cpe)
    assert parsed["vendor"] == r"ex\:ample"
    assert parsed["product"] == r"app\*\?\\"
    assert cpe_matches(cpe, cpe)
    assert not cpe_matches(APP, APP.replace("app", r"app\*"))


def test_all_eleven_attributes_and_directional_wildcards():
    parsed = parse_cpe(APP)
    assert set(parsed) == {"part", "vendor", "product", "version", "update", "edition",
                           "language", "sw_edition", "target_sw", "target_hw", "other"}
    assert cpe_matches(APP, APP.replace("example", "ex*"))
    assert not cpe_matches(APP.replace("example", "*"), APP)
    assert not cpe_matches(APP, APP.replace(":*:*:*", ":windows:*:*", 1))
    assert not cpe_matches(APP, APP.replace(":1.2:", ":-:"))


def test_native_list_wrapper_and_direct_container():
    cfg = configurations(node(leaf()))
    for value in (cfg, {"configurations": cfg}, cfg[0]):
        assert determine_applicability(APP, value).status == Status.CONFIRMED


def test_supporting_platform_is_condition_not_vulnerable_target():
    cfg = configurations(node(leaf(), leaf(OS, False), operator="AND"))
    result = determine_applicability([APP, OS], cfg)
    assert result.status == Status.CONFIRMED
    support = next(e for e in result.evidence if e.get("criteria") == OS and e.get("matched"))
    assert support["condition_match"] is True
    assert support["is_vulnerable_component"] is False
    assert determine_applicability(APP, cfg).status == Status.UNKNOWN
    assert determine_applicability(APP, cfg, inventory_verified=True).status == Status.NOT_APPLICABLE


def test_supporting_only_or_branch_cannot_confirm_unmatched_target():
    cfg = configurations(node(leaf(OTHER), leaf(OS, False)))
    assert determine_applicability([APP, OS], cfg, inventory_verified=True).status == Status.UNKNOWN
    assert determine_applicability(OS, configurations(node(leaf(OS, False)))).status == Status.UNKNOWN


def test_vulnerable_witness_must_belong_to_satisfied_branch():
    blocked = node(operator="AND", children=[node(leaf()), node(leaf(OTHER, False))])
    cfg = configurations(node(operator="OR", children=[blocked, node(leaf(OS, False))]))
    assert determine_applicability([APP, OS], cfg, inventory_verified=True).status == Status.UNKNOWN


def test_config_level_or_and_negate():
    for op, expected in [("OR", Status.CONFIRMED), ("AND", Status.NOT_APPLICABLE)]:
        cfg = configurations(node(leaf()), node(leaf(OS, False)), operator=op)
        assert determine_applicability(APP, cfg, inventory_verified=True).status == expected
    cfg = configurations(node(leaf()), operator="OR", negate=True)
    assert determine_applicability(APP, cfg, inventory_verified=True).status == Status.NOT_APPLICABLE
    # A satisfied negation alone is not positive vulnerable-target evidence.
    assert determine_applicability(OTHER, cfg, inventory_verified=True).status == Status.UNKNOWN


def test_negated_supporting_condition_can_constrain_positive_target():
    cfg = configurations(node(leaf()), node(leaf(OS, False), negate=True), operator="AND")
    assert determine_applicability(APP, cfg).status == Status.UNKNOWN
    assert determine_applicability(APP, cfg, inventory_verified=True).status == Status.CONFIRMED
    assert determine_applicability([APP, OS], cfg, inventory_verified=True).status == Status.NOT_APPLICABLE


@pytest.mark.parametrize("bad", [None, "bad", [None], [{"nodes": "bad"}],
                                      [{"nodes": []}], [{"nodes": [None]}],
                                      configurations(node(leaf()), node(leaf(OS, False))),
                                      configurations(node(leaf(), operator="XOR")),
                                      configurations(node(leaf(), negate="false")),
                                      configurations(node({"criteria": APP})),
                                      configurations(node(leaf(), cpeMatch="bad"))])
def test_unsupported_configuration_is_unknown(bad):
    assert determine_applicability(APP, bad, inventory_verified=True).status == Status.UNKNOWN


def test_missing_inventory_is_unknown_and_unverified_nonmatches_are_unknown():
    cfg = configurations(node(leaf()))
    for inventory in (None, [], "unknown", "", OTHER):
        assert determine_applicability(inventory, cfg).status == Status.UNKNOWN
    assert determine_applicability(OTHER, cfg, inventory_verified=True).status == Status.NOT_APPLICABLE
    assert determine_applicability(OTHER, cfg, inventory_verified="true").status == Status.UNKNOWN
    for mixed_inventory in ([OTHER, "unknown"], [OTHER, None], OTHER + ";unknown"):
        assert determine_applicability(mixed_inventory, cfg, inventory_verified=True).status == Status.UNKNOWN


def test_unknown_asset_attributes_do_not_prove_match():
    cfg = configurations(node(leaf()))
    assert determine_applicability(APP.replace("example", "*"), cfg).status == Status.UNKNOWN
    assert determine_applicability(APP.replace(":1.2:", ":*:"), cfg).status == Status.UNKNOWN
    unknown_identity = APP.replace("example", "*")
    assert determine_applicability(unknown_identity, configurations(node(leaf(unknown_identity)))).status == Status.UNKNOWN
    wildcard_version = APP.replace(":1.2:", ":1.*:")
    assert determine_applicability(wildcard_version, configurations(node(leaf(wildcard_version)))).status == Status.UNKNOWN


def test_alphanumeric_exact_match_needs_no_ordering_but_range_is_unknown():
    beta = APP.replace(":1.2:", ":1.2beta:")
    assert determine_applicability(beta, configurations(node(leaf(beta)))).status == Status.CONFIRMED
    assert determine_applicability(beta, configurations(node(leaf(APP)))).status == Status.UNKNOWN
    cfg = configurations(node(leaf(APP.replace(":1.2:", ":*:"), versionEndIncluding="2.0")))
    assert determine_applicability(beta, cfg).status == Status.UNKNOWN


@pytest.mark.parametrize("version", ["v1.2", "1.v2", "01.2", "1.2-rc1"])
def test_no_guessing_numeric_version_schemes(version):
    cfg = configurations(node(leaf(APP.replace(":1.2:", ":*:"), versionEndIncluding="2.0")))
    assert determine_applicability(APP.replace(":1.2:", f":{version}:"), cfg).status == Status.UNKNOWN


def test_exact_version_and_ranges_both_constrain_condition():
    cfg = configurations(node(leaf(versionStartIncluding="1.0", versionEndIncluding="3.0")))
    assert determine_applicability(APP.replace(":1.2:", ":2.0:"), cfg, inventory_verified=True).status == Status.NOT_APPLICABLE


@pytest.mark.parametrize("bounds", [
    {"versionStartIncluding": "1.0", "versionStartExcluding": "1.1"},
    {"versionStartIncluding": "3.0", "versionEndIncluding": "1.0"},
    {"versionEndIncluding": 2}, {"versionStartIncluding": "1.2", "versionEndExcluding": "1.2"},
])
def test_malformed_or_empty_range_is_unknown(bounds):
    cfg = configurations(node(leaf(APP.replace(":1.2:", ":*:"), **bounds)))
    assert determine_applicability(APP, cfg, inventory_verified=True).status == Status.UNKNOWN


def test_delimited_inventory_keeps_escaped_separator():
    escaped = APP.replace("example", r"ex\,ample\;inc")
    assert determine_applicability(escaped + ";" + OS, configurations(node(leaf(escaped)))).status == Status.CONFIRMED


def test_inventory_verification_requires_explicit_valid_complete_records():
    assert not inventory_is_verified({"cpe": APP})
    assert inventory_is_verified({"cpe": APP, "inventory_verified": True})
    assert inventory_is_verified({"cpe": APP, "inventory_verified": "true"})
    for invalid_flag in ("True", "false", "yes", 1, None):
        assert not inventory_is_verified({"cpe": APP, "inventory_verified": invalid_flag})
    assert not inventory_is_verified({"cpe": APP, "inventory_verified": True, "software_inventory_json": "bad"})
    assert not inventory_is_verified({"cpe": APP, "inventory_verified": True,
                                     "software_inventory_json": json.dumps([{"cpe": "bad"}])})
    assert collect_asset_cpes({"cpe": "cpe:2.3:a:example:app:1.2"}) == []


def test_genuine_nvd_native_configurations_snapshot():
    raw = json.loads((Path(__file__).parent / "fixtures" / "nvd-CVE-2021-44228.json").read_text())
    cve = raw["vulnerabilities"][0]["cve"]
    assert cve["id"] == "CVE-2021-44228"
    cfg = cve["configurations"]
    assert isinstance(cfg, list)
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*", cfg).status == Status.CONFIRMED
    assert determine_applicability("cpe:2.3:a:apache:log4j:2.17.1:*:*:*:*:*:*:*", cfg, inventory_verified=True).status == Status.NOT_APPLICABLE
    assert determine_applicability("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*", cfg).status == Status.UNKNOWN


def test_genuine_nvd_firmware_requires_nonvulnerable_hardware_condition():
    raw = json.loads((Path(__file__).parent / "fixtures" / "nvd-CVE-2021-44228.json").read_text())
    cfg = raw["vulnerabilities"][0]["cve"]["configurations"]
    firmware = "cpe:2.3:o:siemens:6bk1602-0aa12-0tp0_firmware:2.6.0:*:*:*:*:*:*:*"
    hardware = "cpe:2.3:h:siemens:6bk1602-0aa12-0tp0:-:*:*:*:*:*:*:*"
    assert determine_applicability([firmware, hardware], cfg).status == Status.CONFIRMED
    assert determine_applicability(firmware, cfg).status == Status.UNKNOWN
    assert determine_applicability(firmware, cfg, inventory_verified=True).status == Status.NOT_APPLICABLE
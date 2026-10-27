"""Conservative NVD applicability, not a complete CPE reference implementation.

Supports full CPE 2.3 formatted strings with escaping and eleven attributes;
CPE URI binding (cpe:/...) is unsupported. Version ordering supports canonical
dotted integers only, with zero-padding (1.2 == 1.2.0); identical concrete vendor
versions match without ordering. Unsupported ordering returns Unknown.

Conditions and positive vulnerable targets are separate through AND/OR trees.
Supporting vulnerable:false entries can satisfy conditions but cannot establish
a target. Negation never creates positive target evidence. Multi-node containers
without explicit operators are Unknown; single-node containers need no default.

Missing inventory is open-world. Absent products prove exclusion only when
inventory_verified is explicitly True. Neither a primary CPE nor a nonempty
software list automatically attests completeness. Evidence includes leaf
condition_match/is_vulnerable_component and tree combination outcomes.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class ApplicabilityStatus(str, Enum):
    CONFIRMED = "Confirmed"
    NOT_APPLICABLE = "Not Applicable"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class ApplicabilityResult:
    status: ApplicabilityStatus
    evidence: Sequence[Mapping[str, Any]]
    explanation: str


def _is_blank_cpe(cpe: Optional[str]) -> bool:
    return cpe is None or str(cpe).strip() == "" or str(cpe).lower() in {
        "unknown", "n/a", "none", "-"
    }


def _normalize_cpes(asset_cpes: Union[None, str, Iterable[str]]) -> List[str]:
    """Accept a single CPE string, a list, or a delimited string."""
    if asset_cpes is None:
        return []
    if isinstance(asset_cpes, str):
        if _is_blank_cpe(asset_cpes):
            return []
        parts, current, escaped = [], [], False
        for char in asset_cpes:
            if char in ",;\n" and not escaped:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
            escaped = char == "\\" and not escaped
        parts.append("".join(current))
        return [p.strip() for p in parts if p.strip()]
    if not isinstance(asset_cpes, Iterable) or isinstance(asset_cpes, Mapping):
        return ["invalid_inventory"]
    # Do not discard invalid/unknown members of an otherwise valid inventory:
    # doing so could turn a verified non-match into an unsupported exclusion.
    return [str(c).strip() for c in asset_cpes]


CPE_FIELDS = ("part", "vendor", "product", "version", "update", "edition",
              "language", "sw_edition", "target_sw", "target_hw", "other")
RANGE_FIELDS = ("versionStartIncluding", "versionStartExcluding",
                "versionEndIncluding", "versionEndExcluding")


def _component_tokens(value: str) -> Optional[List[Tuple[str, bool]]]:
    tokens = []
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\":
            i += 1
            if i == len(value) or value[i].isalnum() or value[i] in "_." or not (33 <= ord(value[i]) <= 126):
                return None
            tokens.append((value[i], True))
        elif char.isascii() and (char.isalnum() or char in "_.-*?"):
            tokens.append((char, False))
        else:
            return None
        i += 1
    return tokens or None


def _valid_component(value: str) -> bool:
    if value in ("*", "-"):
        return True
    tokens = _component_tokens(value)
    if not tokens:
        return False
    shape = "".join(char if not quoted and char in "*?" else "x" for char, quoted in tokens)
    return re.fullmatch(r"(?:\*|\?+)?x+(?:\*|\?+)?|\?+", shape) is not None


def parse_cpe(cpe_uri: str) -> Optional[Dict[str, str]]:
    """Validate CPE 2.3 formatted strings and retain all escaped attributes.

    Literal punctuation, ANY (*), and NA (-) remain distinct. URI binding
    (cpe:/...) is intentionally unsupported.
    """
    if not isinstance(cpe_uri, str) or not cpe_uri.startswith("cpe:2.3:"):
        return None
    parts, current, escaped = [], [], False
    for char in cpe_uri[8:]:
        if char == ":" and not escaped:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
    parts.append("".join(current))
    if len(parts) != 11 or any(not _valid_component(part) for part in parts):
        return None
    if parts[0] not in ("a", "h", "o", "*", "-"):
        return None
    if parts[6] not in ("*", "-") and re.fullmatch(r"[A-Za-z]{2,3}(?:-(?:[A-Za-z]{2}|[0-9]{3}))?", parts[6]) is None:
        return None
    return dict(zip(CPE_FIELDS, parts))


def _parse_version(version: str) -> Optional[List[int]]:
    """Parse a dotted-numeric version. Return None for unsupported schemes."""
    if not isinstance(version, str) or len(version) > 256 or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*", version) is None:
        return None
    parts = version.split(".")
    numeric = []
    for part in parts:
        numeric.append(int(part))
    return numeric


def _compare_versions(v1: str, v2: str) -> Optional[int]:
    """Return -1, 0, 1, or None if the versions cannot be reliably compared."""
    p1 = _parse_version(v1)
    p2 = _parse_version(v2)
    if p1 is None or p2 is None:
        return None
    n = max(len(p1), len(p2))
    p1 = p1 + [0] * (n - len(p1))
    p2 = p2 + [0] * (n - len(p2))
    for a, b in zip(p1, p2):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def _version_in_range(version: str, match: Mapping[str, Any]) -> Optional[bool]:
    """Check whether `version` satisfies the range qualifiers in `match`.

    Returns True/False or None when a boundary cannot be evaluated.
    """
    if ("versionStartIncluding" in match and "versionStartExcluding" in match) or (
        "versionEndIncluding" in match and "versionEndExcluding" in match
    ):
        return None
    start = next((key for key in RANGE_FIELDS[:2] if key in match), None)
    end = next((key for key in RANGE_FIELDS[2:] if key in match), None)
    if start and end:
        ordering = _compare_versions(match[start], match[end])
        if ordering is None or ordering > 0 or (ordering == 0 and (start.endswith("Excluding") or end.endswith("Excluding"))):
            return None
    values = []
    for key in RANGE_FIELDS:
        if key not in match:
            continue
        ordering = _compare_versions(version, match[key])
        if ordering is None:
            return None
        values.append({"versionStartIncluding": ordering >= 0,
                       "versionStartExcluding": ordering > 0,
                       "versionEndIncluding": ordering <= 0,
                       "versionEndExcluding": ordering < 0}[key])
    return all(values) if values else None


def _component_match(asset_value: str, criteria_value: str) -> Optional[bool]:
    if criteria_value == "*":
        return True
    if asset_value == "*":
        return None
    if criteria_value == "-" or asset_value == "-":
        return criteria_value == asset_value
    asset_tokens = _component_tokens(asset_value)
    criteria_tokens = _component_tokens(criteria_value)
    if asset_tokens is None or criteria_tokens is None:
        return None
    if any(char in "*?" and not quoted for char, quoted in asset_tokens):
        return None
    pattern = "".join(".*" if char == "*" and not quoted else "." if char == "?" and not quoted else re.escape(char)
                      for char, quoted in criteria_tokens)
    literal = "".join(char for char, _ in asset_tokens)
    return re.fullmatch(pattern, literal, re.IGNORECASE) is not None


def _cpe_component_matches(asset_value: str, criteria_value: str) -> bool:
    return _component_match(asset_value, criteria_value) is True


def _build_evidence(
    path: str,
    asset_cpe: str,
    criteria: str,
    match: Mapping[str, Any],
    *,
    matched: Optional[bool],
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "path": path,
        "asset_cpe": asset_cpe,
        "criteria": criteria,
        "vulnerable": match.get("vulnerable"),
        "is_vulnerable_component": match.get("vulnerable") is True,
        "matchCriteriaId": match.get("matchCriteriaId"),
        "matched": matched,
        "condition_match": matched,
        "reason": reason,
    }
    ranges = {k: match[k] for k in (
        "versionStartIncluding", "versionStartExcluding",
        "versionEndIncluding", "versionEndExcluding",
    ) if k in match}
    if ranges:
        ev["versionRange"] = ranges
    return ev


# Three-valued logic helpers --------------------------------------------------
def _kleene_and(values: List[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _kleene_or(values: List[Optional[bool]]) -> Optional[bool]:
    if any(v is True for v in values):
        return True
    if any(v is None for v in values):
        return None
    return False


def _kleene_not(value: Optional[bool]) -> Optional[bool]:
    if value is True:
        return False
    if value is False:
        return True
    return None


def _kleene_combine(values: List[Optional[bool]], operator: str) -> Optional[bool]:
    if not values or not isinstance(operator, str):
        return None
    op = operator.upper()
    if op == "AND":
        return _kleene_and(values)
    if op == "OR":
        return _kleene_or(values)
    return None


# Match evaluation ------------------------------------------------------------
def _evaluate_match_for_asset(
    asset_cpe: str,
    match: Mapping[str, Any],
    path: str,
) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """Evaluate one cpeMatch entry against one asset CPE.

    Returns (result, evidence). result is True/False/Unknown.
    """
    criteria_cpe = match.get("criteria", "")
    asset = parse_cpe(asset_cpe)
    criteria = parse_cpe(criteria_cpe)

    if asset is None:
        return (
            None,
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=None, reason="asset_cpe_unparseable"),
        )
    if criteria is None:
        return (
            None,
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=None, reason="criteria_unparseable"),
        )

    if any(asset[key] in ("*", "-") or any(
        char in "*?" and not quoted for char, quoted in _component_tokens(asset[key]) or []
    ) for key in ("part", "vendor", "product")):
        return None, _build_evidence(path, asset_cpe, criteria_cpe, match, matched=None, reason="asset_identity_incomplete")

    attributes = [_component_match(asset[key], criteria[key]) for key in CPE_FIELDS if key != "version"]
    attribute_result = _kleene_and(attributes)
    if attribute_result is False:
        return False, _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="attribute_mismatch")

    version = asset.get("version", "*")
    crit_version = criteria.get("version", "*")
    has_range = any(k in match for k in (
        "versionStartIncluding", "versionStartExcluding",
        "versionEndIncluding", "versionEndExcluding",
    ))

    # Exact identifiers do not require vendor-specific version ordering. For
    # differing numeric representations preserve the existing dotted comparator.
    if crit_version == "*":
        exact_result = True
    elif version == "*":
        exact_result = None
    elif any(char in "*?" and not quoted for char, quoted in _component_tokens(version) or []):
        exact_result = None
    elif crit_version == "-" or version == "-":
        exact_result = version == crit_version
    elif asset["version"].lower() == criteria["version"].lower():
        exact_result = True
    elif any(char in "*?" and not quoted for char, quoted in _component_tokens(crit_version) or []):
        exact_result = _component_match(version, crit_version)
    else:
        cmp = _compare_versions(version, crit_version)
        exact_result = None if cmp is None else cmp == 0

    if has_range:
        range_result = _version_in_range(version, match)
        if range_result is None:
            return (
                None,
                _build_evidence(path, asset_cpe, criteria_cpe, match, matched=None, reason="version_range_unsupported"),
            )
        version_match = _kleene_and([exact_result, range_result])
    else:
        version_match = exact_result

    condition = _kleene_and([attribute_result, version_match])
    if condition is True:
        return (
            True,
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=True),
        )
    if condition is None:
        return None, _build_evidence(path, asset_cpe, criteria_cpe, match, matched=None, reason="attribute_or_version_unknown")

    # Explicit non-match: same product, version outside affected set.
    return (
        False,
        _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="version_explicitly_outside_range"),
    )


def _evaluate_match_against_assets(
    match: Mapping[str, Any],
    asset_cpes: Sequence[str],
    path: str,
    inventory_verified: bool = False,
) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    """Evaluate one cpeMatch entry against all installed asset CPEs (OR).

    Unknown (None) results from any asset are preserved: OR(False, Unknown)
    is Unknown.
    """
    has_true = False
    has_unknown = False
    evidence: List[Dict[str, Any]] = []
    if not isinstance(match, Mapping) or type(match.get("vulnerable")) is not bool:
        return None, [{"path": path, "condition_match": None, "reason": "invalid_match_metadata"}]
    for asset_cpe in asset_cpes:
        res, ev = _evaluate_match_for_asset(asset_cpe, match, path)
        if ev is not None:
            evidence.append(ev)
        if res is True:
            has_true = True
        elif res is None:
            has_unknown = True
        # res is False -> condition unsatisfied for this asset, no flag needed
    if has_true:
        return True, evidence
    if has_unknown:
        return None, evidence
    if not inventory_verified:
        evidence.append({"path": path, "criteria": match.get("criteria"), "matched": None,
                         "condition_match": None, "reason": "inventory_not_verified"})
        return None, evidence
    return False, evidence


# Node / configuration evaluation ---------------------------------------------
@dataclass
class _Condition:
    matched: Optional[bool]
    affected: Optional[bool]
    evidence: List[Dict[str, Any]]


def _unknown_condition(path: str, reason: str) -> _Condition:
    return _Condition(None, None, [{"path": path, "condition_match": None, "reason": reason}])


def _combine_conditions(results: List[_Condition], operator: Any, negate: Any, path: str) -> _Condition:
    evidence = [entry for result in results for entry in result.evidence]
    # No arbitrary AND/OR default for ambiguous containers. Native single-node
    # configurations work without an operator because there is no combination.
    if operator is None and len(results) == 1:
        operator = "OR"
    if not results or operator not in ("AND", "OR") or type(negate) is not bool:
        unknown = _unknown_condition(path, "unsupported_or_ambiguous_configuration")
        unknown.evidence = evidence + unknown.evidence
        return unknown
    matched = _kleene_combine([result.matched for result in results], operator)
    if operator == "AND":
        affected = _kleene_and([matched, _kleene_or([result.affected for result in results])])
    else:
        # Child witnesses already include their own AND guards. A vulnerable
        # leaf in an unsatisfied OR branch must never leak into a supporting one.
        affected = _kleene_or([result.affected for result in results])
    if negate:
        matched = _kleene_not(matched)
        affected = False  # Negation cannot manufacture a positive target.
    evidence.append({"path": path, "operator": operator, "negate": negate,
                     "condition_match": matched, "vulnerable_target_match": affected,
                     "reason": "combined_configuration"})
    return _Condition(matched, affected, evidence)


def _evaluate_node(
    node: Mapping[str, Any],
    asset_cpes: Sequence[str],
    path: str,
    inventory_verified: bool = False,
    depth: int = 0,
) -> _Condition:
    if not isinstance(node, Mapping) or depth > 32:
        return _unknown_condition(path, "invalid_node_or_depth_limit")
    matches, children = node.get("cpeMatch", []), node.get("children", [])
    if not isinstance(matches, list) or not isinstance(children, list) or "nodes" in node:
        return _unknown_condition(path, "invalid_node_children")
    results = []
    for idx, match in enumerate(matches):
        subpath = f"{path}.cpeMatch[{idx}]"
        condition, evidence = _evaluate_match_against_assets(match, asset_cpes, subpath, inventory_verified)
        vulnerable = match.get("vulnerable") if isinstance(match, Mapping) else None
        affected = condition if vulnerable is True else False if vulnerable is False else None
        results.append(_Condition(condition, affected, evidence))
    for idx, child in enumerate(children):
        results.append(_evaluate_node(child, asset_cpes, f"{path}.children[{idx}]", inventory_verified, depth + 1))
    return _combine_conditions(results, node.get("operator"), node.get("negate", False), path)


def _normalize_configs(nvd_configurations: Any) -> List[Any]:
    """Native NVD arrays, wrapped arrays, or legacy direct node containers.

    Preserve malformed entries as Unknown branches instead of dropping them.
    """
    if isinstance(nvd_configurations, list):
        return nvd_configurations
    if isinstance(nvd_configurations, Mapping):
        if "configurations" in nvd_configurations:
            value = nvd_configurations["configurations"]
            return value if isinstance(value, list) else [None]
        if "nodes" in nvd_configurations:
            return [nvd_configurations]
    return []


def determine_applicability(
    asset_cpes: Union[None, str, Iterable[str]],
    nvd_configurations: Any,
    inventory_verified: bool = False,
) -> ApplicabilityResult:
    """Determine CVE applicability to an asset from NVD CPE data.

    Args:
        asset_cpes: Asset CPE string(s). A single string may contain multiple
            CPEs separated by comma, semicolon, or newline.
        nvd_configurations: Native NVD configurations list, wrapped list, or
            legacy direct configuration container.
        inventory_verified: Explicit boolean attestation that installed-product
            inventory is complete. Defaults False: absent products remain Unknown.

    Returns:
        ApplicabilityResult with status, traceable evidence, and explanation.
    """
    cpe_list = _normalize_cpes(asset_cpes)
    if not cpe_list:
        return ApplicabilityResult(
            status=ApplicabilityStatus.UNKNOWN,
            evidence=[],
            explanation="Asset CPE inventory is missing, empty, or marked unknown; applicability cannot be determined.",
        )

    configs = _normalize_configs(nvd_configurations)
    if not configs:
        return ApplicabilityResult(
            status=ApplicabilityStatus.UNKNOWN,
            evidence=[],
            explanation="NVD has no CPE configuration data for this CVE.",
        )

    config_results: List[_Condition] = []
    all_evidence: List[Dict[str, Any]] = []

    for cfg_idx, config in enumerate(configs):
        cfg_path = f"configurations[{cfg_idx}]"
        if not isinstance(config, Mapping) or not isinstance(config.get("nodes"), list):
            result = _unknown_condition(cfg_path, "invalid_configuration")
        else:
            nodes = [_evaluate_node(node, cpe_list, f"{cfg_path}.nodes[{idx}]", inventory_verified is True)
                     for idx, node in enumerate(config["nodes"])]
            result = _combine_conditions(nodes, config.get("operator"), config.get("negate", False), cfg_path)
        config_results.append(result)
        all_evidence.extend(result.evidence)

    overall = _kleene_or([result.matched for result in config_results])
    affected = _kleene_or([result.affected for result in config_results])

    if overall is True and affected is True:
        return ApplicabilityResult(
            status=ApplicabilityStatus.CONFIRMED,
            evidence=all_evidence,
            explanation="Asset CPE(s) match vulnerable configuration criteria in NVD.",
        )
    if overall is False and inventory_verified is True:
        return ApplicabilityResult(
            status=ApplicabilityStatus.NOT_APPLICABLE,
            evidence=all_evidence,
            explanation="Verified complete asset inventory is outside all evaluated NVD configurations.",
        )

    return ApplicabilityResult(
        status=ApplicabilityStatus.UNKNOWN,
        evidence=all_evidence,
        explanation="Applicability is unresolved: inventory, configuration semantics, or positive vulnerable-target evidence is insufficient.",
    )


# Convenience predicate kept for callers that only need a boolean-like answer.
def cpe_matches(asset_cpe: str, criteria_cpe: str) -> bool:
    """Return True when `asset_cpe` matches `criteria_cpe` (CPE 2.3).

    This is a strict exact/wildcard match and does not consider version ranges.
    """
    asset = parse_cpe(asset_cpe)
    criteria = parse_cpe(criteria_cpe)
    if asset is None or criteria is None:
        return False
    for key in CPE_FIELDS:
        if not _cpe_component_matches(asset.get(key, "*"), criteria.get(key, "*")):
            return False
    return True

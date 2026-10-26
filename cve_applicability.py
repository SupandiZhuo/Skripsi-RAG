"""Hardened CVE applicability matching using NVD CPE configurations (Phase A.1).

Determines whether a CVE is applicable to an organizational asset by evaluating
NVD CPE configurations with three-valued logic (True / False / Unknown) while
preserving AND/OR/negate structure and version-range semantics.

Result states (unchanged from Phase A):
- Confirmed      — the asset is in the set of affected configurations.
- Not Applicable — the asset is explicitly outside the affected set.
- Unknown        — insufficient or unparseable information to decide.

Design decisions:
- Version-range qualifiers (versionStartIncluding, versionStartExcluding,
  versionEndIncluding, versionEndExcluding) are evaluated against the asset's
  declared CPE version using a dotted-numeric comparator. Non-numeric or
  otherwise unparseable versions fall back to Unknown.
- Each `cpeMatch` is evaluated as a three-valued condition. A match that is
  `vulnerable: true` yields True; a match that is `vulnerable: false` yields
  False. An asset whose vendor/product overlap with the criteria but whose
  version is explicitly outside the criteria's affected range also yields False.
- Nodes are combined according to their `operator` (AND / OR) and `negate`
  flag using three-valued Kleene logic.
- Multiple installed CPEs are supported: pass a list or a comma/semicolon/newline
  separated string in the asset CPE field.
- `vulnerable: false` is treated as a contextual condition, not as automatic
  proof that the asset is unaffected. It participates in the same Boolean
  structure as `vulnerable: true` matches.
- Traceable evidence records the configuration path, criteria, range
  conditions, the matched asset CPE, and the reason for the condition value.

Limitations (documented):
- CPE 2.3 URI parsing only; CPE 2.2 is treated as opaque and cannot be matched.
- Dotted-numeric version comparison handles common schemes (1.2.3). Mixed
  alphanumeric versions (e.g., "2.14a", "stable") return Unknown.
- NVD configurations with deeply unusual logic may still yield Unknown when the
  evaluator cannot decide reliably.
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
        parts = re.split(r"[,;\n]+", asset_cpes)
        return [p.strip() for p in parts if p.strip() and not _is_blank_cpe(p.strip())]
    return [
        str(c).strip()
        for c in asset_cpes
        if str(c).strip() and not _is_blank_cpe(str(c).strip())
    ]


def parse_cpe(cpe_uri: str) -> Optional[Dict[str, str]]:
    """Parse a CPE 2.3 URI into its component parts.

    Format: cpe:2.3:<part>:<vendor>:<product>:<version>:<update>:<edition>:...
    """
    if not cpe_uri or not cpe_uri.startswith("cpe:2.3:"):
        return None
    parts = cpe_uri.split(":")
    if len(parts) < 6:
        return None
    return {
        "part": parts[2],
        "vendor": parts[3],
        "product": parts[4],
        "version": parts[5] if len(parts) > 5 else "*",
        "update": parts[6] if len(parts) > 6 else "*",
        "edition": parts[7] if len(parts) > 7 else "*",
    }


def _parse_version(version: str) -> Optional[List[int]]:
    """Parse a dotted-numeric version. Return None for unsupported schemes."""
    if not version or version == "*" or version == "-":
        return None
    parts = version.split(".")
    numeric = []
    for part in parts:
        part = part.lstrip("vV")
        if not re.fullmatch(r"\d+", part):
            return None
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
    lower: Optional[bool] = None
    if "versionStartIncluding" in match:
        cmp = _compare_versions(version, match["versionStartIncluding"])
        if cmp is None:
            return None
        lower = cmp >= 0
    elif "versionStartExcluding" in match:
        cmp = _compare_versions(version, match["versionStartExcluding"])
        if cmp is None:
            return None
        lower = cmp > 0

    upper: Optional[bool] = None
    if "versionEndIncluding" in match:
        cmp = _compare_versions(version, match["versionEndIncluding"])
        if cmp is None:
            return None
        upper = cmp <= 0
    elif "versionEndExcluding" in match:
        cmp = _compare_versions(version, match["versionEndExcluding"])
        if cmp is None:
            return None
        upper = cmp < 0

    if lower is False or upper is False:
        return False
    if lower is None and upper is None:
        return None
    if lower is None:
        return upper
    if upper is None:
        return lower
    return lower and upper


def _cpe_component_matches(asset_value: str, criteria_value: str) -> bool:
    crit = criteria_value.strip()
    if crit == "*" or crit == "-":
        return True
    return crit.lower() == asset_value.strip().lower()


def _build_evidence(
    path: str,
    asset_cpe: str,
    criteria: str,
    match: Mapping[str, Any],
    *,
    matched: bool,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    ev: Dict[str, Any] = {
        "path": path,
        "asset_cpe": asset_cpe,
        "criteria": criteria,
        "vulnerable": bool(match.get("vulnerable", True)),
        "matchCriteriaId": match.get("matchCriteriaId"),
        "matched": matched,
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
    op = (operator or "OR").upper()
    if op == "AND":
        return _kleene_and(values)
    return _kleene_or(values)


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
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="asset_cpe_unparseable"),
        )
    if criteria is None:
        return (
            None,
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="criteria_unparseable"),
        )

    # Vendor/product relevance check.
    for key in ("part", "vendor", "product", "update", "edition"):
        if not _cpe_component_matches(asset.get(key, "*"), criteria.get(key, "*")):
            # Different product family: this condition is False for this asset.
            return False, None

    version = asset.get("version", "*")
    crit_version = criteria.get("version", "*")
    has_range = any(k in match for k in (
        "versionStartIncluding", "versionStartExcluding",
        "versionEndIncluding", "versionEndExcluding",
    ))

    if has_range:
        range_result = _version_in_range(version, match)
        if range_result is None:
            return (
                None,
                _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="version_range_unsupported"),
            )
        version_match = range_result
    elif crit_version == "*" or crit_version == "-":
        version_match = True
    else:
        cmp = _compare_versions(version, crit_version)
        if cmp is None:
            return (
                None,
                _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="version_compare_unsupported"),
            )
        version_match = cmp == 0

    if version_match:
        vulnerable = bool(match.get("vulnerable", True))
        return (
            vulnerable,
            _build_evidence(path, asset_cpe, criteria_cpe, match, matched=True),
        )

    # Explicit non-match: same product, version outside affected set.
    return (
        False,
        _build_evidence(path, asset_cpe, criteria_cpe, match, matched=False, reason="version_explicitly_outside_range"),
    )


def _evaluate_match_against_assets(
    match: Mapping[str, Any],
    asset_cpes: Sequence[str],
    path: str,
) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    """Evaluate one cpeMatch entry against all installed asset CPEs (OR).

    Unknown (None) results from any asset are preserved: OR(False, Unknown)
    is Unknown.
    """
    has_true = False
    has_unknown = False
    evidence: List[Dict[str, Any]] = []
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
    # No asset matched and none were indeterminate -> condition is False.
    return False, evidence


# Node / configuration evaluation ---------------------------------------------
def _evaluate_node(
    node: Mapping[str, Any],
    asset_cpes: Sequence[str],
    path: str,
) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    """Evaluate an NVD configuration node recursively.

    Returns (result, evidence).
    """
    results: List[Optional[bool]] = []
    evidence: List[Dict[str, Any]] = []

    for idx, match in enumerate(node.get("cpeMatch", [])):
        subpath = f"{path}.cpeMatch[{idx}]"
        res, ev = _evaluate_match_against_assets(match, asset_cpes, subpath)
        evidence.extend(ev)
        results.append(res)

    for idx, child in enumerate(node.get("children", [])):
        subpath = f"{path}.children[{idx}]"
        res, ev = _evaluate_node(child, asset_cpes, subpath)
        evidence.extend(ev)
        results.append(res)

    if not results:
        return None, evidence

    combined = _kleene_combine(results, node.get("operator", "OR"))
    if node.get("negate"):
        combined = _kleene_not(combined)
    return combined, evidence


def _normalize_configs(nvd_configurations: Optional[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if not nvd_configurations:
        return []
    if "configurations" in nvd_configurations:
        configs = nvd_configurations["configurations"]
        return list(configs) if isinstance(configs, Sequence) and not isinstance(configs, str) else []
    # Backward compatibility: a raw node container passed directly.
    return [{"nodes": nvd_configurations.get("nodes", [])}]


def determine_applicability(
    asset_cpes: Union[None, str, Iterable[str]],
    nvd_configurations: Optional[Mapping[str, Any]],
) -> ApplicabilityResult:
    """Determine CVE applicability to an asset from NVD CPE data.

    Args:
        asset_cpes: Asset CPE string(s). A single string may contain multiple
            CPEs separated by comma, semicolon, or newline.
        nvd_configurations: NVD CVE `configurations` object.

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

    config_results: List[Optional[bool]] = []
    all_evidence: List[Dict[str, Any]] = []

    for cfg_idx, config in enumerate(configs):
        cfg_path = f"configurations[{cfg_idx}]"
        nodes = config.get("nodes", [])
        node_results: List[Optional[bool]] = []
        node_evidence: List[Dict[str, Any]] = []

        for node_idx, node in enumerate(nodes):
            node_path = f"{cfg_path}.nodes[{node_idx}]"
            res, ev = _evaluate_node(node, cpe_list, node_path)
            node_evidence.extend(ev)
            node_results.append(res)

        all_evidence.extend(node_evidence)
        if node_results:
            # Multiple nodes within one configuration are ANDed.
            # Unknown (None) results participate in Kleene logic so that
            # e.g. True AND Unknown -> Unknown, not True.
            config_results.append(_kleene_and(node_results))

    if not config_results:
        # No node produced a determinable result (e.g., all unparseable).
        has_evidence = any(
            e.get("reason") in (
                "asset_cpe_unparseable", "criteria_unparseable",
                "version_range_unsupported", "version_compare_unsupported",
            )
            for e in all_evidence
        )
        if has_evidence:
            return ApplicabilityResult(
                status=ApplicabilityStatus.UNKNOWN,
                evidence=all_evidence,
                explanation="Asset CPE(s) overlap the CVE configuration but could not be evaluated reliably.",
            )
        return ApplicabilityResult(
            status=ApplicabilityStatus.NOT_APPLICABLE,
            evidence=all_evidence,
            explanation="No NVD CPE criteria matched the asset's declared product family; the CVE is not applicable.",
        )

    overall = _kleene_or(config_results)

    if overall is True:
        return ApplicabilityResult(
            status=ApplicabilityStatus.CONFIRMED,
            evidence=all_evidence,
            explanation="Asset CPE(s) match vulnerable configuration criteria in NVD.",
        )
    if overall is False:
        return ApplicabilityResult(
            status=ApplicabilityStatus.NOT_APPLICABLE,
            evidence=all_evidence,
            explanation="Asset CPE(s) are explicitly outside the affected configuration set or match only non-vulnerable/negated criteria.",
        )

    return ApplicabilityResult(
        status=ApplicabilityStatus.UNKNOWN,
        evidence=all_evidence,
        explanation="Applicability could not be determined from the available NVD CPE data.",
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
    for key in ("part", "vendor", "product", "version", "update", "edition"):
        if not _cpe_component_matches(asset.get(key, "*"), criteria.get(key, "*")):
            return False
    return True

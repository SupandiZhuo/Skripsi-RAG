"""Deterministic organizational remediation priority (Phase C).

Computes Organizational Priority (P1–P4) for a vulnerability affecting a
specific asset. The module is intentionally separate from CVSS scoring:
CVSS severity measures the technical characteristics of the vulnerability,
while this rubric measures how urgently the organization should remediate it
on this asset.

Active rubric: seven factors totaling exactly 100 points, defined by
FACTOR_MAX_POINTS. Thresholds: P1 >= 80, P2 >= 60, P3 >= 40, else P4.
Only Confirmed applicability can produce a remediation priority.
Not Applicable produces priority=None and status="not_applicable".
Unknown produces priority=None, status="needs_verification", and a separate
investigation_priority when business-criticality evidence supports triage.
Missing/invalid applicability never authorizes remediation classification.

Graph dependencies are inactive: the API argument is retained for caller
compatibility but has no effect, even when relationship data is supplied.
Optional unknown factors contribute zero and remain explicitly disclosed.
Weights and thresholds are researcher-defined, not expert-approved or
empirically validated. See docs/PRIORITY_RUBRIC.md for the active policy.
Identical inputs produce identical frozen-dataclass results.
"""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cve_applicability import ApplicabilityStatus

RUBRIC_VERSION = "organizational-priority-v2.0"

FACTOR_MAX_POINTS: Dict[str, int] = {
    "vulnerability severity": 30,
    "business criticality": 25,
    "internet exposure": 15,
    "data sensitivity": 8,
    "environment": 5,
    "CIA baseline": 2,
    "CPE applicability": 15,
}
MAX_SCORE = sum(FACTOR_MAX_POINTS.values())
assert MAX_SCORE == 100, "Review active factor weights before running experiments"

# Thresholds for P1..P4.
SCORE_THRESHOLDS: Sequence[Tuple[int, str, str]] = (
    (80, "P1", "Urgent"),
    (60, "P2", "High"),
    (40, "P3", "Normal"),
)

# Factor point tables.
_SEVERITY_POINTS: Dict[str, int] = {
    "None": 0,
    "Critical": 30,
    "High": 23,
    "Medium": 12,
    "Low": 3,
}
_CRITICALITY_POINTS: Dict[str, int] = {
    "Mission Critical": 25,
    "High": 18,
    "Medium": 10,
    "Low": 3,
}
_SENSITIVITY_POINTS: Dict[str, int] = {
    "Restricted": 8,
    "Confidential": 5,
    "Internal": 2,
    "Public": 0,
}

_STATUS_COMPLETE = "complete"
_STATUS_INCOMPLETE = "incomplete"
_STATUS_NOT_APPLICABLE = "not_applicable"
_STATUS_NEEDS_VERIFICATION = "needs_verification"


class PriorityLevel(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


@dataclass(frozen=True)
class PriorityRuleContribution:
    """One rubric factor's contribution to the priority score."""

    rule: str
    factor: str
    points: int
    max_points: int
    evidence_value: str
    evidence_source: str


@dataclass(frozen=True)
class OrganizationalPriorityResult:
    """Structured, reproducible result of the priority rubric."""

    priority: Optional[str]  # P1..P4 or None
    status: str  # complete | incomplete | not_applicable | needs_verification
    score: Optional[int]
    max_score: int
    level: Optional[str]  # Urgent / High / Normal / Low
    rule_contributions: Tuple[PriorityRuleContribution, ...]
    missing_evidence: Tuple[str, ...]
    unknown_factors: Tuple[str, ...]
    explanation: str
    cve_id: Optional[str] = None
    asset_id: Optional[str] = None
    investigation_priority: Optional[str] = None  # triage only, never remediation
    rubric_version: str = RUBRIC_VERSION


def _severity_band(score: Optional[float]) -> Optional[str]:
    if score is None or not isfinite(score) or not 0.0 <= score <= 10.0:
        return None
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    if score > 0.0:
        return "Low"
    return "None"


def _level_for_score(score: int) -> Tuple[str, str]:
    for threshold, priority, level in SCORE_THRESHOLDS:
        if score >= threshold:
            return priority, level
    return "P4", "Low"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _build_explanation(
    priority: Optional[str],
    level: Optional[str],
    score: Optional[int],
    contributions: Sequence[PriorityRuleContribution],
    missing: Sequence[str],
    unknowns: Sequence[str],
    status: str,
) -> str:
    if status == _STATUS_NOT_APPLICABLE:
        lines = [
            "Not Applicable — exclude this CVE/asset pair from the remediation queue; "
            "no remediation priority is assigned.",
        ]
    elif status == _STATUS_NEEDS_VERIFICATION:
        if priority is None:
            lines = [
                "Applicability is Unknown: verify whether the CVE affects the asset. "
                "An investigation priority cannot be determined because required evidence is missing: "
                + ", ".join(missing) + ". This is not a remediation decision."
            ]
        else:
            lines = [
                f"Investigation priority {priority} ({level}) — provisional triage score "
                f"{score}/{MAX_SCORE}. Applicability is Unknown: verify the affected "
                "product/version before assigning remediation. This is not a remediation priority."
            ]
        if contributions:
            lines.append("Available evidence: " + "; ".join(
                f"{c.factor}={c.evidence_value} ({c.points}/{c.max_points})"
                for c in contributions
            ) + ".")
    elif status == _STATUS_INCOMPLETE:
        lines = [
            "Priority cannot be determined confidently because required evidence is missing: "
            + ", ".join(missing) + "."
        ]
        if contributions:
            lines.append(
                "Available evidence: "
                + "; ".join(
                    f"{c.factor}={c.evidence_value} ({c.points}/{c.max_points})"
                    for c in contributions
                )
                + "."
            )
        return " ".join(lines)
    else:
        lines = [
            f"{priority} ({level}) — organizational priority score {score}/{MAX_SCORE}. "
            "Contributing factors: "
            + "; ".join(
                f"{c.factor}={c.evidence_value} ({c.points}/{c.max_points})"
                for c in contributions
            )
            + ".",
        ]

    if unknowns:
        lines.append(
            "Unknown evidence (optional factors counted as 0, not low risk): "
            + ", ".join(unknowns) + "."
        )
    return " ".join(lines)


def calculate_organizational_priority(
    asset_metadata: Mapping[str, Any],
    *,
    base_score: Optional[float] = None,
    context_score: Optional[float] = None,
    applicability: Optional[Any] = None,
    graph_dependencies: Optional[Mapping[str, Any]] = None,
    cve_id: Optional[str] = None,
) -> OrganizationalPriorityResult:
    """Compute deterministic Organizational Priority (P1–P4).

    Args:
        asset_metadata: Asset Schema v2 metadata dict (from RAG or ChromaDB).
        base_score: Optional NVD CVSS base score.
        context_score: Optional contextual/environmental CVSS score.
        applicability: Optional ApplicabilityResult or dict-like object with a
            `.status` or `['status']` attribute/value.
        graph_dependencies: Compatibility-only argument; ignored by the active
            rubric. No graph points or graph missing-evidence penalty is used.
        cve_id: Optional CVE identifier for traceability.

    Returns:
        OrganizationalPriorityResult with priority, status, score,
        contributions, missing/unknown evidence, and explanation. Unknown
        applicability uses investigation_priority, never remediation priority.
    """
    ctx: Mapping[str, Any] = asset_metadata if isinstance(asset_metadata, Mapping) else {}
    asset_id = _text(ctx.get("asset_id")) or None

    contributions: List[PriorityRuleContribution] = []
    missing: List[str] = []
    unknowns: List[str] = []
    score = 0

    # --- CPE applicability (evaluated first because it can exclude or block) ---
    app_status: Optional[Any] = None
    app_evidence_source = "cve_applicability"
    if hasattr(applicability, "status"):
        app_status = applicability.status
    elif isinstance(applicability, Mapping):
        app_status = applicability.get("status")

    # Normalize to enum when possible, so str/Enum handling works across
    # Python versions.
    if isinstance(app_status, str):
        try:
            app_status = ApplicabilityStatus(app_status)
        except ValueError:
            pass
    if app_status is None:
        app_status = ""

    if app_status == ApplicabilityStatus.NOT_APPLICABLE:
        # CVE does not affect this asset: exclude, do not label as low priority.
        return OrganizationalPriorityResult(
            priority=None,
            status=_STATUS_NOT_APPLICABLE,
            score=None,
            max_score=MAX_SCORE,
            level=None,
            rule_contributions=(),
            missing_evidence=(),
            unknown_factors=(),
            explanation=_build_explanation(
                None, None, None, (), (), (), _STATUS_NOT_APPLICABLE,
            ),
            cve_id=cve_id,
            asset_id=asset_id,
        )

    # --- Vulnerability severity (prefer context_score, fall back to base_score) ---
    severity: Optional[str] = None
    severity_source = "nvd.base_score"
    severity_score = context_score
    if _severity_band(severity_score) is not None:
        severity_source = "contextual.cvss_environmental_score"
    else:
        severity_score = base_score

    severity = _severity_band(severity_score)

    if severity:
        points = _SEVERITY_POINTS[severity]
        score += points
        contributions.append(
            PriorityRuleContribution(
                rule=f"vulnerability_severity:{severity}",
                factor="vulnerability severity",
                points=points,
                max_points=FACTOR_MAX_POINTS["vulnerability severity"],
                evidence_value=f"{severity} ({severity_score:g})",
                evidence_source=severity_source,
            )
        )
    else:
        missing.append("vulnerability severity (base or contextual CVSS)")

    # --- Business criticality (required) ---
    criticality = _text(ctx.get("business_criticality"))
    if criticality in _CRITICALITY_POINTS:
        points = _CRITICALITY_POINTS[criticality]
        score += points
        contributions.append(
            PriorityRuleContribution(
                rule=f"business_criticality:{criticality}",
                factor="business criticality",
                points=points,
                max_points=FACTOR_MAX_POINTS["business criticality"],
                evidence_value=criticality,
                evidence_source="asset.business_criticality",
            )
        )
    else:
        missing.append("business criticality")

    # --- Internet exposure (required field, but unknown values are handled
    # explicitly as 0 points rather than blocking classification) ---
    exposed = _text(ctx.get("internet_exposed")).lower()
    zone = _text(ctx.get("network_zone"))
    if exposed == "yes":
        score += 15
        contributions.append(
            PriorityRuleContribution(
                rule="exposure:internet_exposed_yes",
                factor="internet exposure",
                points=15,
                max_points=FACTOR_MAX_POINTS["internet exposure"],
                evidence_value="internet_exposed=yes",
                evidence_source="asset.internet_exposed",
            )
        )
    elif zone == "DMZ":
        score += 10
        contributions.append(
            PriorityRuleContribution(
                rule="exposure:dmz_inferred",
                factor="internet exposure",
                points=10,
                max_points=FACTOR_MAX_POINTS["internet exposure"],
                evidence_value="network_zone=DMZ (internet-adjacent)",
                evidence_source="asset.network_zone",
            )
        )
    elif exposed == "no" or zone in ("Internal", "Restricted"):
        contributions.append(
            PriorityRuleContribution(
                rule="exposure:not_exposed",
                factor="internet exposure",
                points=0,
                max_points=FACTOR_MAX_POINTS["internet exposure"],
                evidence_value=f"internet_exposed={exposed or 'unknown'}; network_zone={zone or 'unknown'}",
                evidence_source="asset.internet_exposed / asset.network_zone",
            )
        )
    else:
        # Unknown exposure is treated as 0 points (not low risk) and recorded
        # as an unknown factor so the result remains explainable.
        contributions.append(
            PriorityRuleContribution(
                rule="exposure:unknown",
                factor="internet exposure",
                points=0,
                max_points=FACTOR_MAX_POINTS["internet exposure"],
                evidence_value=f"internet_exposed={exposed or 'unknown'}; network_zone={zone or 'unknown'}",
                evidence_source="asset.internet_exposed / asset.network_zone",
            )
        )
        unknowns.append("internet exposure")

    # --- Data sensitivity (optional) ---
    sensitivity = _text(ctx.get("data_sensitivity"))
    if sensitivity in _SENSITIVITY_POINTS:
        points = _SENSITIVITY_POINTS[sensitivity]
        score += points
        contributions.append(
            PriorityRuleContribution(
                rule=f"data_sensitivity:{sensitivity}",
                factor="data sensitivity",
                points=points,
                max_points=FACTOR_MAX_POINTS["data sensitivity"],
                evidence_value=sensitivity,
                evidence_source="asset.data_sensitivity",
            )
        )
    else:
        unknowns.append("data sensitivity")

    # --- Environment (optional) ---
    environment = _text(ctx.get("environment"))
    if environment == "production":
        score += 5
        contributions.append(
            PriorityRuleContribution(
                rule="environment:production",
                factor="environment",
                points=5,
                max_points=FACTOR_MAX_POINTS["environment"],
                evidence_value="production",
                evidence_source="asset.environment",
            )
        )
    elif environment in ("staging", "development"):
        contributions.append(
            PriorityRuleContribution(
                rule=f"environment:{environment}",
                factor="environment",
                points=0,
                max_points=FACTOR_MAX_POINTS["environment"],
                evidence_value=environment,
                evidence_source="asset.environment",
            )
        )
    else:
        unknowns.append("environment")

    # --- CIA baseline (optional) ---
    cia_values = [
        _text(ctx.get(f))
        for f in ("confidentiality_requirement", "integrity_requirement", "availability_requirement")
    ]
    if any(v == "H" for v in cia_values):
        score += 2
        contributions.append(
            PriorityRuleContribution(
                rule="cia_baseline:high_declared",
                factor="CIA baseline",
                points=2,
                max_points=FACTOR_MAX_POINTS["CIA baseline"],
                evidence_value="at least one CR/IR/AR = H",
                evidence_source="asset.confidentiality_requirement/integrity_requirement/availability_requirement",
            )
        )
    elif any(v in ("L", "M", "X") for v in cia_values):
        contributions.append(
            PriorityRuleContribution(
                rule="cia_baseline:non_high",
                factor="CIA baseline",
                points=0,
                max_points=FACTOR_MAX_POINTS["CIA baseline"],
                evidence_value="no CR/IR/AR = H",
                evidence_source="asset.confidentiality_requirement/integrity_requirement/availability_requirement",
            )
        )
    else:
        unknowns.append("CIA baseline")

    # --- CPE applicability (required) ---
    if app_status == ApplicabilityStatus.CONFIRMED:
        score += 15
        contributions.append(
            PriorityRuleContribution(
                rule="applicability:confirmed",
                factor="CPE applicability",
                points=15,
                max_points=FACTOR_MAX_POINTS["CPE applicability"],
                evidence_value="Confirmed",
                evidence_source=app_evidence_source,
            )
        )
    elif app_status == ApplicabilityStatus.UNKNOWN:
        score += 5
        contributions.append(
            PriorityRuleContribution(
                rule="applicability:unknown",
                factor="CPE applicability",
                points=5,
                max_points=FACTOR_MAX_POINTS["CPE applicability"],
                evidence_value="Unknown",
                evidence_source=app_evidence_source,
            )
        )
        unknowns.append("CPE applicability")
    else:
        missing.append("CPE applicability")

    # graph_dependencies is intentionally ignored (compatibility-only).

    # --- Status resolution ---
    has_required_criticality = criticality in _CRITICALITY_POINTS
    missing = list(dict.fromkeys(missing))
    if app_status == ApplicabilityStatus.UNKNOWN:
        # Unknown never enters the confirmed-remediation queue. The score and
        # P-band here rank verification work, not patching decisions.
        investigation_priority = None
        investigation_level = None
        investigation_score = None
        if has_required_criticality:
            investigation_score = score
            investigation_priority, investigation_level = _level_for_score(score)
        return OrganizationalPriorityResult(
            priority=None,
            status=_STATUS_NEEDS_VERIFICATION,
            score=investigation_score,
            max_score=MAX_SCORE,
            level=None,
            rule_contributions=tuple(contributions),
            missing_evidence=tuple(missing),
            unknown_factors=tuple(unknowns),
            explanation=_build_explanation(
                investigation_priority, investigation_level, investigation_score,
                contributions, missing, unknowns, _STATUS_NEEDS_VERIFICATION,
            ),
            cve_id=cve_id,
            asset_id=asset_id,
            investigation_priority=investigation_priority,
        )

    # Confirmed applicability is mandatory for remediation. Exposure remains
    # optional (unknown = 0, disclosed); confirmed applicability can support
    # classification when numerical severity is missing, as in the prior API.
    can_classify = has_required_criticality and app_status == ApplicabilityStatus.CONFIRMED

    if not can_classify:
        missing = list(dict.fromkeys(missing))  # dedupe while preserving order
        return OrganizationalPriorityResult(
            priority=None,
            status=_STATUS_INCOMPLETE,
            score=None,
            max_score=MAX_SCORE,
            level=None,
            rule_contributions=tuple(contributions),
            missing_evidence=tuple(missing),
            unknown_factors=tuple(unknowns),
            explanation=_build_explanation(None, None, None, contributions, missing, unknowns, _STATUS_INCOMPLETE),
            cve_id=cve_id,
            asset_id=asset_id,
        )

    priority, level = _level_for_score(score)
    return OrganizationalPriorityResult(
        priority=priority,
        status=_STATUS_COMPLETE,
        score=score,
        max_score=MAX_SCORE,
        level=level,
        rule_contributions=tuple(contributions),
        missing_evidence=tuple(missing),
        unknown_factors=tuple(unknowns),
        explanation=_build_explanation(priority, level, score, contributions, missing, unknowns, _STATUS_COMPLETE),
        cve_id=cve_id,
        asset_id=asset_id,
    )

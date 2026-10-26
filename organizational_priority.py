"""Deterministic organizational remediation priority (Phase C).

Computes Organizational Priority (P1–P4) for a vulnerability affecting a
specific asset. The module is intentionally separate from CVSS scoring:
CVSS severity measures the technical characteristics of the vulnerability,
while this rubric measures how urgently the organization should remediate it
on this asset.

Rubric (max 100 points):

  Factor                          Points                                  Kind
  ------------------------------  --------------------------------------  --------
  Vulnerability severity          Critical 20 / High 15 / Medium 8 / Low 2  required
  Business criticality            Mission Critical 25 / High 18 /       required
                                  Medium 10 / Low 3
  Internet exposure               yes=15 / DMZ=10 / no=0                  required
  Data sensitivity                Restricted 8 / Confidential 5 /         optional
                                  Internal 2 / Public 0
  Environment                     production=5 (others 0)                 optional
  CIA baseline                    any declared H = 2                      optional
  CPE applicability               Confirmed 15 / Unknown 5 /              required
                                  Not Applicable 0 (caps at P4)
  Graph dependencies              incoming dependency = 10 /              optional
                                  outgoing dependency = 5

  Thresholds: P1 (Urgent) >= 80 | P2 (High) >= 60 | P3 (Normal) >= 40 | P4 (Low) < 40

Design rules:
- Not Applicable CPE status caps the result at P4, because a CVE that does
  not affect the asset does not warrant urgent remediation.
- Missing required evidence (business criticality, or both vulnerability
  severity AND applicability) produces an explicit incomplete state with
  priority=None instead of silently defaulting to low risk.
- Optional unknown factors contribute 0 points and are listed in
  `unknown_factors`; they do not block classification.
- Graph dependencies are used only when the caller supplies verified
  relationship data; no relationships are fabricated.
- Identical inputs always produce identical outputs (pure function, frozen
  result dataclasses).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cve_applicability import ApplicabilityStatus

MAX_SCORE = 100

# Thresholds for P1..P4.
SCORE_THRESHOLDS: Sequence[Tuple[int, str, str]] = (
    (80, "P1", "Urgent"),
    (60, "P2", "High"),
    (40, "P3", "Normal"),
)

# Factor point tables.
_SEVERITY_POINTS: Dict[str, int] = {
    "Critical": 20,
    "High": 15,
    "Medium": 8,
    "Low": 2,
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
    status: str  # complete | incomplete | not_applicable
    score: Optional[int]
    max_score: int
    level: Optional[str]  # Urgent / High / Normal / Low
    rule_contributions: Tuple[PriorityRuleContribution, ...]
    missing_evidence: Tuple[str, ...]
    unknown_factors: Tuple[str, ...]
    explanation: str
    cve_id: Optional[str] = None
    asset_id: Optional[str] = None


def _severity_band(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    if score > 0.0:
        return "Low"
    return None


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
            "P4 (Low) — the CVE is Not Applicable to this asset based on CPE matching; "
            "no urgent remediation is warranted.",
        ]
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
            "Optional factors with unknown evidence (counted as 0, not low risk): "
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
        graph_dependencies: Optional dict with keys like
            `incoming_count` / `outgoing_count` describing verified graph
            relationships for the asset.
        cve_id: Optional CVE identifier for traceability.

    Returns:
        OrganizationalPriorityResult with priority, status, score,
        contributions, missing/unknown evidence, and explanation.
    """
    ctx: Mapping[str, Any] = asset_metadata if isinstance(asset_metadata, Mapping) else {}
    asset_id = _text(ctx.get("asset_id")) or None

    contributions: List[PriorityRuleContribution] = []
    missing: List[str] = []
    unknowns: List[str] = []
    score = 0

    # --- CPE applicability (evaluated first because it can cap or block) ---
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
        # CVE does not affect this asset -> priority is P4 by definition.
        return OrganizationalPriorityResult(
            priority="P4",
            status=_STATUS_NOT_APPLICABLE,
            score=0,
            max_score=MAX_SCORE,
            level="Low",
            rule_contributions=(),
            missing_evidence=(),
            unknown_factors=(),
            explanation="P4 (Low) — the CVE is Not Applicable to this asset based on CPE matching; "
                        "no urgent remediation is warranted.",
            cve_id=cve_id,
            asset_id=asset_id,
        )

    # --- Vulnerability severity (prefer context_score, fall back to base_score) ---
    severity: Optional[str] = None
    severity_source = "nvd.base_score"
    severity_score = context_score
    if severity_score is not None and severity_score > 0.0:
        severity_source = "contextual.cvss_environmental_score"
    else:
        severity_score = base_score

    if severity_score is not None and severity_score > 0.0:
        severity = _severity_band(severity_score)

    if severity:
        points = _SEVERITY_POINTS[severity]
        score += points
        contributions.append(
            PriorityRuleContribution(
                rule=f"vulnerability_severity:{severity}",
                factor="vulnerability severity",
                points=points,
                max_points=20,
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
                max_points=25,
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
                max_points=15,
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
                max_points=15,
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
                max_points=15,
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
                max_points=15,
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
                max_points=8,
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
                max_points=5,
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
                max_points=5,
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
                max_points=2,
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
                max_points=2,
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
                max_points=15,
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
                max_points=15,
                evidence_value="Unknown",
                evidence_source=app_evidence_source,
            )
        )
        unknowns.append("CPE applicability")
    else:
        missing.append("CPE applicability")

    # --- Graph dependencies (optional) ---
    dep_points = 0
    dep_description = []
    if graph_dependencies:
        incoming = int(graph_dependencies.get("incoming_count", 0))
        outgoing = int(graph_dependencies.get("outgoing_count", 0))
        if incoming > 0:
            dep_points += 10
            dep_description.append(f"{incoming} incoming dependency(s)")
        if outgoing > 0:
            dep_points += 5
            dep_description.append(f"{outgoing} outgoing dependency(s)")
    if dep_points > 0:
        score += dep_points
        contributions.append(
            PriorityRuleContribution(
                rule=f"graph_dependencies:{dep_points}",
                factor="business dependencies",
                points=dep_points,
                max_points=15,
                evidence_value="; ".join(dep_description),
                evidence_source="asset_graph",
            )
        )
    else:
        unknowns.append("business dependencies")

    # --- Status resolution ---
    has_required_severity = severity is not None
    has_required_criticality = criticality in _CRITICALITY_POINTS
    has_required_applicability = app_status in (
        ApplicabilityStatus.CONFIRMED,
        ApplicabilityStatus.UNKNOWN,
    )
    # We need severity OR applicability to reason about urgency; criticality is mandatory.
    # Exposure is intentionally treated as optional (unknown = 0 points, not blocking).
    can_classify = has_required_criticality and (
        has_required_severity or has_required_applicability
    )

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
        missing_evidence=(),
        unknown_factors=tuple(unknowns),
        explanation=_build_explanation(priority, level, score, contributions, missing, unknowns, _STATUS_COMPLETE),
        cve_id=cve_id,
        asset_id=asset_id,
    )

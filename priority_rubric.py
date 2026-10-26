"""Deterministic organizational remediation-priority rubric (Phase 2C).

Computes Organizational Priority (P1-P4) from verified evidence only:
- Organizational asset context (Asset Schema v2 fields retrieved via RAG).
- The NVD CVSS v3.1 base score (technical baseline; banding only - the CVSS
  formulas themselves are never modified or recomputed here).

This is intentionally SEPARATE from CVSS Base/Threat/Environmental scoring:
CVSS severity answers "how technically severe is the vulnerability", while
this rubric answers "in what order should THIS organization remediate it",
combining technical severity with asset criticality, exposure, data
sensitivity, environment, and declared CIA baselines.

Responsibility boundary (do not blur):
- The rubric NEVER modifies a CVSS vector or score (see app.build_environmental_vector
  and the `cvss` library for that).
- The rubric NEVER invents missing attributes: unknown/absent evidence is
  surfaced explicitly (`unknown_factors`) and, when a required factor is
  unknown, the result is an explicit incomplete state (priority=None,
  status="incomplete") instead of a silently low priority.
- The LLM's organizational_priority (llm_assessment.ContextualAssessment) is
  preserved unchanged as a model suggestion; this module is the deterministic
  counterpart used for reproducible/defensible results and later evaluation.

Rubric (max 100 points; weights justified in docs/PRIORITY_RUBRIC.md):

  Factor                  Points                              Kind
  ----------------------  ----------------------------------  --------
  vulnerability_severity  Critical=35 High=27 Medium=14 Low=4 required
  business_criticality    MissionCritical=30 High=22         required
                          Medium=11 Low=3
  exposure                internet yes=20; zone DMZ=12;      required
                          internal/restricted=0
  data_sensitivity        Restricted=8 Confidential=5        optional
                          Internal=2 Public=0
  environment             production=5 (staging/dev=0)       optional
  cia_baseline            any declared CR/IR/AR = H -> 2     optional

  Thresholds: P1 >= 70 | P2 >= 50 | P3 >= 30 | P4 < 30

Design properties (tested):
- vulnerability_severity is capped at 35 (< 50% of max): a vulnerability can
  never reach P1 on technical severity alone; P1 requires organizational
  co-evidence (e.g. Mission Critical + Restricted data, or High criticality
  + internet exposure).
- A Low-severity vulnerability can never exceed P2 even on the highest-stakes
  asset (max 4+30+20+8+5+2 = 69 < 70).
- Conflicting evidence (e.g. internet_exposed=yes on an Internal zone) never
  silently downgrades risk: the higher-risk interpretation is used and the
  conflict is recorded.

Stdlib-only (like asset_schema_v2.py) so the rubric is reusable by evaluation
scripts and testable without Streamlit/LLM/ChromaDB dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

MAX_SCORE = 100
PRIORITY_LEVELS: Tuple[str, ...] = ("P1", "P2", "P3", "P4")

# (minimum_score, priority) evaluated in order; below the last threshold -> P4.
SCORE_THRESHOLDS: Tuple[Tuple[int, str], ...] = ((70, "P1"), (50, "P2"), (30, "P3"))

# CVSS v3.1 qualitative severity rating scale (FIRST), banded from the NVD
# base score. Used only to bucket an already-computed NVD score - no CVSS
# math happens in this module.
_SEVERITY_POINTS: Dict[str, int] = {"Critical": 35, "High": 27, "Medium": 14, "Low": 4}

# Asset Schema v2 enum values -> points (weights documented above and in
# docs/PRIORITY_RUBRIC.md).
_BUSINESS_CRITICALITY_POINTS: Dict[str, int] = {
    "Mission Critical": 30,
    "High": 22,
    "Medium": 11,
    "Low": 3,
}
_DATA_SENSITIVITY_POINTS: Dict[str, int] = {
    "Restricted": 8,
    "Confidential": 5,
    "Internal": 2,
    "Public": 0,
}
_ENVIRONMENT_PRODUCTION_POINTS = 5
_ENVIRONMENT_KNOWN_NON_PRODUCTION = ("staging", "development")
_CIA_HIGH_BASELINE_POINTS = 2
_CIA_KNOWN_CODES = ("X", "L", "M", "H")

_EXPOSURE_EXPLICIT_YES = 20
_EXPOSURE_DMZ_DERIVED = 12  # zone-derived (internet-adjacent segment)
_EXPOSURE_INTERNAL_DERIVED = 0  # zone-derived (non-DMZ segment)

_NETWORK_ZONES = ("Internal", "DMZ", "Restricted")
_EXPOSURE_YES = "yes"
_EXPOSURE_NO = "no"

_FACTOR_LABELS: Dict[str, str] = {
    "vulnerability_severity": "kerentanan teknis (banding skor dasar NVD)",
    "business_criticality": "kritikalitas bisnis",
    "exposure": "paparan internet",
    "data_sensitivity": "sensitivitas data",
    "environment": "lingkungan deployment",
    "cia_baseline": "baseline CR/IR/AR yang dideklarasikan",
}

_STATUS_COMPLETE = "complete"
_STATUS_INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class RuleContribution:
    """One rubric factor's contribution to the priority score."""

    factor: str
    value: str  # human-readable evidence value used
    points: int
    evidence_source: str  # which field(s) the value came from


@dataclass(frozen=True)
class PriorityResult:
    """Structured result of the deterministic priority rubric.

    When status == "incomplete", priority/score are None and the explanation
    lists exactly which required evidence is missing - missing evidence is
    never silently scored as low risk.
    """

    priority: Optional[str]  # "P1".."P4" or None when incomplete
    status: str  # "complete" | "incomplete"
    score: Optional[int]  # 0..MAX_SCORE, None when incomplete
    max_score: int
    rule_contributions: Tuple[RuleContribution, ...]
    unknown_factors: Tuple[str, ...]
    conflicts: Tuple[str, ...]
    explanation: str


def _severity_band(score: float) -> str:
    """Bucket a CVSS base score into the FIRST qualitative severity scale."""
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    return "Low"


def _level_for_score(score: int) -> str:
    for threshold, level in SCORE_THRESHOLDS:
        if score >= threshold:
            return level
    return "P4"  # below the lowest threshold


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _build_explanation(
    priority: Optional[str],
    score: Optional[int],
    contributions: List[RuleContribution],
    unknown_factors: List[str],
    conflicts: List[str],
    missing_gates: List[str],
) -> str:
    if missing_gates:
        lines = [
            "Tidak dapat mengklasifikasikan prioritas secara meyakinkan "
            f"(status: {_STATUS_INCOMPLETE}) karena bukti wajib berikut "
            "tidak tersedia: " + "; ".join(missing_gates) + "."
        ]
        if contributions:
            recorded = ", ".join(
                f"{_FACTOR_LABELS.get(c.factor, c.factor)}={c.value} ({c.points} poin)"
                for c in contributions
            )
            lines.append("Bukti yang tersedia tetap dicatat: " + recorded + ".")
        lines.append(
            "Bukti yang hilang TIDAK diperlakukan sebagai risiko rendah; "
            "lengkapi metadata aset / data NVD lalu hitung ulang."
        )
        return " ".join(lines)

    parts = [
        f"{priority} - skor prioritas {score}/{MAX_SCORE}.",
        "Kontribusi: "
        + "; ".join(
            f"{_FACTOR_LABELS.get(c.factor, c.factor)}={c.value} ({c.points} poin)"
            for c in contributions
        )
        + ".",
    ]
    if unknown_factors:
        parts.append(
            "Faktor opsional dengan bukti tidak diketahui (dihitung 0 poin, "
            "bukan berisiko rendah): " + ", ".join(unknown_factors) + "."
        )
    if conflicts:
        parts.append(
            "Bukti bertentangan (kebijakan: dipakai interpretasi berisiko "
            "lebih tinggi): " + "; ".join(conflicts) + "."
        )
    return " ".join(parts)


def calculate_organizational_priority(
    asset_context: Optional[Mapping[str, Any]],
    base_score: Optional[float],
) -> PriorityResult:
    """Compute the deterministic Organizational Priority (P1-P4).

    Args:
        asset_context: Retrieved Asset Schema v2 metadata (e.g.
            rag_retrieval.filtered_metadata output). Missing/unknown fields
            are surfaced, never fabricated.
        base_score: Verified NVD CVSS v3.1 base score. 0.0/None/invalid is
            treated as "no severity evidence" (NVD reports 0.0 when no CVSS
            data is available) and makes the result incomplete.

    Returns:
        PriorityResult with priority, score, per-factor rule contributions
        (including evidence sources), unknown factors, conflicts, and a
        human-readable explanation. Pure function: same inputs -> same
        output.
    """
    ctx: Mapping[str, Any] = asset_context if isinstance(asset_context, Mapping) else {}
    contributions: List[RuleContribution] = []
    unknown_factors: List[str] = []
    conflicts: List[str] = []
    missing_gates: List[str] = []
    score = 0

    # -- Factor 1: vulnerability severity (required; NVD base score banding)
    try:
        numeric_score = float(base_score) if base_score is not None else None
    except (TypeError, ValueError):
        numeric_score = None
    if numeric_score is not None and 0.0 < numeric_score <= 10.0:
        band = _severity_band(numeric_score)
        points = _SEVERITY_POINTS[band]
        score += points
        contributions.append(
            RuleContribution(
                factor="vulnerability_severity",
                value=f"{band} (skor dasar NVD {numeric_score:g})",
                points=points,
                evidence_source="nvd.base_score",
            )
        )
    else:
        unknown_factors.append("vulnerability_severity")
        missing_gates.append(
            "skor dasar CVSS NVD tidak tersedia/tidak valid "
            "(0.0/None/ di luar rentang 0-10)"
        )

    # -- Factor 2: business criticality (required; Asset Schema v2 enum)
    criticality = _text(ctx.get("business_criticality"))
    if criticality in _BUSINESS_CRITICALITY_POINTS:
        points = _BUSINESS_CRITICALITY_POINTS[criticality]
        score += points
        contributions.append(
            RuleContribution(
                factor="business_criticality",
                value=criticality,
                points=points,
                evidence_source="asset.business_criticality",
            )
        )
    else:
        unknown_factors.append("business_criticality")
        missing_gates.append(
            "business_criticality aset tidak diketahui/tidak valid"
        )

    # -- Factor 3: exposure (required; internet_exposed, zone-derived fallback)
    exposed = _text(ctx.get("internet_exposed")).lower()
    zone = _text(ctx.get("network_zone"))
    if exposed == _EXPOSURE_YES:
        contributions.append(
            RuleContribution(
                factor="exposure",
                value="internet_exposed=yes",
                points=_EXPOSURE_EXPLICIT_YES,
                evidence_source="asset.internet_exposed",
            )
        )
        score += _EXPOSURE_EXPLICIT_YES
        if zone in ("Internal", "Restricted"):
            conflicts.append(
                f"internet_exposed=yes bertentangan dengan network_zone={zone}; "
                "dipakai bukti eksplisit (paparan lebih tinggi)"
            )
    elif exposed == _EXPOSURE_NO:
        contributions.append(
            RuleContribution(
                factor="exposure",
                value="internet_exposed=no",
                points=_EXPOSURE_INTERNAL_DERIVED,
                evidence_source="asset.internet_exposed",
            )
        )
    elif zone == "DMZ":
        contributions.append(
            RuleContribution(
                factor="exposure",
                value="internet_exposed tidak diketahui; network_zone=DMZ (tersirat internet-adjacent)",
                points=_EXPOSURE_DMZ_DERIVED,
                evidence_source="asset.network_zone (tersirat)",
            )
        )
        score += _EXPOSURE_DMZ_DERIVED
    elif zone in ("Internal", "Restricted"):
        contributions.append(
            RuleContribution(
                factor="exposure",
                value=f"internet_exposed tidak diketahui; network_zone={zone} (tersirat non-DMZ)",
                points=_EXPOSURE_INTERNAL_DERIVED,
                evidence_source="asset.network_zone (tersirat)",
            )
        )
    else:
        unknown_factors.append("exposure")
        missing_gates.append(
            "paparan internet tidak dapat ditentukan "
            "(internet_exposed dan network_zone tidak diketahui)"
        )

    # -- Factor 4: data sensitivity (optional boost)
    sensitivity = _text(ctx.get("data_sensitivity"))
    if sensitivity in _DATA_SENSITIVITY_POINTS:
        points = _DATA_SENSITIVITY_POINTS[sensitivity]
        score += points
        contributions.append(
            RuleContribution(
                factor="data_sensitivity",
                value=sensitivity,
                points=points,
                evidence_source="asset.data_sensitivity",
            )
        )
    else:
        unknown_factors.append("data_sensitivity")

    # -- Factor 5: environment (optional boost; production only)
    environment = _text(ctx.get("environment"))
    if environment == "production":
        score += _ENVIRONMENT_PRODUCTION_POINTS
        contributions.append(
            RuleContribution(
                factor="environment",
                value="production",
                points=_ENVIRONMENT_PRODUCTION_POINTS,
                evidence_source="asset.environment",
            )
        )
    elif environment in _ENVIRONMENT_KNOWN_NON_PRODUCTION:
        contributions.append(
            RuleContribution(
                factor="environment",
                value=environment,
                points=0,
                evidence_source="asset.environment",
            )
        )
    else:
        unknown_factors.append("environment")

    # -- Factor 6: declared CIA baseline (optional; any H -> small boost)
    cia_values = [
        _text(ctx.get(f))
        for f in ("confidentiality_requirement", "integrity_requirement",
                  "availability_requirement")
    ]
    if any(v == "H" for v in cia_values):
        score += _CIA_HIGH_BASELINE_POINTS
        contributions.append(
            RuleContribution(
                factor="cia_baseline",
                value="salah satu CR/IR/AR dideklarasikan H",
                points=_CIA_HIGH_BASELINE_POINTS,
                evidence_source="asset.confidentiality_requirement/"
                "integrity_requirement/availability_requirement",
            )
        )
    elif any(v in _CIA_KNOWN_CODES for v in cia_values):
        contributions.append(
            RuleContribution(
                factor="cia_baseline",
                value="CR/IR/AR dideklarasikan tanpa nilai H",
                points=0,
                evidence_source="asset.confidentiality_requirement/"
                "integrity_requirement/availability_requirement",
            )
        )
    else:
        unknown_factors.append("cia_baseline")

    # -- Status & classification
    if missing_gates:
        priority: Optional[str] = None
        status = _STATUS_INCOMPLETE
        final_score: Optional[int] = None
    else:
        priority = _level_for_score(score)
        status = _STATUS_COMPLETE
        final_score = score

    explanation = _build_explanation(
        priority, final_score, contributions, unknown_factors,
        conflicts, missing_gates,
    )

    return PriorityResult(
        priority=priority,
        status=status,
        score=final_score,
        max_score=MAX_SCORE,
        rule_contributions=tuple(contributions),
        unknown_factors=tuple(unknown_factors),
        conflicts=tuple(conflicts),
        explanation=explanation,
    )

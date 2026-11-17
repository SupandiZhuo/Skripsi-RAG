"""Strict CVSS v3.1 Base validation and deterministic Environmental scoring.

Only CR/IR/AR may be appended; no Base override, Temporal or Modified metric
is accepted as an NVD baseline. The model never supplies a numerical score.
"""
from collections.abc import Mapping
from enum import Enum

from cvss import CVSS3

BASE_VALUES = {"AV": {"N", "A", "L", "P"}, "AC": {"L", "H"},
               "PR": {"N", "L", "H"}, "UI": {"N", "R"}, "S": {"U", "C"},
               "C": {"N", "L", "H"}, "I": {"N", "L", "H"}, "A": {"N", "L", "H"}}
REQUIREMENT_FIELDS = {"CR": "confidentiality_requirement", "IR": "integrity_requirement",
                      "AR": "availability_requirement"}


def _parse(vector: str, allow_requirements: bool = False) -> dict[str, str]:
    if not isinstance(vector, str) or not vector.startswith("CVSS:3.1/"):
        raise ValueError("Only literal CVSS:3.1 vectors are supported.")
    result = {}
    for token in vector.split("/")[1:]:
        if token.count(":") != 1:
            raise ValueError("Malformed CVSS metric token.")
        key, value = token.split(":")
        allowed = BASE_VALUES.get(key)
        if allow_requirements and key in REQUIREMENT_FIELDS:
            allowed = {"X", "L", "M", "H"}
        if key in result:
            raise ValueError(f"Duplicate CVSS metric: {key}.")
        if allowed is None or value not in allowed:
            raise ValueError(f"Unsupported metric or invalid value: {key}.")
        result[key] = value
    if not BASE_VALUES.keys() <= result.keys():
        raise ValueError("CVSS vector is missing required Base metrics.")
    return result


def validate_base_vector(vector: str) -> str:
    """Return the original vector unchanged, or reject non-Base/duplicate input."""
    _parse(vector)
    CVSS3(vector)
    return vector


def build_environmental_vector(vector: str, assessment) -> str:
    validate_base_vector(vector)
    tokens = []
    for metric, field in REQUIREMENT_FIELDS.items():
        value = assessment.get(field) if isinstance(assessment, Mapping) else getattr(assessment, field, None)
        if isinstance(value, Enum):
            value = value.value
        if value not in {"X", "L", "M", "H"}:
            raise ValueError(f"Invalid or missing {metric} requirement.")
        tokens.append(f"{metric}:{value}")
    result = vector + "/" + "/".join(tokens)
    CVSS3(result)
    return result


def severity(score: float) -> str:
    if score == 0:
        return "None"
    if score < 4:
        return "Low"
    if score < 7:
        return "Medium"
    if score < 9:
        return "High"
    return "Critical"


def calculate_cvss(vector: str) -> dict:
    """Score only a validated Base vector, optionally with CR/IR/AR."""
    _parse(vector, allow_requirements=True)
    base, temporal, environmental = (float(value) for value in CVSS3(vector).scores())
    return {"base_score": base, "temporal_score": temporal,
            "environmental_score": environmental, "technical_severity": severity(base),
            "environmental_severity": severity(environmental),
            "score_method": "cvss-library-v3.1"}

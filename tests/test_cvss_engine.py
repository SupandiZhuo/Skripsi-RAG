"""CVSS deterministic-only contracts; fixtures are not research labels."""
from types import SimpleNamespace

import pytest

from cvss_engine import build_environmental_vector, calculate_cvss, validate_base_vector

BASE = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def assessment(cr="H", ir="M", ar="L"):
    return SimpleNamespace(confidentiality_requirement=cr, integrity_requirement=ir,
                           availability_requirement=ar)


def test_preserves_exact_base_tokens():
    vector = build_environmental_vector(BASE, assessment())
    assert vector == BASE + "/CR:H/IR:M/AR:L"
    assert "MAV:" not in vector


@pytest.mark.parametrize("suffix", ["/AV:A", "/E:F", "/CR:H", "/MAV:L", "/MC:H", "/ZZ:X"])
def test_non_base_or_duplicate_rejected(suffix):
    with pytest.raises(ValueError):
        build_environmental_vector(BASE + suffix, assessment())


@pytest.mark.parametrize("vector", [BASE.replace("3.1", "3.0"), BASE.replace("/A:H", ""),
                                    BASE.replace("AV:N", "AV:Z"), " " + BASE, BASE + "/"])
def test_strict_base_validation(vector):
    with pytest.raises(ValueError):
        validate_base_vector(vector)


def test_invalid_requirement_rejected():
    with pytest.raises(ValueError):
        build_environmental_vector(BASE, assessment(cr="unknown"))


def test_cvss_golden_base_and_undefined_requirements():
    scored = calculate_cvss(BASE)
    assert scored["base_score"] == 9.8
    assert scored["technical_severity"] == "Critical"
    assert calculate_cvss(build_environmental_vector(BASE, assessment("X", "X", "X")))["environmental_score"] == 9.8


def test_zero_score_is_not_missing():
    vector = BASE.replace("C:H/I:H/A:H", "C:N/I:N/A:N")
    scored = calculate_cvss(vector)
    assert scored["base_score"] == 0.0
    assert scored["technical_severity"] == "None"

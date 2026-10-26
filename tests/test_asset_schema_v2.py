"""Targeted checks for Asset Knowledge Base Schema v2 (Phase 1).

Covers: v2 CSV validates cleanly, seven original assets retained, v1 fields
migrated byte-identical, unknown values not fabricated, evidence-based
sample values, and validator error reporting (missing required field, bad
enum, bad CVSS requirement code, duplicate asset_id, bad header).

No ChromaDB or pipeline files are touched. Run:
python -m pytest test_asset_schema_v2.py -q
"""

import csv
from pathlib import Path

from asset_schema_v2 import (
    V2_COLUMNS,
    load_assets,
    validate_csv,
    validate_row,
)

ROOT = Path(__file__).resolve().parents[1]
V1_CSV = ROOT / "asset_data.csv"
V2_CSV = ROOT / "data" / "asset_data_v2.csv"

V1_COLUMNS = [
    "asset_id",
    "asset_name",
    "ip_address",
    "network_zone",
    "business_criticality",
    "data_sensitivity",
    "description",
]


def read_v2_rows():
    return load_assets(str(V2_CSV))


def valid_row():
    """A fully valid template row for override-based negative tests."""
    return {column: "unknown" for column in V2_COLUMNS} | {
        "asset_id": "SRV-999",
        "asset_name": "Test Asset",
        "ip_address": "10.0.0.1",
        "network_zone": "Internal",
        "business_criticality": "Low",
        "data_sensitivity": "Internal",
        "description": "Row used only by schema tests.",
        "environment": "unknown",
        "internet_exposed": "unknown",
        "asset_type": "other",
        "confidentiality_requirement": "X",
        "integrity_requirement": "X",
        "availability_requirement": "X",
        "last_updated": "2026-10-21",
        "source": "synthetic_sample",
        "source_reference": "schema test",
        "is_synthetic": "true",
        "schema_version": "2.0",
    }


# ---------------------------------------------------------------------------
# 1. Real dataset parses and validates
# ---------------------------------------------------------------------------
def test_v2_csv_passes_validation():
    assert validate_csv(str(V2_CSV)) == []


def test_seven_original_assets_retained():
    rows = read_v2_rows()
    assert len(rows) == 7
    assert [row["asset_id"] for row in rows] == [
        "SRV-001", "SRV-002", "SRV-003", "SRV-004",
        "SRV-005", "SRV-006", "SRV-007",
    ]


def test_v1_fields_preserved_identically():
    v1 = {row["asset_id"]: row for row in load_assets(str(V1_CSV))}
    v2 = {row["asset_id"]: row for row in read_v2_rows()}
    assert v1.keys() == v2.keys()
    for asset_id, v1_row in v1.items():
        for column in V1_COLUMNS:
            assert v2[asset_id][column] == v1_row[column], (
                f"{asset_id}: v1 column '{column}' changed during migration"
            )


def test_v2_columns_are_superset_of_v1():
    for column in V1_COLUMNS:
        assert column in V2_COLUMNS


# ---------------------------------------------------------------------------
# 2. Unknown information is not fabricated (evidence-based migration)
# ---------------------------------------------------------------------------
def test_unknown_fields_not_fabricated():
    rows = {row["asset_id"]: row for row in read_v2_rows()}
    for asset_id in ("SRV-003", "SRV-004", "SRV-005", "SRV-007"):
        row = rows[asset_id]
        for field in ("owner", "vendor", "software_version", "cpe"):
            assert row[field] == "unknown", f"{asset_id}: {field} fabricated"
        for field in (
            "confidentiality_requirement",
            "integrity_requirement",
            "availability_requirement",
        ):
            assert row[field] == "X", f"{asset_id}: {field} invented"
        assert row["environment"] == "unknown"


def test_evidence_based_values_only():
    rows = {row["asset_id"]: row for row in read_v2_rows()}
    assert rows["SRV-001"]["internet_exposed"] == "no"   # "terisolasi dari internet"
    assert rows["SRV-002"]["internet_exposed"] == "yes"  # "menghadap ... internet publik"
    assert rows["SRV-006"]["software_name"] == "WordPress"  # "Server CMS WordPress"
    # Nothing else inferred: no version/CPE anywhere.
    for row in rows.values():
        assert row["software_version"] == "unknown"
        assert row["cpe"] == "unknown"
        assert row["vendor"] == "unknown"


def test_rows_flagged_as_synthetic():
    for row in read_v2_rows():
        assert row["is_synthetic"] == "true"
        assert row["source"] == "synthetic_sample"
        assert row["schema_version"] == "2.0"


# ---------------------------------------------------------------------------
# 3. Validator error reporting
# ---------------------------------------------------------------------------
def test_valid_template_row_has_no_errors():
    assert validate_row(valid_row(), row_number=2) == []


def test_missing_required_field_rejected_with_message():
    row = valid_row()
    row["asset_id"] = ""
    errors = validate_row(row, row_number=2)
    assert len(errors) == 1
    assert "row 2" in errors[0]
    assert "'asset_id'" in errors[0]
    assert "missing or empty" in errors[0]


def test_invalid_enum_value_rejected_with_expected_values():
    row = valid_row()
    row["business_criticality"] = "Critical"
    errors = validate_row(row, row_number=3)
    assert len(errors) == 1
    assert "'business_criticality'" in errors[0]
    assert "Mission Critical" in errors[0]  # allowed values shown


def test_invalid_cia_code_rejected():
    row = valid_row()
    row["confidentiality_requirement"] = "High"  # full word, not CVSS code H
    errors = validate_row(row, row_number=4)
    assert len(errors) == 1
    assert "'confidentiality_requirement'" in errors[0]
    assert "[X, L, M, H]" in errors[0]


def test_invalid_date_rejected():
    row = valid_row()
    row["last_updated"] = "21-10-2026"
    errors = validate_row(row, row_number=5)
    assert len(errors) == 1
    assert "'last_updated'" in errors[0]
    assert "YYYY-MM-DD" in errors[0]


def test_invalid_schema_version_rejected():
    row = valid_row()
    row["schema_version"] = "1.0"
    errors = validate_row(row, row_number=6)
    assert any("'schema_version'" in e for e in errors)


def test_wrong_header_rejected(tmp_path):
    bad = tmp_path / "bad_header.csv"
    rows = read_v2_rows()
    header = [c for c in V2_COLUMNS if c != "vendor"]  # dropped column
    with open(bad, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[c] for c in header])
    errors = validate_csv(str(bad))
    assert any("missing column(s): vendor" in e for e in errors)


def test_duplicate_asset_id_rejected(tmp_path):
    dup = tmp_path / "dup.csv"
    rows = read_v2_rows()
    with open(dup, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(V2_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in V2_COLUMNS])
        writer.writerow([rows[0][c] for c in V2_COLUMNS])  # duplicate SRV-001
    errors = validate_csv(str(dup))
    assert any("duplicate asset_id" in e for e in errors)


def test_missing_file_reported():
    errors = validate_csv(str(ROOT / "data" / "does_not_exist.csv"))
    assert len(errors) == 1
    assert "file not found" in errors[0]

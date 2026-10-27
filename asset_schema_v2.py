"""Asset Knowledge Base Schema v2 validator (Phase 1).

Validates rows of `data/asset_data_v2.csv` against the Schema v2 defined in
`docs/ASSET_SCHEMA_V2.md`. Stdlib-only (csv module) so no new dependency is
introduced. Kept separate from Streamlit/ingestion so it can be reused by
evaluation scripts.

Run: python asset_schema_v2.py            # validate data/asset_data_v2.csv
     python asset_schema_v2.py <path>     # validate another CSV file
"""

import csv
import sys
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

from asset_inventory import is_valid_cpe

SCHEMA_VERSION = "2.0"
UNKNOWN = "unknown"

V2_COLUMNS: List[str] = [
    "asset_id",
    "asset_name",
    "ip_address",
    "network_zone",
    "business_criticality",
    "data_sensitivity",
    "description",
    "environment",
    "internet_exposed",
    "asset_type",
    "owner",
    "owner_contact",
    "business_function",
    "vendor",
    "software_name",
    "software_version",
    "cpe",
    "confidentiality_requirement",
    "integrity_requirement",
    "availability_requirement",
    "data_classification",
    "last_updated",
    "source",
    "source_reference",
    "is_synthetic",
    "schema_version",
]

# Runtime extensions may be appended in this order without changing existing
# dataset headers. Loading never infers completeness from a primary CPE.
V2_EXTENSION_COLUMNS = ("software_inventory_json", "inventory_verified")

# Required fields must be present and non-empty (v1 columns + structural v2).
REQUIRED_FIELDS: List[str] = [
    "asset_id",
    "asset_name",
    "ip_address",
    "network_zone",
    "business_criticality",
    "data_sensitivity",
    "description",
    "environment",
    "internet_exposed",
    "asset_type",
    "confidentiality_requirement",
    "integrity_requirement",
    "availability_requirement",
    "last_updated",
    "source",
    "source_reference",
    "is_synthetic",
    "schema_version",
]

OPTIONAL_FIELDS: List[str] = [
    "owner",
    "owner_contact",
    "business_function",
    "vendor",
    "software_name",
    "software_version",
    "cpe",
    "data_classification",
]

# Allowed values (v1 enum values are preserved exactly, including casing).
ALLOWED_VALUES: Dict[str, Sequence[str]] = {
    "network_zone": ("Internal", "DMZ", "Restricted"),
    "business_criticality": ("Low", "Medium", "High", "Mission Critical"),
    "data_sensitivity": ("Public", "Internal", "Confidential", "Restricted"),
    "environment": ("production", "staging", "development", "unknown"),
    "internet_exposed": ("yes", "no", "unknown"),
    "asset_type": (
        "database",
        "web_server",
        "domain_controller",
        "file_server",
        "cms",
        "other",
        "unknown",
    ),
    # CVSS v3.1 Environmental Requirement codes (FIRST): Not Defined/L/M/H.
    "confidentiality_requirement": ("X", "L", "M", "H"),
    "integrity_requirement": ("X", "L", "M", "H"),
    "availability_requirement": ("X", "L", "M", "H"),
    "is_synthetic": ("true", "false"),
    "schema_version": (SCHEMA_VERSION,),
}


def _format_allowed(values: Sequence[str]) -> str:
    return "[" + ", ".join(values) + "]"


def validate_row(
    row: Mapping[str, Any], row_number: Optional[int] = None
) -> List[str]:
    """Validate one asset row; returns a list of error messages (empty = valid).

    Error messages identify the row, the asset_id when known, the offending
    field, and the expected vs actual value.
    """
    errors: List[str] = []
    prefix = f"row {row_number}" if row_number is not None else "row"

    if not isinstance(row, Mapping):
        return [f"{prefix}: row must be a mapping, got {type(row).__name__}"]

    asset_id = str(row.get("asset_id") or "").strip()
    if asset_id:
        prefix = f"{prefix} ({asset_id})"

    missing = [f for f in REQUIRED_FIELDS if not str(row.get(f, "")).strip()]
    for field in missing:
        errors.append(
            f"{prefix}: required field '{field}' is missing or empty "
            f"(required fields: {', '.join(REQUIRED_FIELDS)})"
        )

    for field, allowed in ALLOWED_VALUES.items():
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            errors.append(
                f"{prefix}: field '{field}' has value {value!r} but only "
                f"allowed values are {_format_allowed(allowed)}"
            )

    flag = row.get("inventory_verified", "")
    if type(flag) is not bool and (not isinstance(flag, str) or flag not in ("", "unknown", "true", "false")):
        errors.append(f"{prefix}: field 'inventory_verified' must be true, false, unknown, or empty")
    cpe = row.get("cpe", "")
    if cpe not in (None, "", "unknown") and not is_valid_cpe(cpe):
        errors.append(f"{prefix}: field 'cpe' must be a complete CPE 2.3 formatted string, unknown, or empty")

    last_updated = str(row.get("last_updated", "")).strip()
    if last_updated:
        try:
            date.fromisoformat(last_updated)
        except ValueError:
            errors.append(
                f"{prefix}: field 'last_updated' has value {last_updated!r} "
                f"but must be an ISO date (YYYY-MM-DD)"
            )

    # Optional fields may be empty or 'unknown'; if present they must be
    # non-blank-after-strip (no whitespace-only cells) - no other constraints.
    for field in OPTIONAL_FIELDS:
        if field in row and row[field] is not None:
            raw = str(row[field])
            if raw.strip() == "" and raw != "":
                errors.append(
                    f"{prefix}: optional field '{field}' contains only "
                    f"whitespace; use an empty value or '{UNKNOWN}' instead"
                )

    return errors


def validate_csv(path: str) -> List[str]:
    """Validate a Schema v2 CSV file. Returns all errors (empty list = valid).

    Checks V2_COLUMNS plus optional appended extensions in declared order,
    structural
    row width, per-row field rules, and asset_id uniqueness (CM-8 style
    identity - a duplicate id would silently collide as RAG evidence).
    """
    errors: List[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return [f"{path}: file is empty (no header row)"]

            expected_header = V2_COLUMNS + [field for field in V2_EXTENSION_COLUMNS if field in header]
            if header != expected_header:
                missing = [c for c in V2_COLUMNS if c not in header]
                extra = [c for c in header if c not in V2_COLUMNS and c not in V2_EXTENSION_COLUMNS]
                if missing:
                    errors.append(
                        f"{path}: header is missing column(s): "
                        f"{', '.join(missing)}"
                    )
                if extra:
                    errors.append(
                        f"{path}: header has unexpected column(s): "
                        f"{', '.join(extra)}"
                    )
                if not missing and not extra:
                    errors.append(
                        f"{path}: header columns out of order; expected "
                        f"{', '.join(expected_header)}"
                    )

            seen_ids: Dict[str, int] = {}
            for index, values in enumerate(reader, start=2):  # 1-based + header
                if not any(str(v).strip() for v in values):
                    continue  # skip blank lines
                if len(values) != len(header):
                    errors.append(
                        f"{path}: row {index}: expected {len(header)} fields "
                        f"but found {len(values)}"
                    )
                    continue
                row = dict(zip(header, values))
                errors.extend(validate_row(row, row_number=index))

                current_id = str(row.get("asset_id", "")).strip()
                if current_id:
                    if current_id in seen_ids:
                        errors.append(
                            f"{path}: row {index}: duplicate asset_id "
                            f"{current_id!r} (first seen at row "
                            f"{seen_ids[current_id]})"
                        )
                    else:
                        seen_ids[current_id] = index
    except FileNotFoundError:
        errors.append(f"{path}: file not found")
    except OSError as exc:
        errors.append(f"{path}: could not be read ({exc})")

    return errors


def load_assets(path: str) -> List[Dict[str, str]]:
    """Parse a Schema v2 CSV into a list of row dicts (no validation)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str]) -> int:
    path = argv[1] if len(argv) > 1 else "data/asset_data_v2.csv"
    errors = validate_csv(path)
    if errors:
        print(f"FAIL: {path} - {len(errors)} error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    assets = load_assets(path)
    print(f"OK: {path} - {len(assets)} asset(s) validated against schema v{SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

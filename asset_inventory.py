"""Editable asset inventory helpers (Phase B1).

Supports associating multiple installed software records (with CPE 2.3
identifiers) to an asset and collecting the full CPE inventory for the
applicability evaluator. The software inventory is stored as a JSON string in
ChromaDB metadata under `software_inventory_json`, which is a runtime extension
of Asset Schema v2 and is treated as optional everywhere.
"""

import json
from typing import Any, Dict, List, Mapping

from cve_applicability import parse_cpe

SOFTWARE_INVENTORY_FIELD = "software_inventory_json"
INVENTORY_VERIFIED_FIELD = "inventory_verified"


def load_software_inventory(asset_metadata: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Load software records from asset metadata."""
    raw = asset_metadata.get(SOFTWARE_INVENTORY_FIELD, "")
    if not raw or str(raw).lower() == "unknown":
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [dict(r) for r in data if isinstance(r, Mapping)]
    except (ValueError, TypeError):
        pass
    return []


def save_software_inventory(software: List[Mapping[str, Any]]) -> str:
    """Serialize software records for storage in ChromaDB metadata."""
    return json.dumps([dict(r) for r in software], ensure_ascii=False)


def is_valid_cpe(cpe: Any) -> bool:
    """Return True only for a complete, escaped CPE 2.3 formatted string."""
    return parse_cpe(cpe) is not None


def inventory_is_verified(asset_metadata: Mapping[str, Any]) -> bool:
    """Require explicit completeness attestation and no discarded bad records.

    This is not inferred from nonempty inventory. Use together with
    collect_asset_cpes when passing inventory_verified to the evaluator.
    Native True and the canonical CSV string 'true' are explicit attestations;
    other strings and numeric 1 are not. Existing records with no verification
    metadata remain unverified.
    """
    flag = asset_metadata.get(INVENTORY_VERIFIED_FIELD)
    if flag is not True and not (isinstance(flag, str) and flag == "true"):
        return False
    legacy = str(asset_metadata.get("cpe") or "").strip()
    if legacy and legacy.lower() not in ("unknown", "n/a", "none", "-") and not is_valid_cpe(legacy):
        return False
    raw = asset_metadata.get(SOFTWARE_INVENTORY_FIELD, "")
    if str(raw).lower() == "unknown":
        return False
    if raw is not None and raw != "":
        try:
            records = json.loads(raw)
        except (ValueError, TypeError):
            return False
        if not isinstance(records, list) or any(
            not isinstance(record, Mapping) or not is_valid_cpe(record.get("cpe"))
            for record in records
        ):
            return False
    return bool(collect_asset_cpes(asset_metadata))


def collect_asset_cpes(asset_metadata: Mapping[str, Any]) -> List[str]:
    """Return all usable CPEs for the asset: legacy `cpe` + software inventory.

    Duplicates are removed while preserving order. Invalid or blank CPEs are
    skipped. This does not attest inventory completeness: callers must use
    inventory_is_verified rather than infer verification from this list.
    """
    seen: set = set()
    cpes: List[str] = []

    legacy = str(asset_metadata.get("cpe") or "").strip()
    if legacy and legacy.lower() != "unknown" and is_valid_cpe(legacy):
        seen.add(legacy)
        cpes.append(legacy)

    for record in load_software_inventory(asset_metadata):
        cpe = str(record.get("cpe") or "").strip()
        if cpe and cpe not in seen and is_valid_cpe(cpe):
            seen.add(cpe)
            cpes.append(cpe)

    return cpes

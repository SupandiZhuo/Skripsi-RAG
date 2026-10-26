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


def is_valid_cpe(cpe: str) -> bool:
    """Return True if `cpe` is a parseable CPE 2.3 URI."""
    return parse_cpe(cpe) is not None


def collect_asset_cpes(asset_metadata: Mapping[str, Any]) -> List[str]:
    """Return all usable CPEs for the asset: legacy `cpe` + software inventory.

    Duplicates are removed while preserving order. Invalid or blank CPEs are
    skipped, but the legacy `cpe` field is included even if it is "unknown"
    when it cannot be parsed? No — only valid CPEs are returned so the
    applicability evaluator can reason about them. Missing/invalid inventory
    is surfaced via the UI, not silently fabricated here.
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

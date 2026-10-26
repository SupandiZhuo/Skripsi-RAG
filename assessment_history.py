"""Minimal assessment history persistence (Phase 5).

Stores completed vulnerability assessments as JSON lines/objects in
`data/assessment_history.json` so the Analysis History page can display them
without introducing a database. The file is created on first save if it does
not exist.

This module is intentionally small: it records only assessment metadata,
scores, and references to evidence — not full ChromaDB documents or LLM
internals, to keep the file bounded and human-readable.
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

HISTORY_PATH = "data/assessment_history.json"


@dataclass
class AssessmentRecord:
    timestamp: str
    cve_id: str
    asset_id: str
    asset_name: str
    base_score: float
    context_score: Optional[float]
    priority: Optional[str]
    llm_priority: Optional[str]
    vector_string: str
    modified_vector: str
    retrieval_confidence: Optional[float]
    evidence: Dict[str, Any]
    # Unique id for reliable deletion (added Phase 4; missing in legacy records).
    id: Optional[str] = None
    # Phase A: CPE-based applicability (optional for backward compatibility)
    applicability: Optional[str] = None
    applicability_explanation: Optional[str] = None
    applicability_evidence: Optional[List[Dict[str, Any]]] = None
    # Phase C: deterministic organizational priority evidence
    priority_status: Optional[str] = None
    priority_score: Optional[int] = None
    priority_explanation: Optional[str] = None
    priority_missing_evidence: Optional[List[str]] = None
    priority_unknown_factors: Optional[List[str]] = None
    priority_contributions: Optional[List[Dict[str, Any]]] = None
    graph_dependencies: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history(path: str = HISTORY_PATH) -> List[AssessmentRecord]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable history file should not crash the UI.
        return []
    if not isinstance(data, list):
        return []
    records = []
    for item in data:
        try:
            records.append(AssessmentRecord(**item))
        except TypeError:
            # Skip malformed records rather than crashing the history page.
            continue
    return records


def save_assessment(record: AssessmentRecord, path: str = HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    history = load_history(path)
    history.append(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in history], handle, indent=2)


def record_from_analysis(
    cve_id: str,
    asset_metadata: Mapping[str, Any],
    base_score: float,
    context_score: Optional[float],
    priority: Optional[str],
    llm_priority: Optional[str],
    vector_string: str,
    modified_vector: str,
    retrieval_confidence: Optional[float],
    evidence: Optional[Mapping[str, Any]] = None,
    applicability: Optional[str] = None,
    applicability_explanation: Optional[str] = None,
    applicability_evidence: Optional[List[Dict[str, Any]]] = None,
    priority_result: Optional[Any] = None,
    graph_dependencies: Optional[Mapping[str, Any]] = None,
) -> AssessmentRecord:
    return AssessmentRecord(
        id=str(uuid.uuid4()),
        timestamp=_utc_now_iso(),
        cve_id=cve_id,
        asset_id=str(asset_metadata.get("asset_id") or ""),
        asset_name=str(asset_metadata.get("asset_name") or ""),
        base_score=base_score,
        context_score=context_score,
        priority=priority,
        llm_priority=llm_priority,
        vector_string=vector_string,
        modified_vector=modified_vector,
        retrieval_confidence=retrieval_confidence,
        evidence=dict(evidence) if evidence else {},
        applicability=applicability,
        applicability_explanation=applicability_explanation,
        applicability_evidence=applicability_evidence or [],
        priority_status=priority_result.status if priority_result else None,
        priority_score=priority_result.score if priority_result else None,
        priority_explanation=priority_result.explanation if priority_result else None,
        priority_missing_evidence=list(priority_result.missing_evidence) if priority_result else None,
        priority_unknown_factors=list(priority_result.unknown_factors) if priority_result else None,
        priority_contributions=[{
            "rule": c.rule,
            "factor": c.factor,
            "points": c.points,
            "max_points": c.max_points,
            "evidence_value": c.evidence_value,
            "evidence_source": c.evidence_source,
        } for c in (priority_result.rule_contributions if priority_result else ())],
        graph_dependencies=dict(graph_dependencies) if graph_dependencies else None,
    )


def clear_history(path: str = HISTORY_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)


def _record_identity(r: AssessmentRecord) -> tuple:
    """Value-based identity for deduplication/deletion."""
    return (r.timestamp, r.cve_id, r.asset_id, r.base_score, r.context_score)


def delete_record(record: AssessmentRecord, path: str = HISTORY_PATH) -> None:
    """Remove a single record from the history file and re-save.

    Uses the record's unique id when available; falls back to value-based
    identity only for legacy records that lack an id.
    """
    history = load_history(path)
    if record.id:
        history = [r for r in history if r.id != record.id]
    else:
        target = _record_identity(record)
        history = [r for r in history if _record_identity(r) != target]
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in history], handle, indent=2)

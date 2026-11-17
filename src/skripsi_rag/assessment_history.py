"""Persist assessment traces as a JSON array, including evidence and model payloads.

Writes replace the file atomically and reject unreadable or malformed history.
This single-user store does not coordinate concurrent writers.
"""

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

HISTORY_PATH = "data/assessment_history.json"


class HistoryReadError(ValueError):
    """Existing history cannot be loaded without losing data."""


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
    retrieval_confidence: Optional[float]  # Legacy name for retrieval relevance, not probability.
    evidence: Dict[str, Any]
    # Older history records may not have an ID.
    id: Optional[str] = None
    # Optional fields allow older history records to load.
    applicability: Optional[str] = None
    applicability_explanation: Optional[str] = None
    applicability_evidence: Optional[List[Dict[str, Any]]] = None
    priority_status: Optional[str] = None
    priority_score: Optional[int] = None
    priority_explanation: Optional[str] = None
    priority_missing_evidence: Optional[List[str]] = None
    priority_unknown_factors: Optional[List[str]] = None
    priority_contributions: Optional[List[Dict[str, Any]]] = None
    graph_dependencies: Optional[Dict[str, Any]] = None
    investigation_priority: Optional[str] = None
    rubric_version: Optional[str] = None
    model_output: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_history(path: str = HISTORY_PATH, *, strict: bool = False) -> List[AssessmentRecord]:
    """Read compatible records; strict mode rejects errors and discarded entries."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        if strict:
            raise HistoryReadError(f"Assessment history cannot be read ({type(exc).__name__}).") from exc
        return []
    if not isinstance(data, list):
        if strict:
            raise HistoryReadError("Assessment history must contain a JSON array.")
        return []
    records = []
    for item in data:
        try:
            records.append(AssessmentRecord(**item))
        except TypeError as exc:
            if strict:
                raise HistoryReadError("Assessment history contains a malformed record.") from exc
            continue
    return records


def _write_history(history: List[AssessmentRecord], path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".assessment-history-", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump([asdict(record) for record in history], handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass


def save_assessment(record: AssessmentRecord, path: str = HISTORY_PATH) -> None:
    history = load_history(path, strict=True)
    history.append(record)
    _write_history(history, path)


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
    model_output: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
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
        investigation_priority=getattr(priority_result, "investigation_priority", None),
        rubric_version=getattr(priority_result, "rubric_version", None),
        model_output=dict(model_output) if model_output else None,
        provenance=dict(provenance) if provenance else None,
    )


def clear_history(path: str = HISTORY_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)


def _record_identity(r: AssessmentRecord) -> tuple:
    """Fallback deletion identity for records without IDs."""
    return (r.timestamp, r.cve_id, r.asset_id, r.base_score, r.context_score)


def delete_record(record: AssessmentRecord, path: str = HISTORY_PATH) -> None:
    """Remove a single record from the history file and re-save.

    Uses the record's unique id when available; falls back to value-based
    identity only for legacy records that lack an id.
    """
    history = load_history(path, strict=True)
    if record.id:
        history = [r for r in history if r.id != record.id]
    else:
        target = _record_identity(record)
        history = [r for r in history if _record_identity(r) != target]
    _write_history(history, path)

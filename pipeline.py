"""UI-independent, auditable E0-E3 orchestration. No implicit cloud consent.

E0 = NVD Base severity only; E1 = LLM with CVE/Base only; E2 = direct
minimized asset profile; E3 = same profile + scoped, attributed document RAG.
The returned JSON-safe record is an experiment trace, not an expert label.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from collections.abc import Mapping

from asset_inventory import collect_asset_cpes
from cve_applicability import determine_applicability
from cvss_engine import build_environmental_vector, calculate_cvss, validate_base_vector
from llm_assessment import DEFAULT_MODEL, ContextualAssessment, analyze_with_llm
from organizational_priority import calculate_organizational_priority

# Deliberate allowlist: hidden labels, contact/IP information and arbitrary
# scenario payloads must never become model input. Policy CIA fields are
# handled separately by build_model_context's provenance/leakage policy.
_ASSET_FIELDS = {
    "asset_id", "asset_name", "environment", "internet_exposed", "business_criticality",
    "data_sensitivity", "data_classification", "network_zone", "asset_type",
    "business_function", "vendor", "software_name", "software_version", "cpe",
    "software_inventory_json", "inventory_verified", "source", "source_id",
    "source_reference", "source_version", "last_updated", "is_synthetic", "schema_version",
    "confidentiality_requirement", "integrity_requirement", "availability_requirement",
    "policy_baseline_source", "policy_baseline_verified", "baseline_origin",
}


def json_safe(value):
    """Strict JSON values; no NaN/Infinity, SDK objects or exception serialization."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Not a JSON-safe experiment value: {type(value).__name__}")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def run_assessment(nvd_data, asset_metadata, api_key: str = "", experiment_id: str = "E3",
                   context_store=None, model: str = DEFAULT_MODEL, top_k: int = 3,
                   relevance_threshold: float = 0.15, *, cloud_consent: bool = False,
                   scenario_id: str | None = None, run_id: str | None = None,
                   vector_override: str | None = None, override_reason: str = "",
                   graph_dependencies=None, analyzer=None, retrieval_query: str | None = None,
                   embedding_model: str | None = None, config_hash: str | None = None) -> dict:
    """Return baseline/predictions/provenance/failures without importing Streamlit.

    Unknown applicability never produces remediation priority; any contextual
    score is provisional. Not Applicable gates contextual assessment entirely.
    E1 never reads asset evidence for applicability or deterministic priority.
    E3 retrieval errors stop the run rather than silently becoming E2.
    Callers must explicitly consent to sending minimized context to a cloud LLM.
    Inject ``analyzer`` for offline contract tests; no model is called for E0.
    """
    if experiment_id not in {"E0", "E1", "E2", "E3"}:
        raise ValueError("experiment_id must be E0, E1, E2, or E3.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if isinstance(relevance_threshold, bool) or not isinstance(relevance_threshold, (float, int)) or not math.isfinite(relevance_threshold) or not 0 <= relevance_threshold <= 1:
        raise ValueError("relevance_threshold must be finite and between 0 and 1.")
    start = time.monotonic()
    nvd = dict(nvd_data or {})
    asset = {key: value for key, value in dict(asset_metadata or {}).items() if key in _ASSET_FIELDS}
    import llm_assessment
    import organizational_priority
    prompt = getattr(llm_assessment, "TEMPLATE", "")
    rubric_hash = hashlib.sha256(__import__("pathlib").Path(organizational_priority.__file__).read_bytes()).hexdigest()
    record = {
        "record_version": "assessment-v1", "run_id": run_id or str(uuid.uuid4()),
        "experiment_id": experiment_id, "scenario_id": scenario_id,
        "cve_id": nvd.get("cve_id"), "asset_id": asset.get("asset_id"),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "pending",
        "base_vector": nvd.get("vector_string", ""), "base_score": None,
        "nvd_base_score": nvd.get("base_score"), "technical_severity": None,
        "vector_origin": "nvd", "override_reason": None, "main_test_set_eligible": True,
        "predicted_cr": None, "predicted_ir": None, "predicted_ar": None,
        "llm_priority": None, "deterministic_priority": None, "investigation_priority": None,
        "computed_environmental_vector": None, "computed_environmental_score": None,
        "environmental_severity": None, "contextual_score_status": "not_applicable",
        "applicability": None, "priority_result": None, "model_input": None,
        "model_output": None, "retrieval": None, "retrieved_document_ids": [],
        "retrieval_scores": [], "failures": [], "warnings": [],
        "nvd_snapshot_id": nvd.get("nvd_snapshot_id"),
        "nvd_snapshot_sha256": nvd.get("nvd_snapshot_sha256"),
        "nvd_retrieved_at": nvd.get("nvd_retrieved_at"),
        "nvd_cvss_source": nvd.get("nvd_cvss_source"),
        "nvd_metric_selection_policy": nvd.get("metric_selection_policy"),
        "nvd_retry_count": nvd.get("nvd_retry_count", 0),
        "asset_snapshot_id": _hash(asset), "score_method": "cvss-library-v3.1",
        "embedding_model": embedding_model if experiment_id == "E3" else None,
        "top_k": top_k if experiment_id == "E3" else None,
        "relevance_threshold": relevance_threshold if experiment_id == "E3" else None,
        "llm_provider": "google_genai" if experiment_id != "E0" else None,
        "llm_model": model if experiment_id != "E0" else None,
        "temperature": 0 if experiment_id != "E0" else None,
        "prompt_version": getattr(llm_assessment, "PROMPT_VERSION", "sha256-addressed"),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "rubric_version": getattr(organizational_priority, "RUBRIC_VERSION", "sha256:" + rubric_hash),
        "rubric_sha256": rubric_hash, "config_hash": config_hash,
        "inputs": {"description": nvd.get("description", ""), "asset_metadata": asset,
                   "graph_dependencies": json_safe(graph_dependencies or {}),
                   "nvd_configurations": json_safe(nvd.get("configurations", [])),
                   "nvd_source_url": nvd.get("source_url")},
    }

    def finish(status, error_type=None, stage=None):
        record["status"] = status
        if error_type:
            record["failures"].append({"error_type": error_type, "stage": stage})
        record["latency_ms"] = round((time.monotonic() - start) * 1000, 3)
        return json_safe(record)

    vector = nvd.get("vector_string", "")
    if vector_override is not None:
        if not override_reason.strip():
            return finish("failed", "override_reason_required", "baseline")
        vector = vector_override
        record.update(vector_origin="manual_override", override_reason=override_reason,
                      main_test_set_eligible=False)
    elif nvd.get("vector_origin", "nvd") != "nvd":
        return finish("failed", "unattributed_vector_origin", "baseline")
    try:
        validate_base_vector(vector)
        baseline = calculate_cvss(vector)
    except (ValueError, TypeError):
        return finish("failed", "invalid_or_unsupported_base_vector", "baseline")
    record.update(base_vector=vector, base_score=baseline["base_score"],
                  technical_severity=baseline["technical_severity"])
    if nvd.get("base_score") is not None and nvd["base_score"] != baseline["base_score"]:
        record["warnings"].append("nvd_score_differs_from_deterministic_base")
    if experiment_id == "E0":
        return finish("complete")
    if cloud_consent is not True:
        return finish("blocked", "cloud_consent_required", "privacy")
    if analyzer is None and not api_key:
        return finish("blocked", "api_key_required", "model")

    applicability = None
    if experiment_id in {"E2", "E3"}:
        try:
            verified = asset.get("inventory_verified") is True or asset.get("inventory_verified") == "true"
            applicability = determine_applicability(collect_asset_cpes(asset),
                {"configurations": nvd.get("configurations", [])}, inventory_verified=verified)
            record["applicability"] = json_safe(applicability)
            priority = calculate_organizational_priority(asset, base_score=baseline["base_score"],
                applicability=applicability, graph_dependencies=graph_dependencies or {}, cve_id=nvd.get("cve_id"))
            record["priority_result"] = json_safe(priority)
            record["deterministic_priority"] = priority.priority
            record["investigation_priority"] = getattr(priority, "investigation_priority", None)
            if json_safe(applicability.status) == "Not Applicable":
                return finish("not_applicable")
        except Exception as exc:
            record["failures"].append({"error_type": "applicability_failure", "stage": "applicability",
                                        "exception_class": type(exc).__name__})
            return finish("failed")
    context = {}
    if experiment_id in {"E2", "E3"}:
        evidence = []
        if experiment_id == "E3":
            if context_store is None:
                return finish("failed", "retrieval_unavailable", "retrieval")
            try:
                from rag_retrieval import retrieve_document_context
                retrieval = retrieve_document_context(context_store,
                    retrieval_query if retrieval_query is not None else nvd.get("description", ""),
                    asset.get("asset_id", ""), top_k=top_k, relevance_threshold=relevance_threshold)
                record["retrieval"] = json_safe(retrieval)
                if retrieval.error:
                    return finish("failed", "retrieval_failure", "retrieval")
                evidence = retrieval.accepted
                record["retrieved_document_ids"] = [item.metadata["document_id"] for item in evidence]
                record["retrieval_scores"] = [item.relevance_score for item in evidence]
                if not evidence:
                    record["warnings"].append("no_accepted_document_evidence")
            except Exception as exc:
                record["failures"].append({"error_type": "retrieval_failure", "stage": "retrieval",
                                            "exception_class": type(exc).__name__})
                return finish("failed")
        try:
            from llm_assessment import build_model_context
            context = build_model_context(asset, retrieved_evidence=evidence)
        except Exception as exc:
            record["failures"].append({"error_type": "context_validation_failure", "stage": "context",
                                        "exception_class": type(exc).__name__})
            return finish("failed")
    model_input = {"cve_desc": nvd.get("description", "")[:20000], "original_vector": vector,
                   "asset_context": json_safe(context)}
    record["model_input"] = model_input
    record["model_input_sha256"] = _hash(model_input)
    try:
        raw = (analyzer or analyze_with_llm)(api_key=api_key, model=model, **model_input)
    except Exception as exc:
        cause = exc.__cause__ or exc
        error_type = "schema_failure" if isinstance(cause, (ValueError, TypeError)) else "model_api_failure"
        record["failures"].append({"error_type": error_type, "stage": "model",
                                    "exception_class": type(cause).__name__})
        return finish("failed")
    try:
        # Revalidate even an unchecked model_construct() from a provider/adapter.
        assessment = ContextualAssessment.model_validate(raw.model_dump() if isinstance(raw, ContextualAssessment) else raw)
        valid_ids = set()
        if context.get("asset_evidence_id"):
            valid_ids.add(context["asset_evidence_id"])
        valid_ids.update(item["evidence_id"] for item in context.get("retrieved_evidence", []))
        for metric in ("CR", "IR", "AR"):
            if set(getattr(assessment.metric_reasons, metric).evidence_ids) - valid_ids:
                raise ValueError("Assessment cites evidence not supplied to the model.")
        record["model_output"] = json_safe(assessment)
        record.update(predicted_cr=json_safe(assessment.confidentiality_requirement),
                      predicted_ir=json_safe(assessment.integrity_requirement),
                      predicted_ar=json_safe(assessment.availability_requirement),
                      llm_priority=json_safe(assessment.organizational_priority))
        environmental_vector = build_environmental_vector(vector, assessment)
        environmental = calculate_cvss(environmental_vector)
        record.update(computed_environmental_vector=environmental_vector,
                      computed_environmental_score=environmental["environmental_score"],
                      environmental_severity=environmental["environmental_severity"])
    except (ValueError, TypeError):
        return finish("failed", "schema_failure", "model_output")
    if applicability is not None:
        priority = calculate_organizational_priority(asset, base_score=baseline["base_score"],
            context_score=environmental["environmental_score"], applicability=applicability,
            graph_dependencies=graph_dependencies or {}, cve_id=nvd.get("cve_id"))
        record.update(priority_result=json_safe(priority), deterministic_priority=priority.priority,
                      investigation_priority=getattr(priority, "investigation_priority", None))
        if json_safe(applicability.status) != "Confirmed":
            record["contextual_score_status"] = "provisional_unknown_applicability"
            return finish("needs_verification")
    record["contextual_score_status"] = "computed" if applicability else "unverified_no_asset_context"
    return finish("complete")

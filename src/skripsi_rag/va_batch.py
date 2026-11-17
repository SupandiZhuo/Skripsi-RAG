"""Operational VA batch triage, independent of Streamlit and experiment splits."""
from datetime import datetime, timezone
from collections.abc import Mapping
import uuid

_ASSESSMENT_STATUSES = {"complete", "failed", "blocked", "not_applicable", "needs_verification"}


def run_va_batch(report, target_assets, assets, *, api_key="", model=None,
                 context_store=None, cloud_consent=False, relevance_threshold=0.15,
                 top_k=3, fetcher=None, assessor=None, on_progress=None):
    """Assess each distinct target/component/asset/CVE; keep every occurrence.

    Acunetix finding data remains local provenance, never LLM input. Scanner
    scores never substitute NVD metrics or confirm software applicability.
    Adapter failures retain exception classes, not credential-bearing messages.
    """
    actionable = any(f["import_status"] == "ready" and f["cve_ids"]
                     and assets.get(target_assets.get(f["target"])) for f in report["findings"])
    if actionable and cloud_consent is not True:
        raise ValueError("Explicit cloud consent is required before batch analysis.")
    from skripsi_rag.nvd_client import fetch_cve_from_nvd
    from skripsi_rag.pipeline import run_assessment
    from skripsi_rag.llm_assessment import DEFAULT_MODEL
    fetcher = fetcher or fetch_cve_from_nvd
    assessor = assessor or run_assessment
    batch = {"batch_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
             "report": report, "target_assets": dict(target_assets), "cases": []}
    nvd_cache = {}
    assessment_cache = {}
    for index, finding in enumerate(report["findings"]):
        target = finding["target"]
        asset_id = target_assets.get(target, "")
        asset = assets.get(asset_id)
        for cve in finding["cve_ids"] or [None]:
            case = {"finding_id": finding["finding_id"], "cve_id": cve, "asset_id": asset_id,
                    "status": None, "assessment": None, "issues": list(finding["issues"]), "failures": []}
            batch["cases"].append(case)
            if finding["import_status"] != "ready":
                case["status"] = finding["import_status"]
                continue
            if not cve:
                case["status"] = "no_cve"
                continue
            if not asset or asset.get("asset_id") != asset_id:
                case["status"] = "unmapped_asset"
                continue
            if cve not in nvd_cache:
                try:
                    response = fetcher(cve)
                    if not isinstance(response, (tuple, list)) or len(response) != 2:
                        raise ValueError("NVD adapter must return a data/message pair.")
                    nvd = response[0]
                    if nvd is not None and not isinstance(nvd, Mapping):
                        raise ValueError("NVD data must be a mapping.")
                    if nvd is not None and nvd.get("cve_id") != cve:
                        raise ValueError("NVD response does not match the requested CVE.")
                    failure = None if nvd is not None else {"stage": "nvd", "error_type": "not_available"}
                    nvd_cache[cve] = (nvd, failure)
                except Exception as exc:
                    nvd_cache[cve] = (
                        None, {"stage": "nvd", "error_type": "lookup_failure",
                               "exception_class": type(exc).__name__},
                    )
            nvd, failure = nvd_cache[cve]
            if nvd is None:
                case["status"] = "nvd_unavailable"
                case["failures"].append(failure)
                continue
            if not nvd.get("vector_string"):
                case["status"] = "unsupported_or_missing_nvd_cvss"
                case["failures"].append({"stage": "baseline", "error_type": "unsupported_or_missing_nvd_cvss"})
                continue
            # Reusing a CVE assessment across targets or scanner modules can mix asset context.
            key = (target, asset_id, finding["module"] or finding["name"], cve)
            if key not in assessment_cache:
                try:
                    assessment = assessor(nvd, asset, api_key=api_key,
                        experiment_id="E3", context_store=context_store, model=model or DEFAULT_MODEL,
                        relevance_threshold=relevance_threshold, top_k=top_k, cloud_consent=True)
                    if (not isinstance(assessment, Mapping)
                            or not isinstance(assessment.get("status"), str)
                            or assessment["status"] not in _ASSESSMENT_STATUSES
                            or (assessment.get("applicability") is not None
                                and not isinstance(assessment["applicability"], Mapping))):
                        raise ValueError("Assessment adapter returned an invalid terminal result.")
                    assessment_cache[key] = assessment
                except Exception as exc:
                    assessment_cache[key] = {"status": "failed", "failures": [
                        {"stage": "assessment", "error_type": "assessment_failure",
                         "exception_class": type(exc).__name__}]}
            assessment = assessment_cache[key]
            case["assessment"] = assessment
            case["status"] = assessment["status"]
            if isinstance(assessment.get("failures"), list):
                case["failures"] = [{key: failure[key] for key in ("stage", "error_type", "exception_class") if key in failure}
                                    for failure in assessment["failures"] if isinstance(failure, Mapping)]
            if (assessment.get("applicability") or {}).get("status") == "Not Applicable":
                case["status"] = "scanner_inventory_conflict"
                case["issues"].append("Scanner reports this CVE but inventory matching says Not Applicable; review both sources.")
        if on_progress:
            on_progress(index + 1, len(report["findings"]))
    return batch


def batch_rows(batch):
    findings = {finding["finding_id"]: finding for finding in batch["report"]["findings"]}
    rows = []
    for case in batch["cases"]:
        finding = findings[case["finding_id"]]
        assessment = case["assessment"] or {}
        rows.append({"Row": finding["row_number"], "Finding": finding["name"], "Target": finding["target"],
                     "Location": finding["affects"], "Asset": case["asset_id"], "CVE": case["cve_id"] or "",
                     "Scanner Severity": finding["scanner_severity"], "Scanner CVSS": finding["scanner_cvss_score"],
                     "NVD Base CVSS": assessment.get("base_score"), "Technical Severity": assessment.get("technical_severity"),
                     "Environmental CVSS": assessment.get("computed_environmental_score"),
                     "Environmental Severity": assessment.get("environmental_severity"),
                     "Contextual Score Status": assessment.get("contextual_score_status"),
                     "Priority": assessment.get("deterministic_priority"),
                     "Investigation Priority": assessment.get("investigation_priority"),
                     "Applicability": (assessment.get("applicability") or {}).get("status"),
                     "Status": case["status"],
                     "Issues": " ".join(case["issues"] + [
                         f"{failure.get('stage', 'assessment')}: {failure.get('error_type', 'failed')}"
                         + (f" ({failure['exception_class']})" if failure.get("exception_class") else "")
                         for failure in case.get("failures", [])])})
    return rows


def case_history_record(batch, case, asset):
    """Reuse the existing history schema; include scanner/report provenance."""
    from skripsi_rag.assessment_history import AssessmentRecord
    assessment = case["assessment"]
    priority = assessment.get("priority_result") or {}
    applicability = assessment.get("applicability") or {}
    retrieval = assessment.get("retrieval") or {}
    finding = next(entry for entry in batch["report"]["findings"] if entry["finding_id"] == case["finding_id"])
    occurrence_ids = {candidate_case["finding_id"] for candidate_case in batch["cases"]
                      if candidate_case["assessment"] and candidate_case["assessment"].get("run_id") == assessment["run_id"]}
    occurrences = [entry for entry in batch["report"]["findings"] if entry["finding_id"] in occurrence_ids]
    return AssessmentRecord(
        id=f"{batch['batch_id']}:{assessment['run_id']}", timestamp=assessment["created_at"],
        cve_id=case["cve_id"], asset_id=case["asset_id"], asset_name=asset.get("asset_name", ""),
        base_score=assessment["base_score"], context_score=assessment.get("computed_environmental_score"),
        priority=assessment.get("deterministic_priority"), llm_priority=assessment.get("llm_priority"),
        vector_string=assessment["base_vector"], modified_vector=assessment.get("computed_environmental_vector") or "",
        retrieval_confidence=max(assessment.get("retrieval_scores") or [], default=None),
        evidence={"nvd": {"description": assessment["inputs"]["description"], "vector_string": assessment["base_vector"],
                          "nvd_snapshot_id": assessment.get("nvd_snapshot_id"), "nvd_cvss_source": assessment.get("nvd_cvss_source")},
                  "document_retrieval": retrieval, "model_context": (assessment.get("model_input") or {}).get("asset_context", {}),
                  "llm_justification": (assessment.get("model_output") or {}).get("justification", ""),
                  "scanner_finding": finding, "scanner_occurrences": occurrences},
        applicability=applicability.get("status"), applicability_explanation=applicability.get("explanation"),
        applicability_evidence=applicability.get("evidence", []),
        priority_status=priority.get("status"), priority_score=priority.get("score"),
        priority_explanation=priority.get("explanation"), priority_missing_evidence=priority.get("missing_evidence"),
        priority_unknown_factors=priority.get("unknown_factors"), priority_contributions=priority.get("rule_contributions"),
        investigation_priority=assessment.get("investigation_priority"), rubric_version=assessment.get("rubric_version"),
        model_output=assessment.get("model_output"),
        provenance={"batch_id": batch["batch_id"], "report_sha256": batch["report"]["report_id"],
                    "source": "Acunetix CSV", "finding_id": finding["finding_id"],
                    "case_status": case["status"], "case_issues": case["issues"],
                    "case_failures": case.get("failures", []),
                    "model_input": assessment.get("model_input"), "model": assessment.get("llm_model"),
                    "contextual_score_status": assessment.get("contextual_score_status"), "assessment": assessment})

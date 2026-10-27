"""Pure-Python evaluation helpers. No labels are invented or inferred.

Unknown reference labels are excluded explicitly. Unknown/missing predictions
remain errors against known references (and reduce coverage), not successes.
Ordinal kappa is only defined on jointly observed ordinal labels, with its
coverage reported. X in CR/IR/AR means Not Defined, not a fourth ordinal level.
"""
from __future__ import annotations

import math
from collections import defaultdict


def classification_metrics(reference, predicted, labels, *, ordinal_labels=None):
    if len(reference) != len(predicted):
        raise ValueError("Reference/prediction lengths must match.")
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("labels must contain distinct known classes.")
    known = [(a, b) for a, b in zip(reference, predicted) if a in labels]
    matrix = {a: {b: 0 for b in labels + ["__abstain__"]} for a in labels}
    for actual, prediction in known:
        matrix[actual][prediction if prediction in labels else "__abstain__"] += 1
    n = len(known)
    f1 = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in labels if a != label)
        fn = sum(matrix[label][b] for b in matrix[label] if b != label)
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result = {"n_total": len(reference), "n_labeled": n,
              "excluded_unknown_reference": len(reference) - n,
              "coverage": sum(b in labels for _, b in known) / n if n else None,
              "accuracy": sum(a == b for a, b in known) / n if n else None,
              "macro_f1": sum(f1.values()) / len(labels) if n else None,
              "per_class_f1": f1 if n else {label: None for label in labels},
              "confusion_matrix": matrix}
    if ordinal_labels:
        order = list(ordinal_labels)
        if len(order) < 2 or len(set(order)) != len(order):
            raise ValueError("Ordinal labels must contain at least two distinct values.")
        pairs = [(order.index(a), order.index(b)) for a, b in known if a in order and b in order]
        count = len(pairs)
        result["kappa_n"] = count
        result["kappa_coverage"] = count / n if n else None
        result["weighted_kappa"] = None
        if count:
            size = len(order)
            rows = [sum(a == i for a, _ in pairs) for i in range(size)]
            cols = [sum(b == i for _, b in pairs) for i in range(size)]
            observed = sum((a - b) ** 2 for a, b in pairs) / count
            expected = sum((i - j) ** 2 * rows[i] * cols[j]
                           for i in range(size) for j in range(size)) / count ** 2
            result["weighted_kappa"] = 1 - observed / expected if expected else None
    return result


def retrieval_metrics(retrieved_ids, relevant_ids, k=3):
    """Document-level IDs, deduplicated in retrieval order; unlabeled => N/A."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    selected = list(dict.fromkeys(retrieved_ids))[:k]
    relevant = set(relevant_ids)
    hits = [i + 1 for i, document in enumerate(selected) if document in relevant]
    return {"n_relevant": len(relevant), "n_retrieved": len(selected),
            "hit_at_1": int(bool(selected and selected[0] in relevant)) if relevant else None,
            "hit_at_k": int(bool(hits)) if relevant else None,
            "recall_at_k": len(hits) / len(relevant) if relevant else None,
            "context_precision": len(hits) / len(selected) if relevant and selected else (0.0 if relevant else None),
            "mrr": 1 / hits[0] if hits else (0.0 if relevant else None)}


def validate_splits(scenarios):
    """Reject scenario-ID or CVE/asset-pair leakage across declared splits."""
    seen = {}
    pairs = {}
    for scenario in scenarios:
        split = scenario.get("split", "unspecified")
        scenario_id = scenario["scenario_id"]
        pair = (scenario["cve_id"], scenario["asset_id"])
        for key, table in ((scenario_id, seen), (pair, pairs)):
            if key in table and table[key] != split:
                raise ValueError(f"Split leakage detected for {key!r}.")
            table[key] = split


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def ranking_metrics(predicted_scores, relevance, k=10):
    """Higher score/relevance => earlier. Use only within the same asset/situation.

    No automatic mapping from generic NVD severity to P1-P4 is made.
    """
    if len(predicted_scores) != len(relevance) or k < 1:
        raise ValueError("Ranking inputs must be paired, with k >= 1.")
    if len(predicted_scores) < 2:
        return {"n": len(predicted_scores), "spearman": None, "ndcg_at_k": None}
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*predicted_scores, *relevance]):
        raise ValueError("Ranking inputs must be finite numbers.")
    if any(v < 0 for v in relevance):
        raise ValueError("NDCG relevance must be nonnegative.")
    x, y = _ranks(predicted_scores), _ranks(relevance)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
    scale = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    ideal = sorted(relevance, reverse=True)[:k]
    ideal_dcg = sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(ideal))
    # Average gain of tied score groups prevents arbitrary input order deciding NDCG.
    gains = {}
    for score in set(predicted_scores):
        group = [2 ** rel - 1 for score_i, rel in zip(predicted_scores, relevance) if score_i == score]
        gains[score] = sum(group) / len(group)
    ordered = sorted(predicted_scores, reverse=True)[:k]
    dcg = sum(gains[score] / math.log2(i + 2) for i, score in enumerate(ordered))
    return {"n": len(x), "spearman": covariance / scale if scale else None,
            "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None}


def summarize_runs(records, labels=None):
    """Join external adjudicated labels by scenario ID only AFTER prediction.

    Labels are optional. No label means N/A, not a synthetic ground truth.
    Environmental MAE is computed from expert CR/IR/AR and the identical Base
    vector; it is not a true-risk error. Ranking requires explicit rank_group.
    """
    labels = labels or {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["experiment_id"]].append(record)
    reports = {}
    for experiment, runs in sorted(grouped.items()):
        failures = [f for run in runs for f in run.get("failures", [])]
        report = {"n_runs": len(runs), "n_scenarios": len({r.get("scenario_id") for r in runs}),
                  "status_counts": {s: sum(r.get("status") == s for r in runs) for s in sorted({r.get("status", "unknown") for r in runs})},
                  "failure_rate": sum(bool(r.get("failures")) for r in runs) / len(runs),
                  "schema_failure_rate": sum(any(f.get("error_type") == "schema_failure" for f in r.get("failures", [])) for r in runs) / len(runs),
                  "api_failure_rate": sum(any(f.get("error_type") == "model_api_failure" for f in r.get("failures", [])) for r in runs) / len(runs),
                  "snapshot_failure_rate": sum(any(f.get("error_type") == "nvd_snapshot_failure" for f in r.get("failures", [])) for r in runs) / len(runs),
                  "nvd_retries": sum(r.get("nvd_retry_count", 0) for r in runs),
                  "mean_latency_ms": sum(r.get("latency_ms", 0) for r in runs) / len(runs),
                  "failure_types": {kind: sum(f.get("error_type") == kind for f in failures) for kind in sorted({f["error_type"] for f in failures})},
                  "labels_status": "external_reference_provided" if labels else "no_reference_labels",
                  "repeats_policy": "run-level metrics; no independence/significance claim"}
        applicable = [labels.get(r.get("scenario_id"), {}).get("applicability") for r in runs]
        predicted = [(r.get("applicability") or {}).get("status") for r in runs]
        report["applicability"] = classification_metrics(applicable, predicted, ["Confirmed", "Not Applicable"]) if experiment in {"E2", "E3"} else {"status": "not_applicable"}
        report["false_negative_scenario_ids"] = sorted({r.get("scenario_id") for r, actual, prediction in zip(runs, applicable, predicted)
                                                        if actual == "Confirmed" and prediction == "Not Applicable"})
        report["confirmed_reference_abstentions"] = sum(actual == "Confirmed" and prediction not in {"Confirmed", "Not Applicable"}
                                                        for actual, prediction in zip(applicable, predicted)) if experiment in {"E2", "E3"} else None
        report["unknown_applicability_rate"] = sum(v in (None, "Unknown") for v in predicted) / len(runs) if experiment in {"E2", "E3"} else None
        report["crirar"] = {"status": "not_applicable"} if experiment == "E0" else {
            name: classification_metrics([labels.get(r.get("scenario_id"), {}).get(name) for r in runs],
                [r.get("predicted_" + name) for r in runs], ["X", "L", "M", "H"], ordinal_labels=["L", "M", "H"])
            for name in ("cr", "ir", "ar")}
        for field in ("llm_priority", "deterministic_priority"):
            if experiment == "E0" or (experiment == "E1" and field == "deterministic_priority"):
                report[field] = {"status": "not_applicable"}
            else:
                report[field] = classification_metrics([labels.get(r.get("scenario_id"), {}).get("priority") for r in runs],
                    [r.get(field) for r in runs], ["P1", "P2", "P3", "P4"], ordinal_labels=["P1", "P2", "P3", "P4"])
        errors = []
        for run in runs:
            label = labels.get(run.get("scenario_id"), {})
            if experiment != "E0" and run.get("computed_environmental_score") is not None and all(label.get(key) in {"X", "L", "M", "H"} for key in ("cr", "ir", "ar")):
                from cvss_engine import build_environmental_vector, calculate_cvss
                requirements = {"confidentiality_requirement": label["cr"], "integrity_requirement": label["ir"], "availability_requirement": label["ar"]}
                expert_vector = build_environmental_vector(run["base_vector"], requirements)
                expert_score = calculate_cvss(expert_vector)["environmental_score"]
                errors.append(abs(run["computed_environmental_score"] - expert_score))
        report["environmental_mae"] = {"n": len(errors), "mae": sum(errors) / len(errors) if errors else None,
                                        "reference": "expert_requirements_same_base_not_true_risk"}
        retrieval = [retrieval_metrics(r.get("retrieved_document_ids", []), labels[r["scenario_id"]]["relevant_document_ids"], r.get("top_k") or 3)
                     for r in runs if experiment == "E3" and r.get("scenario_id") in labels and "relevant_document_ids" in labels[r["scenario_id"]]]
        report["retrieval"] = {"n_labeled": len(retrieval), **{
            key: sum(row[key] for row in retrieval if row[key] is not None) / sum(row[key] is not None for row in retrieval)
            if any(row[key] is not None for row in retrieval) else None
            for key in ("hit_at_1", "hit_at_k", "recall_at_k", "context_precision", "mrr")}}
        rank_groups = defaultdict(list)
        for run in runs:
            label = labels.get(run.get("scenario_id"), {})
            score = run.get("base_score") if experiment == "E0" else run.get("computed_environmental_score")
            if label.get("rank_group") and label.get("ranking_relevance") is not None and score is not None:
                # Keep repeated runs separate to avoid pseudo-replication/tied duplicate scenarios.
                rank_groups[(label["rank_group"], run.get("repeat", 0), run.get("asset_id"))].append((score, label["ranking_relevance"]))
        report["ranking"] = [{"group": group[0], "repeat": group[1], "asset_id": group[2],
                              **ranking_metrics([v[0] for v in rows], [v[1] for v in rows])}
                             for group, rows in rank_groups.items()]
        reports[experiment] = report
    return reports

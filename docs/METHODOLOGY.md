# Experimental methodology

This document specifies a protocol, **not completed experiments**. Read `RESEARCH_SCOPE.md` and `LIMITATIONS.md` before reporting results.

## Source preparation

1. Obtain permission for one internal application and establish the collection boundary with its owner. Enumerate only components actually supported by deployment records, manifests/SBOM or developer evidence.
2. Prepare sanitized asset profiles, component-level software inventory and context documents. Record `source_id`, `source_type`, `source_version`, `collected_at`, verification role/time and `is_synthetic`. Never change the supplied demonstration data to imply real company provenance.
3. Assert `inventory_verified` only when the installed-software inventory is adequate for a negative applicability conclusion. A missing CPE, unsupported version format or incomplete inventory requires conservative Unknown handling.
4. Save NVD API 2.0 snapshots before experiments. Preserve raw configuration arrays and selected metric source/type/version and retrieval time. Snapshot acquisition is distinct from running a cloud-model assessment.
5. Ingest approved context chunks into the separate `organizational_context` collection using stable document IDs and asset filters. Asset identity stays in the inventory collection.

## Independent annotation

Use two or three qualified experts when feasible. Withhold model predictions during independent annotation and adjudication. Follow FIRST CVSS v3.1 Environmental Requirements guidance and the organization's declared operational impact criteria.

Record `(scenario_id, cve_id, asset_id, expert_id, applicability, cr, ir, ar, priority, evidence_ids, rationale, annotated_at)`. Keep annotations outside model input. Report initial inter-rater agreement before consensus; an adjudicated label is not evidence of prior agreement.

Create separate relevance labels identifying authorized document IDs for each query/asset. Group scenarios by asset/application or closely related vulnerabilities before splitting to avoid leakage. Validation selects retrieval thresholds; test data must not tune them or priority weights.

## Ablations

| Experiment | Model input | Appropriate comparison |
|---|---|---|
| E0 | No model: NVD Base CVSS only | Generic technical-severity ranking; no predicted CR/IR/AR or organizational priority |
| E1 | CVE description and the same original Base vector | No-context model baseline |
| E2 | E1 plus sanitized direct asset profile | Organizational-context effect, E1 vs E2 |
| E3 | E2 plus accepted asset-filtered document chunks and source IDs | Retrieval effect, E2 vs E3 |

Hold CVE snapshot, asset profile, model, prompt version, temperature, rubric, embedding, collection contents and evaluation split fixed across comparisons. Log repetitions, failures and abstentions. A missing E3 document is recorded as missing evidence, never silently replaced with an unrelated asset or represented as a successful retrieval.

## Evaluation

- Applicability: per-class confusion matrix, macro-F1, Unknown/coverage rate and individual false-negative analysis.
- Retrieval: Hit@1/Hit@K, Recall@K, MRR and Precision@K; selected-asset rejection/acceptance diagnostics where labeled. Relevance scores are not calibrated correctness probabilities.
- CR/IR/AR: per-metric exact agreement, macro-F1 and weighted kappa when reference and predictions form compatible ordinal labels. Report `X` and missing predictions explicitly.
- Environmental score: MAE against the score calculated from expert metrics on the **identical** Base vector. It is not a true real-world risk score.
- Priority: deterministic-versus-expert and model-versus-expert results separately. Unknown investigation priorities must not masquerade as confirmed remediation rankings. Not Applicable is excluded from remediation P1–P4 classification.
- E0: no CR/IR/AR accuracy or organizational-priority accuracy. Ranking comparisons require several comparable cases; do not turn a single-case score into ranking evidence.
- Reliability: schema/API failures, retrieval misses, abstention, retry counts and run denominators. Do not omit failures from exported runs.

Weighted kappa, ranking NDCG/Spearman, confidence intervals and statistical significance claims require sufficient compatible labels and explicit grouping. Do not infer an improvement from a small example or a unit-test fixture.

## Audit trail and privacy

Persist exact sanitized model input, accepted/rejected evidence and scores, validated model output, NVD provenance, deterministic vectors/scores, rubric result, config, run/scenario IDs and error statuses. Separate raw sensitive source material from sanitized experiment exports. API keys are environment/UI secrets and never log fields. Treat retrieved documents as untrusted data, not instructions.

Cloud runs require explicit authorization. A checkbox records the operator's confirmation; it is not a substitute for written company permission. Use a local model instead if cloud transfer is prohibited. Limit retention/access of history, cached snapshots and experiment logs; the prototype does not provide multiuser access control.

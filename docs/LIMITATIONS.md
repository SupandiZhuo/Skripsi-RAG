# Limitations and outstanding research work

## External blockers

- No authorized internal-company study dataset has been supplied. Existing asset examples remain synthetic.
- No written company permission, verified component inventory, independent expert labels, adjudication or reviewed priority weights have been supplied.
- No validation-selected retrieval threshold or frozen research test-set result exists merely because the software runs.
- E0–E3 execution tooling does not itself demonstrate that organizational context or retrieval improves assessment.

These items require the application owner and experts; software cannot legitimately fabricate them.

## Technical scope

- CVSS v3.1 only. The model interprets CR/IR/AR but is not a CVSS calculator; numerical scores are deterministic and not calibrated real-world risk.
- CPE applicability uses conservative three-valued logic and a bounded version-comparison policy, not a universal vendor-version oracle. Unsupported syntax, incomplete inventory and ambiguous matching remain Unknown. A negative conclusion depends on verified inventory completeness.
- Retrieval relevance depends on collection contents, embedding geometry and threshold. A score is not a probability. Asset-filtered retrieval can miss useful documents, and prompt/schema validation cannot prove the truth of a model explanation.
- Retrieved documents may contain sensitive facts or prompt injection. Input minimization and instruction separation reduce risk but do not prove de-identification or eliminate adversarial behavior. Review source text before cloud ingestion/use.
- Priority weights are researcher-defined, versioned decision rules requiring expert review. Business impact factors may be correlated. Sensitivity analysis should perturb weights on validation data and report ranking stability before freezing the rubric.
- NVD availability, source updates, rate limits and cloud-model access affect live execution. Frozen snapshots isolate public-data drift; deterministic temperature does not guarantee identical cloud responses.
- JSON history and Chroma editing are single-user prototype persistence. There is no production authentication, multiuser transaction model, access policy or automated data-retention service.
- Third-party model IDs and dependency availability must be verified in the actual deployment. Never describe an unavailable cloud model as tested based on mocked calls.

## Before reporting thesis results

Supply the missing authorized data and labels, freeze experimental settings, run E0–E3 on the held-out split, inspect error cases and report failures/Unknown coverage. Record the actual environment and a resolved dependency lock for that deployment; do not invent version pins from minimum requirements. Perform a permitted live end-to-end model assessment separately from offline/mocked tests. Report all unexecuted validation honestly.

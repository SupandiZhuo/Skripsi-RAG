# Implementation and final validation status

## Delivered code

The code-side implementation of `RENCANA_PERBAIKAN_SKRIPSI_RAG.md` is complete for the bounded research prototype. The Streamlit UI follows `Design.png`: three-page sidebar, initial/results analysis states, priority-first cards, separate Base/Environmental technical severity, applicability, CIA requirements, summary/evidence expanders, searchable/editable inventory and history.

No backups, baseline archive, branch, commit or push were created. Test execution began only after the initial implementation and integration work were complete. Bugs discovered during final validation were corrected and retested.

## Audit coverage

| Audit IDs | Implementation / remaining boundary |
|---|---|
| C01–C05 | Native NVD arrays, configuration operators/negation, vulnerable-versus-supporting conditions, strict escaped CPE parsing, conservative supported version comparison and explicit inventory verification. Unsupported/ambiguous cases remain Unknown; not a universal CPE/version oracle. |
| C06–C08 | Active 100-point rubric; Not Applicable has no remediation priority; Unknown has a separate investigation priority. Valid zero CVSS is preserved. |
| C09–C11 | Separate asset-filtered document retrieval, accepted/rejected evidence and one named scoring method; evidence reaches sanitized model input and logs. Threshold remains heuristic until independently labeled validation. |
| C12–C13 | **External blocker:** authorized company dataset and independent expert/retrieval labels have not been supplied. No company records or expert answers were invented. |
| C14–C15 | Strict output schema, independent CR/IR/AR reasons and input-checked citations; raw-label/baseline exclusion and untrusted-data prompt separation. These controls do not prove substantive correctness. |
| C16–C18 | E0–E3 runner and evaluation functions record deterministic/model outputs, failures, missing predictions and reference comparisons. **Not performed:** expert validation, sensitivity study, substantive ablation results. |
| C19 | One documented active rubric, versioned `organizational-priority-v2.0`; legacy module clearly separated. |
| C20 | Asset edits regenerate document content and embeddings through the configured embedding function; attributed context ingestion updates chunks and verifies writes. |
| C21–C25 | Explicit NVD CVSS v3.1 source selection, strict Base-vector validation, no silent v3.0 fallback, raw snapshots/cache/retries/pacing, manual override reason and provenance. |
| C26–C27 | Resolved dependency lock, frozen experiment config, source/runtime provenance, JSONL/CSV results and enriched history with exact sanitized input/output. A recorded seed is not a cloud determinism guarantee. |
| C28 | Offline unit/integration/AppTest suite and actual browser visual checks completed. **Not performed:** authorized live cloud-model end-to-end assessment. |
| C29 | Input allowlists, source/citation checks, untrusted-document separation and explicit cloud consent. **External blocker:** written company authorization/source review. No claim of complete secret redaction from arbitrary prose. |
| C30 | Explicit single-user prototype boundary; no production transaction/authentication claim. |
| C31 | Graph points inactive, including when compatibility arguments are supplied. |
| C32–C35 | CVSS v4, EPSS/KEV/SSVC, fine-tuning and enterprise deployment remain explicitly out of the core scope. |

## Final execution evidence

Runtime: `/usr/bin/python3`, Python 3.13.12. Relevant observed packages: Streamlit 1.66.1, pandas 3.1.0, cvss 3.6, Pydantic 2.14.1, pytest 9.0.3. `requirements.lock` includes the resolved application/test dependency set.

- `/usr/bin/python3 -m pytest -q --tb=short`: **327 passed, 11 subtests passed**, 13.68 seconds; no warnings in that final run.
- Final integration review also confirmed that CSV ingestion preserves explicit native-boolean and canonical-string `inventory_verified` values in stored metadata and embedded text; regression coverage uses isolated collections.
- `git diff --check`: passed.
- Both evaluation CLI help entrypoints executed successfully.
- Actual E0 CLI replay against five attributed NVD snapshots produced five logged records: four completed Base-score assessments and one explicitly recorded unsupported-vector failure (CVE-1999-0095 has no supported v3.1 baseline). No model call was made. JSONL, CSV, manifest and summary artifacts were read back.
- Streamlit server health endpoint returned `ok`.
- Temporary browser-test servers were stopped after verification; ports 8501 and 8502 were confirmed closed. Deprecated width arguments and unused static serving were removed after inspecting server logs.
- Browser inspection covered real inventory, initial analysis and empty history states, plus a clearly labeled **offline UI test fixture** for result cards. The fixture is not a live company/model assessment.
- Browser findings corrected: sidebar Settings placement, navigation icons/wrapping, full-width inventory search, table status colors, aligned result cards, severity badges and New Analysis placement.
- Mobile viewport check: viewport width 390 and document width 390; no horizontal overflow or browser exception in the inspected history state.

The first final suite run exposed legacy tests importing the Streamlit entrypoint at collection time and accessing the project Chroma directory. Tests were changed to import UI-independent modules and use isolated temporary collections. The final suite no longer depends on project Chroma contents or real model downloads.

Temporary smoke artifacts (not research results):

- `/root/.hermes/cache/scratch/skripsi-e0-final-smoke/`
- `/root/.hermes/cache/scratch/skripsi-ui-check/`

## What is not complete

The **thesis research itself is not validated** by this implementation. An authorized study case, verified operational inventory, independent expert labels, reviewed rubric, validation-selected threshold, held-out E1/E2/E3 comparisons and permitted live-model smoke test remain outstanding. See `RESEARCH_SCOPE.md`, `METHODOLOGY.md`, `DATA_COLLECTION.md` and `LIMITATIONS.md`.

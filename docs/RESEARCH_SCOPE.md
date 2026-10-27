# Research scope and evidence boundaries

## Objective

Evaluate whether organizational evidence helps a model assign CVSS v3.1 Environmental Requirements (CR/IR/AR) and whether retrieving documents adds value beyond directly supplied asset metadata. The system is a **single-user research prototype**, not a production vulnerability scanner or a validated risk oracle.

- **RQ1:** Does direct organizational context improve agreement with independent expert CR/IR/AR labels compared with no organizational context (E1 versus E2)?
- **RQ2:** Does asset-filtered document retrieval improve agreement beyond the same direct asset profile (E2 versus E3)?
- **RQ3:** How do deterministic and model-suggested priorities agree with independent organizational expert priorities?

## Fixed boundaries

- NVD is the public vulnerability source. The core experiment accepts **CVSS v3.1 Base vectors only**; CVSS v3.0/v4.0 are not silently substituted.
- Asset identity comes from exact stable-ID lookup. Similarity does not establish identity or software applicability.
- Retrieval operates on separately ingested context documents filtered to the selected asset, never unrelated asset records.
- The model interprets evidence and emits schema-validated CR/IR/AR and a priority suggestion. It never computes numerical CVSS scores or changes Base metrics.
- Deterministic code constructs the Environmental vector; the `cvss` library calculates it. Technical Severity and Organizational Priority remain separate.
- Only confirmed applicability can receive a remediation priority. Unknown applicability has a separate investigation priority; Not Applicable has no remediation priority.
- Graph dependencies, KEV/EPSS, fine-tuning, CVSS v4, multiuser authentication and production deployment are outside the core implementation.

## Data that must be supplied, not invented

The existing seven-asset CSV is synthetic demonstration data, not company evidence. Select one authorized internal application as the study case before collecting its assets, installed software, architecture, data-flow, operational requirements and controls. Keep development, staging and production separate.

Do not create purported company records, expert labels, independent review, approvals or evaluation results from assumptions. A source field with `unknown` is preferable to an invented value. Empty or missing research inputs block a substantive research claim, not merely the UI.

Required external inputs:

1. Written authorization for data collection and, separately, cloud-model processing.
2. Sanitized asset profiles and component-level verified inventory with provenance and completeness assertions.
3. Authorized context documents with stable document/source IDs and explicit synthetic flags.
4. Independent expert labels and retrieval relevance labels; adjudication records when applicable.
5. Frozen validation/test splits and expert-reviewed priority weights.

## No-leakage policy

Expert CR/IR/AR, expert priorities, adjudication outcomes and relevance labels are evaluation targets, not model inputs. Default model context excludes declared CIA metric baselines as well. A future policy-constrained experiment must be separately named and must not be reported as independent prediction.

## Completion boundaries

Code implementation, passing mocked tests, a live NVD request, and expert validation are different achievements. Report them separately. No RAG accuracy or thesis-validity claim is justified until the authorized dataset, independent labels and frozen E1/E2/E3 comparisons exist.

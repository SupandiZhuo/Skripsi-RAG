# AGENTS.md

## Project

S1 thesis prototype: Context-Aware Vulnerability Assessment using NVD + RAG.

Main pipeline:

CVE ID
→ NVD API
→ CVE description + CVSS vector
→ retrieve organizational asset context from ChromaDB
→ model contextual assessment
→ deterministic CVSS calculation
→ Technical Severity + Organizational Priority + explanation

The application currently uses Streamlit, ChromaDB, HuggingFace embeddings, LangChain, and the Python `cvss` library.

## Core Research Requirement

The system must demonstrate:

1. NVD as the vulnerability/technical data source.
2. RAG as the organizational asset-context source.
3. A model receiving vulnerability information + retrieved context.
4. Technical Severity and Organizational Priority as separate outputs.
5. Deterministic CVSS calculation where applicable.
6. Evidence showing what RAG context influenced the result.

Do not blur Technical Severity and Organizational Priority.

## Coding Rules

- Inspect existing code before modifying it.
- Search for relevant symbols/files before reading whole files.
- Read only files needed for the current task.
- Do not scan or summarize the entire repository unless required.
- Prefer minimal modifications over rewrites.
- Reuse existing architecture and dependencies when reasonable.
- Do not create unnecessary abstractions, files, classes, or dependencies.
- Preserve working behavior unless the task requires changing it.
- Avoid unrelated refactoring.
- Do not modify generated files, model files, datasets, ChromaDB data, or secrets unless explicitly required.
- Never hardcode API keys.

## Token Efficiency

- Keep responses concise.
- Do not repeat the task or previously established context.
- Do not paste entire files after editing.
- Report only changed files and important changes.
- Keep command output minimal.
- Use targeted searches instead of broad repository reads.
- Do not repeatedly read unchanged files.
- Do not explain obvious code.
- When context becomes large, retain a short working summary instead of old details.

## RAG Requirements

Asset context should use structured fields where available, such as:

- asset_id
- asset_name
- environment
- internet_exposed
- business_criticality
- data_sensitivity
- network_zone

Retrieval must:

- return multiple candidates when appropriate;
- expose retrieval score/evidence;
- handle missing or low-confidence results;
- never silently treat an unrelated result as valid context.

Do not assume semantic similarity proves asset identity.

## CVSS Requirements

- NVD CVSS is the technical baseline.
- Do not let the LLM invent numerical CVSS scores.
- Numerical CVSS calculation must use deterministic code/library.
- Validate model-generated CVSS metrics/vector before calculation.
- Do not change CVSS metrics without defensible contextual evidence.
- Do not assume an internal/restricted network automatically means MAV:L.

## Model Output

Prefer structured/schema-validated output.

Expected conceptual output:

- modified_vector / contextual metrics
- justification
- organizational_priority (P1-P4)
- remediation
- evidence/retrieved context

Do not depend on regex cleanup as the primary output-validation mechanism.

## Priority

Organizational Priority (P1-P4) is distinct from CVSS severity.

Priority should use explicitly defined organizational rules/context and must be explainable.

## Evaluation

Design changes so results can later be evaluated programmatically.

Important experiment data should be loggable/exportable:

- CVE
- NVD description/vector/score
- retrieved asset context
- retrieval score
- model input/output
- final CVSS
- technical severity
- organizational priority

Do not couple evaluation logic tightly to Streamlit UI.

## Workflow

For each task:

1. Locate relevant code.
2. Understand the smallest affected area.
3. Implement the smallest correct change.
4. Run targeted validation/tests.
5. Fix failures caused by the change.
6. Stop.

Do not continue improving unrelated areas.

## Completion Response

Return only:

- changed files
- short summary of changes
- tests/checks performed
- unresolved issue, if any

Do not produce a long tutorial unless requested.
# Asset Intelligence & Context-Aware Vulnerability Assessment (NVD + RAG)

S1 thesis prototype demonstrating a **Context-Aware Vulnerability Assessment**
pipeline that combines NVD threat intelligence with organizational asset
context retrieved via RAG, interpreted by an LLM, and scored deterministically
with CVSS v3.1. The application presents a searchable asset inventory and a
focused vulnerability-analysis workflow matching `Design.png`. This is a
single-user research prototype; the supplied asset examples are synthetic.
See `docs/RESEARCH_SCOPE.md`, `docs/METHODOLOGY.md` and `docs/LIMITATIONS.md`.

## Pipeline

```
CVE ID
  -> NVD API (CVE description, original CVSS v3.1 vector, base score)
  -> Exact asset-ID lookup + asset-filtered document retrieval (ChromaDB)
  -> Cloud LLM (interprets NVD info + RAG context)
       -> confidentiality_requirement (CR): X | L | M | H
       -> integrity_requirement (IR):        X | L | M | H
       -> availability_requirement (AR):     X | L | M | H
       -> organizational_priority (suggestion): P1 | P2 | P3 | P4
       -> justification, remediation
  -> Deterministic code builds the Environmental CVSS vector
       (Base metrics AV/AC/PR/UI/S/C/I/A preserved as-is from NVD; MAV left
       undefined/X - no MAV inference in this prototype)
  -> `cvss` Python library calculates the numerical Environmental score
   -> Deterministic rubric computes Organizational Priority (P1-P4) from
      contextual CVSS score, Asset Schema v2 evidence, CPE applicability,
      without graph dependency points (organizational_priority.py)
   -> UI: Vulnerability Analysis, Asset Inventory, Analysis History
```

**Responsibility boundaries (must not blur):**

| Component | Produces |
|---|---|
| NVD API | CVE description, Base CVSS v3.1 vector, base score |
| RAG (ChromaDB) | Asset-filtered context documents, source IDs, relevance scores and accepted/rejected evidence |
| Cloud LLM (prototype model) | CR / IR / AR / organizational_priority **suggestion** / justification / remediation only |
| Deterministic code (`cvss_engine.py`) | Strict Base v3.1 validation and Environmental vector construction |
| `cvss` library | The only component that computes numerical CVSS scores |
| `organizational_priority.py` | Confirmed remediation P1–P4; separate investigation priority for Unknown; no priority for Not Applicable; explicit missing-evidence state |
| `priority_rubric.py` | Legacy deterministic priority rubric (kept for reference/tests) |

The LLM never generates a CVSS vector string, a numerical score, or modifies
Base metrics — its output is schema-validated (Pydantic) before use.

## Project layout

- `app.py` — Streamlit UI with initial/result analysis states, inventory editing and history.
- `nvd_client.py`, `cvss_engine.py`, `pipeline.py` — UI-independent NVD, deterministic scoring and E0–E3 assessment functions.
- `evaluation/` — Batch experiment runner, frozen run settings and evaluation metrics; no fabricated expert labels/results.
- `llm_assessment.py` — `ContextualAssessment` schema, prompt template, and `analyze_with_llm()` (currently backed by Gemini via `_build_llm()`, isolated for future model swap).
- `rag_retrieval.py` — Exact inventory lookup, asset-filtered document retrieval and traceable relevance gating. Scores are not calibrated probabilities.
- `ingest_assets.py` — Ingestion scripts: `python ingest_assets.py` embeds v1 `asset_data.csv` into `corporate_assets`; `python ingest_assets.py --v2` embeds `data/asset_data_v2.csv` into `corporate_assets_v2` (Schema v2, deterministic IDs, idempotent upsert).
- `asset_data.csv` / `data/asset_data_v2.csv` — Sample organizational asset inventory; v2 adds environment, internet_exposed, software/CPE, CVSS CR/IR/AR baseline, and provenance fields (see `docs/ASSET_SCHEMA_V2.md`).
- `asset_schema_v2.py` — Stdlib Schema v2 validator used before every v2 ingestion.
- `priority_rubric.py` — Legacy baseline only, not the active application rubric.
- `organizational_priority.py` — Active versioned rubric; see `docs/PRIORITY_RUBRIC.md`. Weights still require independent expert review.
- `asset_inventory.py` — Software inventory serialization, CPE validation, and CPE collection for applicability matching (Phase B1).
- `assessment_history.py` — Minimal JSON persistence for completed vulnerability assessments.
- `cve_applicability.py` — CPE-based CVE applicability matching against NVD CPE configurations.
- `data/assessment_history.json` — Saved assessment results (created on first save).
- `.streamlit/config.toml` — Project theme and server settings.
- `tests/` — All test modules (run with `python -m pytest tests -q` from the project root):
  - `tests/conftest.py` — Puts the project root on `sys.path` so tests import root modules.
  - `tests/test_contextual_assessment.py` — Targeted checks for schema validation, Base metric preservation, MAV absence, and deterministic vector/score construction (LLM calls mocked, no live API usage).
  - `tests/test_priority_rubric.py` — Targeted checks for the priority rubric.
  - `tests/test_app_ui.py` — Headless Streamlit `AppTest` checks: navigation, asset selection/editing, result persistence and a mocked analysis.
  - `tests/test_asset_map_ui.py` — Regression guard that the retired graph UI (Cytoscape / `asset_graph`) is never loaded by the active app.
  - `tests/test_assessment_history.py` — Targeted checks for assessment history save/load/delete.
  - `tests/test_asset_inventory.py` — Tests for asset metadata updates, software inventory persistence, CPE validation, and legacy-collection preservation.

## Setup

```powershell
pip install -r requirements.lock
python ingest_assets.py --v2   # one-time: embeds asset_data_v2.csv into ./chroma_db (corporate_assets_v2)
streamlit run app.py
```

A Google Gemini API key and explicit authorization to send sanitized context
to the cloud model are required for live LLM assessments. Configure the model
ID in Settings for an available model in your account. E0 batch assessments do
not call the model. Do not ingest private material without written permission.

`requirements.lock` resolves application and test dependencies for Python 3.13;
`requirements-dev.txt` contains the development input requirements. Actual
test environment versions are recorded separately from a resolver-produced lock.

## Run tests

```powershell
python -m pytest -q
```

## Research readiness

The implementation separates technical severity, remediation priority,
investigation priority and applicability, and supports E0–E3 ablations outside
the UI. That infrastructure does **not** establish scientific accuracy.

Before claiming improvement, supply authorized company evidence, independent
expert CR/IR/AR and priority labels, document relevance labels and grouped
validation/test splits. Freeze the model, snapshots, collection, threshold and
rubric before final evaluation. Default retrieval thresholds and priority
weights remain research choices requiring validation. Fine-tuning, full graph
dependencies, CVSS v4 and production multiuser deployment are future work.

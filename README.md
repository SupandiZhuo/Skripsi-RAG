# Asset Intelligence & Context-Aware Vulnerability Assessment (NVD + RAG)

S1 thesis prototype demonstrating a **Context-Aware Vulnerability Assessment**
pipeline that combines NVD threat intelligence with organizational asset
context retrieved via RAG, interpreted by an LLM, and scored deterministically
with CVSS v3.1. The application presents a searchable asset inventory and a
focused vulnerability-analysis workflow; graph modules and persisted graph
data are retained but are not loaded by the active UI.

## Pipeline

```
CVE ID
  -> NVD API (CVE description, original CVSS v3.1 vector, base score)
  -> RAG retrieval (ChromaDB: organizational asset context + confidence score)
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
      and optional verified dependencies (organizational_priority.py)
   -> Dashboard UI: Dashboard, Asset Inventory, Vulnerability Analysis,
      Analysis History
```

**Responsibility boundaries (must not blur):**

| Component | Produces |
|---|---|
| NVD API | CVE description, Base CVSS v3.1 vector, base score |
| RAG (ChromaDB) | Retrieved asset metadata + retrieval confidence score |
| Cloud LLM (prototype model) | CR / IR / AR / organizational_priority **suggestion** / justification / remediation only |
| Deterministic code (`app.py`) | Environmental vector construction (CR/IR/AR appended to untouched Base vector) |
| `cvss` library | The only component that computes numerical CVSS scores |
| `organizational_priority.py` | Deterministic Organizational Priority (P1-P4) from contextual CVSS + verified asset evidence + CPE applicability + optional verified dependencies; documented rubric, explicit incomplete state for missing evidence |
| `priority_rubric.py` | Legacy deterministic priority rubric (kept for reference/tests) |

The LLM never generates a CVSS vector string, a numerical score, or modifies
Base metrics — its output is schema-validated (Pydantic) before use.

## Project layout

- `app.py` — Streamlit UI and pipeline orchestration; `build_environmental_vector()` (deterministic CR/IR/AR vector construction).
- `llm_assessment.py` — `ContextualAssessment` schema, prompt template, and `analyze_with_llm()` (currently backed by Gemini via `_build_llm()`, isolated for future model swap).
- `rag_retrieval.py` — ChromaDB retrieval with confidence-score gating (`retrieve_asset_context`, `filtered_metadata`).
- `ingest_assets.py` — Ingestion scripts: `python ingest_assets.py` embeds v1 `asset_data.csv` into `corporate_assets`; `python ingest_assets.py --v2` embeds `data/asset_data_v2.csv` into `corporate_assets_v2` (Schema v2, deterministic IDs, idempotent upsert).
- `asset_data.csv` / `data/asset_data_v2.csv` — Sample organizational asset inventory; v2 adds environment, internet_exposed, software/CPE, CVSS CR/IR/AR baseline, and provenance fields (see `docs/ASSET_SCHEMA_V2.md`).
- `asset_schema_v2.py` — Stdlib Schema v2 validator used before every v2 ingestion.
- `priority_rubric.py` — Legacy deterministic Organizational Priority rubric (P1-P4, documented weights/thresholds, explicit incomplete state; see `docs/PRIORITY_RUBRIC.md`).
- `organizational_priority.py` — Phase C deterministic Organizational Priority rubric (P1-P4) using contextual CVSS, asset evidence, CPE applicability, and optional verified dependencies; frozen dataclass results with contributions/missing/unknown evidence.
- `asset_inventory.py` — Software inventory serialization, CPE validation, and CPE collection for applicability matching (Phase B1).
- `assessment_history.py` — Minimal JSON persistence for completed vulnerability assessments.
- `cve_applicability.py` — CPE-based CVE applicability matching against NVD CPE configurations.
- `data/assessment_history.json` — Saved assessment results (created on first save).
- `.streamlit/config.toml` — Project theme and server settings.
- `tests/` — All test modules (run with `python -m pytest tests -q` from the project root):
  - `tests/conftest.py` — Puts the project root on `sys.path` so tests import root modules.
  - `tests/test_contextual_assessment.py` — Targeted checks for schema validation, Base metric preservation, MAV absence, and deterministic vector/score construction (LLM calls mocked, no live API usage).
  - `tests/test_priority_rubric.py` — Targeted checks for the priority rubric.
  - `tests/test_app_ui.py` — Headless Streamlit `AppTest` checks: four-page navigation, asset selection/editing, analysis preselection, and a mocked end-to-end analysis run.
  - `tests/test_asset_map_ui.py` — Regression guard that the retired graph UI (Cytoscape / `asset_graph`) is never loaded by the active app.
  - `tests/test_assessment_history.py` — Targeted checks for assessment history save/load/delete.
  - `tests/test_asset_inventory.py` — Tests for asset metadata updates, software inventory persistence, CPE validation, and legacy-collection preservation.

## Setup

```powershell
pip install -r requirements.txt
python ingest_assets.py --v2   # one-time: embeds asset_data_v2.csv into ./chroma_db (corporate_assets_v2)
streamlit run app.py
```

A Google Gemini API key (entered in the sidebar) is required to run a live
assessment.

## Run tests

```powershell
python -m pytest -q
```

## Current status

Implemented:
- NVD fetch, RAG retrieval with confidence threshold gating.
- Asset Schema v2 ingestion (`corporate_assets_v2`) and pipeline switch: retrieval context now exposes environment, internet_exposed, software/CPE, data_classification, declared CR/IR/AR baseline, and provenance to the LLM.
- Cloud LLM (Gemini) contextual assessment restricted to CR/IR/AR + organizational_priority, schema-validated.
- Deterministic Environmental vector construction and CVSS v3.1 scoring via the `cvss` library.
- Deterministic Organizational Priority rubric (Phase C): documented weights/thresholds over contextual CVSS + verified Asset Schema v2 evidence + CPE applicability + verified business-graph dependencies; explicit incomplete state for missing evidence and Not Applicable capping at P4; the LLM priority remains visible as a model suggestion.
- Focused prototype navigation: four pages — Dashboard, Asset Inventory, Vulnerability Analysis, Analysis History.
- Asset Inventory provides search/filter, a software/CPE summary table, stable-ID selection, full metadata inspection, metadata/software editing, and persistent ChromaDB updates.
- Vulnerability Analysis selects ChromaDB metadata by stable asset ID, fetches NVD data at run time, and presents semantic candidates as evidence without substituting a different asset.
- Graph modules, Cytoscape assets, and persisted graph data were removed; the active workflow uses no graph UI, and a regression guard (`tests/test_asset_map_ui.py`) keeps the app graph-free.
- Editable Asset Inventory (Phase B1): view/search/filter assets, inspect full metadata, edit schema-v2 fields, manage installed software records with valid CPE 2.3 identifiers, and save changes persistently to `corporate_assets_v2` (no CSV re-ingestion required).
- Multi-CPE software inventory stored in ChromaDB metadata (`software_inventory_json`) and consumed by the CPE applicability evaluator.
- Minimal assessment history persistence in `data/assessment_history.json`.
- Hardened CPE-based CVE applicability matching (Phase A.1): evaluates NVD CPE configurations with three-valued logic (True / False / Unknown), preserves AND/OR/negate structure, supports inclusive/exclusive version-range qualifiers, handles multiple installed CPEs, and reports **Confirmed / Not Applicable / Unknown** with traceable evidence (configuration path, criteria, range conditions) — independent of RAG similarity.
- Streamlit UI showing Base vs. Context-Aware score/severity, Organizational Priority (+ rubric explanation/contributions/missing/unknown evidence), CR/IR/AR, CPE applicability, justification, and RAG evidence.

## Next steps

1. **Fine-tuned model integration** — Replace/augment the cloud LLM (`_build_llm` in `llm_assessment.py`) with the separately trained Llama/Qwen models once ready, without touching RAG retrieval or CVSS calculation.
2. **Evaluation pipeline** — Build a script (decoupled from Streamlit, reusing `rag_retrieval.py` / `llm_assessment.py` / `organizational_priority.py` / `cve_applicability.py` / `app.build_environmental_vector`) to batch-run CVEs and log: CVE, NVD description/vector/score, retrieved asset context + retrieval score, model input/output, final CVSS vector/score, deterministic priority vs. LLM priority, CPE applicability — for programmatic comparison between the cloud LLM, the rubric, and future fine-tuned models.
3. **Asset dataset expansion** — `asset_data_v2.csv` currently has 7 synthetic sample assets; expand for more meaningful retrieval/evaluation coverage once the evaluation pipeline exists.
4. **Dependency pinning** — `pytest` was used ad hoc but is not yet in `requirements.txt`; add it as a dev dependency (or a separate `requirements-dev.txt`) if testing becomes part of the regular workflow.

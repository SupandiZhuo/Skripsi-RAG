# Context-Aware Vulnerability Assessment (NVD + RAG)

S1 thesis prototype demonstrating a **Context-Aware Vulnerability Assessment**
pipeline that combines NVD threat intelligence with organizational asset
context retrieved via RAG, interpreted by an LLM, and scored deterministically
with CVSS v3.1.

## Pipeline

```
CVE ID
  -> NVD API (CVE description, original CVSS v3.1 vector, base score)
  -> RAG retrieval (ChromaDB: organizational asset context + confidence score)
  -> Cloud LLM (interprets NVD info + RAG context)
       -> confidentiality_requirement (CR): X | L | M | H
       -> integrity_requirement (IR):        X | L | M | H
       -> availability_requirement (AR):     X | L | M | H
       -> organizational_priority:           P1 | P2 | P3 | P4
       -> justification, remediation
  -> Deterministic code builds the Environmental CVSS vector
       (Base metrics AV/AC/PR/UI/S/C/I/A preserved as-is from NVD; MAV left
       undefined/X - no MAV inference in this prototype)
  -> `cvss` Python library calculates the numerical Environmental score
  -> UI: NVD Base Score/Severity, Context-Aware Score/Severity,
     Organizational Priority, CR/IR/AR, justification, RAG evidence
```

**Responsibility boundaries (must not blur):**

| Component | Produces |
|---|---|
| NVD API | CVE description, Base CVSS v3.1 vector, base score |
| RAG (ChromaDB) | Retrieved asset metadata + retrieval confidence score |
| Cloud LLM (prototype model) | CR / IR / AR / organizational_priority / justification / remediation only |
| Deterministic code (`app.py`) | Environmental vector construction (CR/IR/AR appended to untouched Base vector) |
| `cvss` library | The only component that computes numerical CVSS scores |

The LLM never generates a CVSS vector string, a numerical score, or modifies
Base metrics — its output is schema-validated (Pydantic) before use.

## Project layout

- `app.py` — Streamlit UI and pipeline orchestration; `build_environmental_vector()` (deterministic CR/IR/AR vector construction).
- `llm_assessment.py` — `ContextualAssessment` schema, prompt template, and `analyze_with_llm()` (currently backed by Gemini via `_build_llm()`, isolated for future model swap).
- `rag_retrieval.py` — ChromaDB retrieval with confidence-score gating (`retrieve_asset_context`, `filtered_metadata`).
- `ingest_assets.py` — One-time script to embed `asset_data.csv` into ChromaDB (`./chroma_db`).
- `asset_data.csv` — Sample organizational asset inventory (asset_id, network_zone, business_criticality, data_sensitivity, etc.).
- `test_contextual_assessment.py` — Targeted checks for schema validation, Base metric preservation, MAV absence, and deterministic vector/score construction (LLM calls mocked, no live API usage).

## Setup

```powershell
pip install -r requirements.txt
python ingest_assets.py       # one-time: embeds asset_data.csv into ./chroma_db
streamlit run app.py
```

A Google Gemini API key (entered in the sidebar) is required to run a live
assessment.

## Run tests

```powershell
python -m pytest test_contextual_assessment.py -q
```

## Current status

Implemented:
- NVD fetch, RAG retrieval with confidence threshold gating.
- Cloud LLM (Gemini) contextual assessment restricted to CR/IR/AR + organizational_priority, schema-validated.
- Deterministic Environmental vector construction and CVSS v3.1 scoring via the `cvss` library.
- Streamlit UI showing Base vs. Context-Aware score/severity, Organizational Priority, CR/IR/AR, justification, and RAG evidence.

## Next steps

1. **Organizational Priority rules** — `organizational_priority` (P1-P4) is currently entirely LLM judgment with no deterministic tie-breaking or documented rubric. Define explicit, explainable organizational-priority rules/thresholds (e.g. combining Environmental CVSS score with business_criticality) so results are reproducible and defensible for the thesis.
2. **Fine-tuned model integration** — Replace/augment the cloud LLM (`_build_llm` in `llm_assessment.py`) with the separately trained Llama/Qwen models once ready, without touching RAG retrieval or CVSS calculation.
3. **Evaluation pipeline** — Build a script (decoupled from Streamlit, reusing `rag_retrieval.py` / `llm_assessment.py` / `app.build_environmental_vector`) to batch-run CVEs and log: CVE, NVD description/vector/score, retrieved asset context + retrieval score, model input/output, final CVSS vector/score, and organizational priority — for programmatic comparison between the cloud LLM and future fine-tuned models.
4. **`internet_exposed` field** — Declared in `rag_retrieval.CONTEXT_METADATA_FIELDS` but not present in `asset_data.csv`/`ingest_assets.py`. Decide whether to populate it as supporting evidence for the LLM's CR/IR/AR reasoning (it is not used for any deterministic MAV rule, per the current no-MAV-inference design).
5. **Asset dataset expansion** — `asset_data.csv` currently has 7 sample assets; expand for more meaningful retrieval/evaluation coverage once the evaluation pipeline exists.
6. **Dependency pinning** — `pytest` was used ad hoc for `test_contextual_assessment.py` but is not yet in `requirements.txt`; add it as a dev dependency (or a separate `requirements-dev.txt`) if testing becomes part of the regular workflow.

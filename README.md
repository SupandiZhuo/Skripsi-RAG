# Context-Aware Vulnerability Assessment — NVD + RAG

S1 thesis prototype for contextual triage of **existing Acunetix Vulnerabilities
CSV reports**. The application combines NVD technical data with organizational
asset evidence from ChromaDB, uses Gemini to interpret CR/IR/AR, and calculates
numerical CVSS scores with deterministic code.

This is a single-user research prototype. The public asset examples are
synthetic; private company records and live test results are not included in Git.

## Assessment workflow

```text
Acunetix Vulnerabilities CSV
  → validate findings and confirm target-to-asset mapping
  → extract explicit CVE identifiers and retrieve NVD CVSS v3.1 data
  → check CPE applicability against the installed-software inventory
  → retrieve attributed organizational documents for the selected asset
  → schema-validated CR / IR / AR, reasons, citations and remediation
  → deterministic Environmental CVSS and organizational priority rubric
  → review, export and assessment history
```

**Technical Severity and Organizational Priority are separate outputs.** NVD
provides the technical baseline. The model does not invent numerical CVSS scores
or change Base metrics. Unknown applicability produces investigation priority
and a provisional contextual score; it does not authorize remediation priority.
CVE-less findings, unsupported vectors, missing evidence and failures remain
visible. The application does not initiate scans or call the scanner API.

## Quick start

Runtime: **Python 3.12.14**, with pinned dependencies in `uv.lock` and CPU PyTorch.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the repository root on Windows:

```powershell
uvx --from uv==0.13.8 uv sync --locked
.\.venv\Scripts\python.exe scripts/ingest_assets.py --v2
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Skip ingestion when your database is already initialized. The `--v2` command
loads only the public synthetic inventory; document-RAG assessment additionally
requires authorized organizational context. For the local Orion package, custom
sources, Docker, Linux commands and runtime configuration, see [Setup](docs/SETUP.md).
Enter the Gemini API key in UI Settings and provide cloud consent before a live
assessment. Open [http://localhost:8501](http://localhost:8501).

With Docker Desktop/Engine and Compose, an inventory-only demo is:

```powershell
docker compose run --rm --build app python scripts/ingest_assets.py --v2
docker compose up -d app
```

The private-package initialization command is documented in [Setup](docs/SETUP.md).
Docker uses persistent volumes for ChromaDB, history, NVD snapshots and model
files. It does not copy private inputs or existing native state into the image.

## Repository layout

```text
app.py                    Streamlit entry point
src/skripsi_rag/          Application UI and assessment/asset/RAG/VA modules
scripts/                  Command-line entry points
evaluation/               UI-independent research runners and metrics
tests/                    Offline regression tests and public-fixture references
data/examples/            Public synthetic asset inventories
data/private/             Local authorized inputs (ignored by Git)
docs/                     Setup, usage, methodology and research boundaries
requirements/             Generated runtime/dev dependency exports
Dockerfile, compose.yaml  Container configuration
pyproject.toml, uv.lock   Python package configuration and dependency lock
```

The `skripsi_rag` package contains `ui.py`, `pipeline.py`, `nvd_client.py`,
`llm_assessment.py`, `rag_retrieval.py`, `cvss_engine.py`, the active
`organizational_priority.py` rubric, asset/schema/ingestion helpers, VA import and
batch helpers, and history persistence. `legacy_priority_rubric.py` contains the
older policy used for baseline comparisons.

## Documentation

- [Setup and Docker](docs/SETUP.md)
- [VA CSV usage and evidence boundaries](docs/USAGE.md)
- [Development, dependency updates and GitHub preparation](docs/DEVELOPMENT.md)
- [Data layout and private inputs](data/README.md)
- [Research scope](docs/RESEARCH_SCOPE.md), [methodology](docs/METHODOLOGY.md),
  [limitations](docs/LIMITATIONS.md)
- [Asset Schema v2](docs/ASSET_SCHEMA_V2.md) and [priority rubric](docs/PRIORITY_RUBRIC.md)

## Tests and research status

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Four existing tests require archived NVD fixtures that are not currently bundled;
see [Development](docs/DEVELOPMENT.md). Mocked tests and live engineering checks
are not expert validation. The E0–E3 experiment infrastructure still requires
independent expert labels, relevance annotations, frozen splits and reviewed
priority weights before making scientific improvement claims.

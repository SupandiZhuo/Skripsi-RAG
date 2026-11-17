# Setup and runtime

Run commands from the repository root. Python is fixed at **3.12.14**.
Application dependencies are locked in `uv.lock`; Windows/Linux use CPU PyTorch.
The Streamlit entry point remains `app.py`, while application code lives in the
installed `skripsi_rag` package under `src/`.

## Windows environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). This command
uses a pinned uv version even if an older uv is on PATH, downloads the required
Python when necessary, creates `.venv`, and installs the project in editable mode:

```powershell
uvx --from uv==0.13.8 uv sync --locked
```

To use the local Orion package, place its original subfolders under
`data/private/orion/`, then initialize once:

```powershell
.\.venv\Scripts\python.exe scripts/ingest_assets.py --orion-package ./data/private/orion --validate-only
.\.venv\Scripts\python.exe scripts/ingest_assets.py --orion-package ./data/private/orion
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Skip ingestion if the database is already initialized. Re-importing the package
upserts the supplied Orion IDs and can overwrite manual edits to those records.
It validates asset schema, software references, catalog hashes, source attribution
and primary asset/document mappings before writing. Proposed cross-asset links,
synthetic S01/S02 scenarios and evaluation labels are excluded.

Without the private package, the public example inventory can be initialized
using `scripts/ingest_assets.py --v2`. Document RAG still requires an authorized
context corpus; the inventory alone is not document evidence.

Use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe` on Linux. No
PowerShell activation script is required. The first embedding-model download
requires internet access. Enter the Gemini API key and model in the UI Settings;
explicit cloud consent is required for live assessments.

## Docker

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with
Linux containers and Compose, or use Docker Engine/Compose on Linux with daemon
access. The source package is private and is not included in the image or GitHub
repository. Put it in `data/private/orion/`, or configure `ORION_PACKAGE_PATH` in
a local `.env` copied from `.env.example`.

```powershell
docker compose --profile setup run --rm --build init
docker compose up -d app
```

Open [http://localhost:8501](http://localhost:8501). The one-time `init` service
mounts the package read-only and shares the four named volumes with the app:
ChromaDB, history/data, NVD snapshots and HuggingFace model cache. Normal app
restarts do not re-ingest data. Native runtime state is separate from Docker
volumes and is not baked into the image. The app runs as a non-root user and
publishes its port on localhost only.

For an inventory-only demo without the private package:

```powershell
docker compose run --rm --build app python scripts/ingest_assets.py --v2
docker compose up -d app
```

This demo still needs organizational documents before document-RAG assessment.

```powershell
docker compose logs -f app
docker compose down
```

`down` preserves named volumes. Set `APP_PORT=8502` in `.env` if another app uses
8501. `GEMINI_MODEL` optionally changes the model default. API keys are entered in
the UI and are not part of the image or Compose configuration. Rebuild after
code changes with `docker compose up -d --build app`.

## Custom sources

```powershell
.\.venv\Scripts\python.exe scripts/ingest_assets.py --v2 --asset-csv ./data/private/assets.csv
.\.venv\Scripts\python.exe scripts/ingest_assets.py --context-jsonl ./data/private/context.jsonl --validate-only
.\.venv\Scripts\python.exe scripts/ingest_assets.py --context-jsonl ./data/private/context.jsonl
```

`--persist-directory PATH` isolates an experiment database. Use the matching
directory in the experiment runner; the default UI uses `./chroma_db`.

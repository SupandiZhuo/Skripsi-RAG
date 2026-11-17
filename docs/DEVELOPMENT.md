# Development and GitHub preparation

The project uses a standard `src/skripsi_rag/` package. Keep UI in `ui.py`, core
assessment/retrieval/scoring modules independent of Streamlit, operational CLI
entry points in `scripts/`, and research batch/evaluation tooling in `evaluation/`.
Run `uv sync` after cloning or changing package/dependency configuration; do not
rely on root-level Python-module imports or manual `PYTHONPATH` changes.

## Checks

```powershell
uvx --from uv==0.13.8 uv sync --locked
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
git status --short
```

Four existing tests require public NVD snapshot fixtures that are absent under
`tests/fixtures/`. This reorganization does not fabricate those archived inputs
or their hashes. Other tests use offline fakes/temporary directories; they are
not an empirical accuracy evaluation.

## Dependencies

Edit `pyproject.toml`; then refresh the lock and generated dependency-only exports:

```powershell
uvx --from uv==0.13.8 uv lock
uvx --from uv==0.13.8 uv export --locked --no-dev --no-emit-project --emit-index-url -o requirements/runtime.txt
uvx --from uv==0.13.8 uv export --locked --group dev --no-emit-project --emit-index-url -o requirements/dev.txt
uvx --from uv==0.13.8 uv sync --locked
```

`uv sync` also installs this package. If using pip instead, install the exported
dependencies and then install the project with `pip install --no-deps -e .`.
Docker installs the package without editable mode after copying its sources.

## Before pushing

Review the staged file list and diff. Private VA reports, the Orion source
package, provider credentials/config, `.env`, model/database files, assessment
history, local exports, build metadata and caches are excluded by `.gitignore`.
Only the synthetic inventory under `data/examples/` is intended as public data.
The repository does not grant permission to redistribute company source data.

```powershell
git add -A
git diff --cached --stat
git diff --cached --name-only
git diff --cached --check
```

Commit and push only after reviewing those changes and selecting your own remote
and branch. Reorganization does not itself commit, push, or rewrite Git history.

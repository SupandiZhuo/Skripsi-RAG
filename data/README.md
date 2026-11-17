# Data layout

- `examples/`: public, synthetic asset CSVs supplied with the prototype. The v1
  file was originally named `asset_data.csv`; its contents and provenance were
  preserved during the move.
- `private/`: local input only, excluded from Git and Docker build contexts.
  Place the authorized Orion package in `private/orion/`, VA CSV exports in
  `private/va-reports/`, and optional local provider settings in
  `private/tooling/`. These directories are not part of a GitHub clone.
- `assessment_history.json` and `va_live_test_results/`: local generated results,
  also excluded from Git. They can contain target locations and organizational
  evidence; the source CSVs and company records are not public examples.

The existing local inputs have been moved without changing file contents. The
application's existing `chroma_db/`, NVD cache and history locations are unchanged.
Do not force-add private inputs, database files, model files, credentials or live
test exports to Git. Distribute authorized data separately from the code.

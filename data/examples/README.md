# Public example assets

- `asset_data_v1.csv`: the original seven synthetic assets (previously
  `asset_data.csv`).
- `asset_data_v2.csv`: the same synthetic inventory in Asset Schema v2.

These files are demonstration data, not company evidence or independent expert
labels. Their bytes and existing provenance fields are unchanged.

From the project root, initialize the v2 example inventory with:

```powershell
.\.venv\Scripts\python.exe scripts/ingest_assets.py --v2
```

The asset CSV alone does not provide the separate organizational document corpus.
For document RAG, ingest authorized attributed context JSONL or a local asset
package as described in [setup](../../docs/SETUP.md).

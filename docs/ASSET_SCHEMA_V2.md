# Asset Knowledge Base — Schema v2

Schema v2 for the organizational asset knowledge base used as the RAG
context source in the Context-Aware Vulnerability Assessment pipeline
(NVD + RAG).

Status (Phase 2):

- **Phase 2A** implemented the schema validator, the sample dataset, and a
  dedicated ingestion pipeline (`ingest_assets.py --v2`) that embeds
  `data/asset_data_v2.csv` into a NEW ChromaDB collection
  `corporate_assets_v2` (deterministic `asset:<asset_id>` IDs, upsert,
  idempotent). The original `corporate_assets` collection and the v1
  ingestion workflow were not modified.
- **Phase 2B** switched the live pipeline to the v2 collection: `app.py`
  loads `corporate_assets_v2`, `rag_retrieval.CONTEXT_METADATA_FIELDS`
  exposes the full v2 evidence, and the LLM prompt carries the schema-v2
  baseline rules (declared CR/IR/AR, `unknown` handling, no MAV inference).

## Design goals

1. **Backward compatibility.** Every v1 column is retained with the same
   name, meaning, and allowed values. A v2 row is a strict superset of a
   v1 row, so existing code paths keep working on v2 data once migration
   proceeds in Phase 2.
2. **No fabrication.** Fields without authoritative source information are
   populated with the explicit literal `unknown` (or `X` = Not Defined for
   CVSS requirement metrics) — never with plausible-looking guesses.
3. **Structured context over free text.** Fields are chosen so they can
   serve as evidence for the LLM's CVSS Environmental Requirements
   (CR/IR/AR) and organizational priority, and can later be logged/exported
   for programmatic evaluation.
4. **Provenance.** Every row records where it came from, when it was last
   updated, and whether it is synthetic sample data.

## Sample dataset

`data/asset_data_v2.csv` contains the seven original sample assets
(SRV-001 … SRV-007) migrated from `asset_data.csv`. **All sample data is
synthetic** (`is_synthetic=true`, `source=synthetic_sample`); it does not
describe any real organization's assets.

Migration policy applied to the sample rows:

- v1 values are copied unchanged (asset names, IPs, zones, criticality,
  sensitivity, Indonesian descriptions).
- A new field is populated only when the original v1 description explicitly
  states the fact (e.g. SRV-001 "terisolasi … dari internet" →
  `internet_exposed=no`; SRV-002 "menghadap langsung ke jaringan internet
  publik" → `internet_exposed=yes`; SRV-006 is described as a "CMS
  WordPress" server → `software_name=WordPress`).
- Everything else is `unknown`, and all three CIA requirement fields are
  `X` (Not Defined) because the sample data does not state them.
- Owners, vendors, versions, and CPEs are **not** invented.

## Field reference

Types are logical (the CSV stores everything as text):

| # | Field | Type | Required | Allowed values | Purpose |
|---|-------|------|----------|----------------|---------|
| 1 | `asset_id` | string | **yes** | unique, non-empty (convention `SRV-###`) | Stable primary key for the asset; used to cite evidence. |
| 2 | `asset_name` | string | **yes** | non-empty | Human-readable name; primary semantic-search content together with `description`. |
| 3 | `ip_address` | string | **yes** | IPv4/IPv6 literal or `unknown` | Network identity of the asset (v1 field). |
| 4 | `network_zone` | enum | **yes** | `Internal`, `DMZ`, `Restricted` | Segmentation zone the asset sits in (v1 field, values unchanged). |
| 5 | `business_criticality` | enum | **yes** | `Low`, `Medium`, `High`, `Mission Critical` | Organizational criticality of the asset (v1 field, values unchanged). Feeds organizational priority. |
| 6 | `data_sensitivity` | enum | **yes** | `Public`, `Internal`, `Confidential`, `Restricted` | Highest sensitivity of data handled (v1 field, values unchanged). Evidence for CR. |
| 7 | `description` | string | **yes** | non-empty (may be Indonesian) | Free-text functional description (v1 field); embedded for retrieval. |
| 8 | `environment` | enum | **yes** | `production`, `staging`, `development`, `unknown` | Deployment environment. Declared in `rag_retrieval.CONTEXT_METADATA_FIELDS` but missing from v1 data; now supplied. |
| 9 | `internet_exposed` | enum | **yes** | `yes`, `no`, `unknown` | Whether the asset is reachable from the public internet. Declared in `rag_retrieval.CONTEXT_METADATA_FIELDS`; evidence for CR/MAV discussion (never an automatic MAV rule). |
| 10 | `asset_type` | enum | **yes** | `database`, `web_server`, `domain_controller`, `file_server`, `cms`, `other`, `unknown` | Coarse technical role; helps match NVD/CPE applicability later. |
| 11 | `owner` | string | no | team/role name or `unknown` | Owning organizational unit (asset identification & ownership). Not invented in samples. |
| 12 | `owner_contact` | string | no | contact or `unknown` | Operational contact for the owner. Not invented in samples. |
| 13 | `business_function` | string | no | text or `unknown` | Business process the asset supports; evidence for criticality/priority. |
| 14 | `vendor` | string | no | vendor name or `unknown` | Software vendor; input to CPE construction. Not invented in samples. |
| 15 | `software_name` | string | no | product name or `unknown` | Installed software/product of primary interest. |
| 16 | `software_version` | string | no | version string or `unknown` | Version of `software_name`; required for exact CPE matching. Not invented in samples. |
| 17 | `cpe` | string | no | CPE 2.3 URI (`cpe:2.3:…`) or `unknown` | NVD-compatible CPE match string for applicability matching (Phase 2). Not invented in samples. |
| 18 | `confidentiality_requirement` | enum | **yes** | `X`, `L`, `M`, `H` | Asset-declared CVSS v3.1 CR baseline. `X` (Not Defined) unless the organization explicitly states it. |
| 19 | `integrity_requirement` | enum | **yes** | `X`, `L`, `M`, `H` | Asset-declared CVSS v3.1 IR baseline. |
| 20 | `availability_requirement` | enum | **yes** | `X`, `L`, `M`, `H` | Asset-declared CVSS v3.1 AR baseline. |
| 21 | `data_classification` | string | no | classification label or `unknown` | Governing classification/regime of the data stored (e.g. `PII`, `PCI-DSS`) when stated. |
| 22 | `last_updated` | date | **yes** | `YYYY-MM-DD` | When this row's content was last revised; supports CMDB freshness (CM-8). |
| 23 | `source` | string | **yes** | e.g. `synthetic_sample` | Provenance system of record for the row. |
| 24 | `source_reference` | string | **yes** | free text | Pointer to the originating artifact (e.g. migration from `asset_data.csv`). |
| 25 | `is_synthetic` | boolean | **yes** | `true`, `false` | Marks sample/illustrative data so it is never mistaken for production inventory. |
| 26 | `schema_version` | string | **yes** | `2.0` | Version stamp enabling future schema evolution. |

Notes:

- Fields 1–7 are the v1 columns, retained unchanged (backward
  compatibility).
- Fields 8, 9 were already declared in `rag_retrieval.CONTEXT_METADATA_FIELDS`
  but had no source data; they are now first-class schema fields.
- Unknown values use the literal `unknown` (string fields) or `X`
  (CVSS requirement metrics). Empty cells are not used for unknown data.

## Relationship to organizational context and CVSS environmental requirements

The pipeline separates **Technical Severity** (NVD Base metrics, untouched)
from **Organizational Priority** (P1–P4) and the CVSS **Environmental
Requirements** (CR/IR/AR) the model assesses.

How schema v2 fields feed that assessment (as *evidence*, not automatic
overrides):

- `confidentiality_requirement` / `integrity_requirement` /
  `availability_requirement` are the asset's declared CVSS v3.1
  Environmental Requirement codes. When set (not `X`), they are the
  deterministic organizational baseline the model must respect and justify
  against; when `X`, the model assesses CR/IR/AR from the remaining context.
  The model still returns its own CR/IR/AR; the numerical Environmental
  score is always computed deterministically by the `cvss` library.
- `data_sensitivity`, `data_classification`, and `business_function`
  justify the confidentiality/integrity dimensions.
- `business_criticality`, `environment`, and `availability_requirement`
  justify the availability dimension and organizational priority.
- `internet_exposed` and `network_zone` give exposure evidence for the
  assessment. Per the prototype design, **no metric is inferred
  automatically from them** — in particular internal/restricted placement
  must not silently become `MAV:L`; `MAV` stays undefined (`X`).
- `vendor` / `software_name` / `software_version` / `cpe` support NVD
  vulnerability applicability matching (Phase 2), keeping technical
  relevance separate from organizational priority.
- Retrieval evidence: all fields flow through ChromaDB metadata, so the
  final report can show exactly which retrieved context influenced the
  result (score + field values).

## References

- **CIS Controls v8.1**
  - Control 1 — Inventory and Control of Enterprise Assets: asset
    identification (`asset_id`, `asset_name`, `ip_address`, `owner`),
    authoritative source and freshness (`source`, `last_updated`).
  - Control 2 — Inventory and Control of Software Assets: `vendor`,
    `software_name`, `software_version`, `cpe`.
  - Control 3 — Data Protection: `data_sensitivity`,
    `data_classification`, CIA requirements.
- **NIST SP 800-53 Rev. 5, CM-8 (System Component Inventory)**: attributes
  for component identification, ownership, placement (`network_zone`,
  `environment`), and currency (`last_updated`, `source`).
- **FIRST CVSS v3.1 Specification** (Environmental Metric Group):
  CR/IR/AR codes `X|L|M|H` used by fields 18–20; the `cvss` Python library
  performs all numerical scoring. CPE 2.3 URIs follow the NVD CPE naming
  specification for applicability matching.

## Validation

Lightweight stdlib validator (no new dependencies):

```powershell
python asset_schema_v2.py                    # validates data/asset_data_v2.csv
python -m pytest test_asset_schema_v2.py -q   # automated tests
```

`asset_schema_v2.validate_csv()` returns a list of error messages (empty
list = valid), reporting row number, `asset_id`, field, and
expected-vs-actual values.

## Migration from v1

| v1 column | v2 treatment |
|-----------|--------------|
| `asset_id`, `asset_name`, `ip_address`, `network_zone`, `business_criticality`, `data_sensitivity`, `description` | Retained byte-identical |
| *(none)* | + 19 new fields, `unknown`/`X` where no source data |

`asset_data.csv` (v1) remains the dataset ingested into the existing
`corporate_assets` ChromaDB collection until Phase 2.

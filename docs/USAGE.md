# VA CSV workflow

1. Open Vulnerability Analysis and upload an existing Acunetix **Vulnerabilities**
   CSV. Locations exports are rejected. Supported input is UTF-8/BOM CSV, up to
   10 MB and 10,000 findings.
2. Review the preview and row diagnostics. Confirm the organizational asset for
   every target. URL/IP or semantic similarity does not establish asset identity.
3. Configure Gemini in Settings, grant cloud consent when appropriate, and run
   the batch. CVEs are taken from explicit `CVEList` identifiers; they are not
   assigned manually or guessed from finding titles.
4. Review results and source evidence. Download CSV/JSON traces or save individual
   assessments to Analysis History.

The importer preserves CVE-less, invalid, scanner-marked false-positive and
unmapped findings. NVD/API/schema failures remain visible instead of disappearing
from the report. Repeated occurrences retain their locations. The recognizable
Acunetix layout that omits two empty trailing CVSS4 cells is repaired with an
explicit diagnostic; other column mismatches are retained for review.

Malformed NVD/assessment adapter responses fail only their affected cases.
JSON evidence records failure stages/types and exception classes; the Issues
column includes those diagnostics without raw exception messages. Python's CSV
field-size limit also applies; oversized-field errors are reported separately
from malformed quoting or an absent header.

Scanner severity, NVD Technical Severity, deterministic Environmental CVSS, and
Organizational Priority are separate values. Only supported NVD CVSS v3.1 Base
vectors enter the scoring pipeline. Unknown applicability gives investigation
priority and a provisional contextual score; it does not authorize remediation
priority. Scanner/inventory disagreements are flagged for review.

The model receives minimized NVD information and organizational evidence, not
raw scanner requests/cookies. Retrieved documents are asset-filtered and retain
source IDs, relevance scores, accepted/rejected evidence and per-metric citations.
The model does not compute numerical CVSS scores. Missing evidence can produce
`X`; it is not a reason to invent facts or force a score change.

Local inputs belong in `data/private/`. History and exported evidence can contain
target and organization information and must be reviewed before sharing.
History is a JSON array of assessment traces, including evidence and model
payloads. Saves and deletions refuse malformed history rather than replacing it;
repair or restore the file before retrying. Writes use atomic replacement, but
this single-user store does not coordinate concurrent writers.
Retrieval relevance is a similarity score, not a calibrated probability. The
stored `retrieval_confidence` field retains its old name for history compatibility.

Asset metadata edits preserve the existing software-inventory JSON without
rewriting it. Invalid inventory JSON, non-object entries or records without
software names show an editor error instead of silently becoming an empty
inventory. Repair the inventory source before editing its software records.
A primary CPE of `unknown` is accepted as missing evidence, not a verified CPE.

Orion demo business categories remain synthetic; questionnaire summaries are
reported context, not independently verified company policy. The application
does not initiate scans or connect to the scanner API.

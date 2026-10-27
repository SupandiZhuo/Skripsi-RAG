# Active Organizational Priority Rubric

## Source of truth and scope

`organizational_priority.py:calculate_organizational_priority` is the **only
active deterministic rubric**. `priority_rubric.py` is a legacy Phase 2C
baseline retained for historical comparisons and compatibility; its 70/50/30
thresholds and six-factor weights are not the active algorithm.

The active algorithm identifier is `organizational-priority-v2.0`, exposed
as `RUBRIC_VERSION` and included in every structured result.

This is a research-prototype policy, defined by the researcher. The weights
and thresholds are **not expert-approved or empirically validated**. Expert
review, independent labels, sensitivity analysis and a frozen pre-test policy
remain required before claiming organizational validity. Reproducibility is
not evidence of correctness or expert agreement.

Technical Severity (NVD baseline and deterministic contextual CVSS) and
Organizational Priority are separate outputs. This module never calculates,
changes or replaces CVSS vectors/scores. It only bands a supplied numerical
score and combines evidence about an individual asset.

## Inputs

- `asset_metadata`: structured asset evidence, including `asset_id`, business
  criticality, exposure, sensitivity, environment, and declared CIA baseline.
- `base_score`: NVD technical baseline; `context_score`: deterministic
  contextual/environmental score. The active policy prefers a valid
  contextual score (including a valid zero) and otherwise falls back to the base score. Numerical
  CVSS calculation and input validation belong upstream, not to an LLM.
- `applicability`: result object or mapping with a `status` equal to
  `Confirmed`, `Unknown`, or `Not Applicable` (including the enum values).
- `graph_dependencies`: accepted for API compatibility but **ignored**, even
  if populated. No relationship data, points, contributions, or missing-graph
  penalties enter the active policy. Graph scoring requires a separately
  reviewed future policy and verified relationship evidence.
- `cve_id`: optional traceability identifier.

## Seven-factor rubric: maximum exactly 100

`FACTOR_MAX_POINTS` defines the factor maxima; `MAX_SCORE` is their sum and
an import-time assertion requires it to equal 100. The score is the raw sum
of active contributions: no clipping or hidden normalization is used.

| Factor | Evidence | Points | Maximum |
|---|---|---|---:|
| Vulnerability severity | Preferred contextual score, otherwise NVD base | Critical 30; High 23; Medium 12; Low 3; None (0.0) 0 | 30 |
| Business criticality | `business_criticality` | Mission Critical 25; High 18; Medium 10; Low 3 | 25 |
| Internet exposure | `internet_exposed`, `network_zone` | yes 15; otherwise DMZ 10; otherwise no / Internal / Restricted 0 | 15 |
| Data sensitivity | `data_sensitivity` | Restricted 8; Confidential 5; Internal 2; Public 0 | 8 |
| Environment | `environment` | production 5; staging / development 0 | 5 |
| Declared CIA baseline | confidentiality / integrity / availability requirement | any H 2; otherwise 0 | 2 |
| CPE applicability | applicability result | Confirmed 15; Unknown 5 (investigation only) | 15 |

The graph factor has been removed. Severity was reweighted from
20/15/8/2 to 30/23/12/3 to make the seven active maxima total exactly 100;
the other active weights and thresholds are retained. This is an explicit
prototype design choice, not a fitted or validated organizational policy.
Business criticality remains the largest exclusively organizational factor.

Severity bands: Critical >= 9.0, High >= 7.0, Medium >= 4.0, Low > 0.
A valid zero is the None severity band and contributes zero points, not
missing evidence. Missing, non-finite or out-of-range severity is disclosed
in `missing_evidence` when no valid fallback exists.

**Thresholds:** P1 (Urgent) >= 80; P2 (High) >= 60;
P3 (Normal) >= 40; P4 (Low) < 40.
The same bands rank investigations, but never imply confirmed vulnerability.

## Decision states and missing evidence

Applicability is evaluated before remediation classification:

| Applicability / evidence | `status` | `priority` | `investigation_priority` | `score` / `level` |
|---|---|---|---|---|
| Not Applicable | `not_applicable` | `None` | `None` | both `None` |
| Unknown + known criticality | `needs_verification` | `None` | P1–P4 | investigation score / `None` |
| Unknown + missing criticality | `needs_verification` | `None` | `None` | both `None` |
| Confirmed + known criticality | `complete` | P1–P4 | `None` | remediation score / Urgent, High, Normal, Low |
| Missing/invalid applicability, or Confirmed + missing criticality | `incomplete` | `None` | `None` | both `None` |

- **Not Applicable is not P4**: exclude that CVE/asset pair from the
  remediation queue. Do not interpret the empty score as zero risk or zero
  technical severity. Technical CVSS remains independently reportable.
- **Unknown never receives a remediation priority**, even for a critical,
  exposed asset. `investigation_priority` ranks product/version verification,
  not patch deployment. Its score uses the same active contributions but
  only 5 applicability points, so its attainable maximum is 90/100.
- Known business criticality is mandatory for either scored decision.
  Confirmed or Unknown applicability can still support scoring without a
  numerical severity input, preserving the existing partial-evidence policy;
  missing severity remains in `missing_evidence`. A `complete` decision means
  the remediation gate is satisfied, not that all evidence is populated.
- Unknown exposure, sensitivity, environment, and CIA evidence contribute
  zero, are disclosed in `unknown_factors`, and do not imply low risk.
- Exposure uses yes first, then DMZ, then explicit no or Internal/Restricted.
  Therefore DMZ contributes 10 even when `internet_exposed=no`; this is the
  existing conservative internet-adjacent rule, not an inferred CVSS MAV
  change. Unknown exposure remains explicitly disclosed.
- CIA evidence: any H contributes 2; otherwise an available L/M/X contributes
  zero; absent/unrecognized evidence is reported as unknown. X means no
  declared adjustment, not proof of a low CIA requirement.
- Inactive graph data is not included in `unknown_factors`.

## Structured result and UI/history integration

`OrganizationalPriorityResult` is an immutable dataclass. Existing fields are
preserved, with an appended defaulted `investigation_priority: Optional[str]`
field for caller compatibility:

- `priority`: remediation P1–P4 **only**; otherwise `None`.
- `status`: `complete`, `incomplete`, `not_applicable`, or `needs_verification`.
- `investigation_priority`: provisional investigation P1–P4 **only** for
  Unknown with sufficient evidence; otherwise `None`.
- `score`: integer sum of disclosed contributions for the decision type,
  or `None` if excluded/unscorable; `max_score`: 100.
- `level`: remediation label only; `None` for investigations or no decision.
- `rule_contributions`: immutable entries with `rule`, `factor`, `points`,
  `max_points`, `evidence_value`, and `evidence_source`.
- `missing_evidence`, `unknown_factors`, `explanation`, `cve_id`, `asset_id`.

Integrators must branch on **status**, not replace a falsey `priority` with P4:

1. `complete`: show **Remediation priority** and its score/level.
2. `needs_verification`: show **Needs verification** and, when available,
   **Investigation priority** and **Investigation score**; never show this as
   a confirmed remediation rank. If no investigation band exists, display the
   missing evidence instead of inventing a band.
3. `not_applicable`: show **N/A — excluded from remediation**; do not put the
   pair in a P4 bucket.
4. `incomplete`: show **Insufficient evidence**, with the missing fields.

Persist/export `status`, nullable `priority`, nullable
`investigation_priority`, nullable `score` and `level`, `max_score`, evidence
contributions, unknown/missing evidence, explanation, CVE and asset IDs.
`dataclasses.asdict(result)` includes the added field for JSON conversion.
Older records without investigation/status semantics must not be silently
reclassified; retain provenance and distinguish legacy records from reruns.
Remediation metrics/ranking should include only confirmed `complete` results;
investigation metrics use a separate queue and label. Preserve the LLM's
priority suggestion separately from both deterministic decision types.

## Regression tests and remaining limitations

`tests/test_organizational_priority.py` covers the exact attainable maximum,
contribution accounting, graph-input inactivity, applicability gating,
separate investigation output, evidence gaps, and reproducibility. Test
execution is deferred until all implementation work is complete, per the
requested workflow; this document makes no passing-test claim.

No EPSS/KEV/exploitation intelligence or graph relationships enter this
rubric. The numerical severity is supplied upstream and may depend on the
quality of environmental metrics. The policy does not prove asset identity,
inventory completeness, or ground-truth applicability. Optional unknown
factors can suppress scores and must remain visible during review.

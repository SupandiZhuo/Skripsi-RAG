# Organizational Priority Rubric (Phase 2C)

Deterministic, explainable rules for **Organizational Remediation Priority**
(P1–P4) in the Context-Aware Vulnerability Assessment pipeline
(NVD + RAG). Implemented in `priority_rubric.py`
(`calculate_organizational_priority`), a pure, stdlib-only function.

## Why deterministic?

Before Phase 2C, `organizational_priority` was entirely LLM judgment
(`llm_assessment.ContextualAssessment`), which is not reproducible or
defensibly auditable for the thesis. The rubric now computes the displayed
priority from verified evidence; the LLM's priority remains in the pipeline
unchanged as a **model suggestion** shown next to the rubric result for
comparison and later evaluation.

## Separation from CVSS (must not blur)

| Dimension | Source | Meaning |
|---|---|---|
| Technical Severity (Base/Threat/Environmental CVSS) | NVD + `cvss` library | How severe the vulnerability is technically |
| Organizational Priority (P1–P4) | This rubric | In what order **this organization** should remediate |

The rubric never modifies a CVSS vector or score, and the CVSS formulas are
never touched. It only *bands* the already-verified NVD CVSS v3.1 base score
into the FIRST qualitative severity scale (Critical ≥ 9.0, High ≥ 7.0,
Medium ≥ 4.0, Low > 0) as one bounded input among organizational factors.

## Inputs (verified evidence only)

1. **Asset context** — the retrieved Asset Schema v2 metadata
   (`rag_retrieval.filtered_metadata` output, or a v1-metadata dict).
2. **NVD CVSS v3.1 base score** — `0.0`/`None`/out-of-range is treated as
   "no severity evidence" (NVD reports `0.0` when no CVSS data exists) and
   makes the result **incomplete**.

No EPSS/KEV/exploit intelligence, no CPE matching, no description parsing,
and no inferred asset attributes are used in this phase. Missing evidence is
surfaced, never fabricated.

## Rubric

Maximum **100 points** across six factors:

| # | Factor | Evidence field(s) | Points | Kind |
|---|--------|-------------------|--------|------|
| 1 | Vulnerability severity | NVD `base_score` banded | Critical 35, High 27, Medium 14, Low 4 | **required** |
| 2 | Business criticality | `business_criticality` | Mission Critical 30, High 22, Medium 11, Low 3 | **required** |
| 3 | Exposure | `internet_exposed` (fallback: `network_zone`) | explicit `yes` 20; unknown + DMZ 12; explicit `no` / unknown + Internal/Restricted 0 | **required** |
| 4 | Data sensitivity | `data_sensitivity` | Restricted 8, Confidential 5, Internal 2, Public 0 | optional |
| 5 | Environment | `environment` | production 5; staging/development 0 | optional |
| 6 | Declared CIA baseline | `confidentiality_requirement` / `integrity_requirement` / `availability_requirement` | any declared `H` → 2 | optional |

**Thresholds:** P1 ≥ 70 · P2 ≥ 50 · P3 ≥ 30 · P4 < 30.

### Weight justifications

- **Severity capped at 35 (< 50% of max).** Technical severity is necessary
  urgency context, but a vulnerability can never reach P1 on severity alone;
  P1 requires organizational co-evidence (e.g. Mission Critical + Restricted
  data, or High criticality + internet exposure). This encodes the thesis'
  core claim that organizational priority is *separate from, but not
  isolated from*, technical severity.
- **Business criticality 30** is the heaviest organizational factor: it is
  the direct measure of business impact if the asset is compromised.
- **Exposure 20** reflects attack-path reachability as an amplifier; the
  zone-derived value (12) is deliberately below an explicit
  `internet_exposed=yes` (20), because inferred placement evidence outranks
  neither an explicit declaration nor carries the same weight.
- **Data sensitivity 8, production 5, CIA-H 2** are small, bounded boosts
  for regulated-data impact (e.g. PCI-DSS/PII in the sample dataset),
  deployment criticality, and declared organizational requirements. They can
  adjust a band, never dominate it.

### Exposure resolution & conflicting evidence

1. `internet_exposed=yes` → 20. If `network_zone` is Internal/Restricted,
   this is a **conflict**: the explicit declaration wins (higher-risk
   interpretation), and the conflict is recorded in the result.
2. `internet_exposed=no` → 0.
3. `internet_exposed` unknown/absent → derived from `network_zone`:
   DMZ → 12 (internet-adjacent segment); Internal/Restricted → 0
   (non-DMZ segment). The contribution's `evidence_source` marks this as
   zone-derived ("tersirat").
4. Neither field usable → the exposure gate fails → incomplete result.

Conflicts never silently downgrade risk.

### Unknown / missing evidence policy

- **Required factors** (severity, business criticality, exposure): if any
  cannot be determined, the result is explicit **incomplete** —
  `priority=None`, `status="incomplete"`, `score=None` — and the explanation
  lists exactly which evidence is missing. Missing information is never
  scored as low risk.
- **Optional factors** (sensitivity, environment, CIA baseline): unknown or
  absent values contribute 0 points but are listed in `unknown_factors`
  and named in the explanation, so a low score can never *hide* missing
  evidence. v1-style metadata (no v2 fields) still classifies, with the
  absent v2 fields reported as unknown.
- Available evidence is still recorded as contributions on incomplete
  results (transparency for debugging/evaluation).

### Design invariants (asserted in tests)

- A **Low**-severity vulnerability can never exceed P2, even on a
  max-stakes asset (max score 4+30+20+8+5+2 = 69 < 70).
- The sum of known contributions equals the reported score; the same inputs
  always produce an identical (frozen-dataclass-equal) result.

## Structured output

```python
@dataclass(frozen=True)
class RuleContribution:
    factor: str           # e.g. "business_criticality"
    value: str            # human-readable evidence value used
    points: int
    evidence_source: str  # e.g. "asset.business_criticality" or "nvd.base_score"

@dataclass(frozen=True)
class PriorityResult:
    priority: Optional[str]        # "P1".."P4", or None when incomplete
    status: str                    # "complete" | "incomplete"
    score: Optional[int]           # 0..100, None when incomplete
    max_score: int                 # 100
    rule_contributions: Tuple[RuleContribution, ...]
    unknown_factors: Tuple[str, ...]
    conflicts: Tuple[str, ...]
    explanation: str               # human-readable Indonesian summary
```

### Sample output (SRV-001 Payment Gateway Database, NVD base score 9.8)

```
priority: "P1"
status: "complete"
score: 73
rule_contributions:
  - vulnerability_severity: "Critical (skor dasar NVD 9.8)"  -> 35  [nvd.base_score]
  - business_criticality:    "Mission Critical"               -> 30  [asset.business_criticality]
  - exposure:                "internet_exposed=no"            ->  0  [asset.internet_exposed]
  - data_sensitivity:        "Restricted"                     ->  8  [asset.data_sensitivity]
unknown_factors: ("environment",)      # environment=unknown in the sample data
conflicts: ()
explanation: "P1 - skor prioritas 73/100. Kontribusi: kerentanan teknis ...;
              Faktor opsional dengan bukti tidak diketahui (dihitung 0 poin,
              bukan berisiko rendah): environment."
```

Same asset, **no NVD base score** (e.g. NVD has no CVSS data):

```
priority: None
status: "incomplete"
score: None
unknown_factors: ("vulnerability_severity", ...)
explanation: "Tidak dapat mengklasifikasikan prioritas secara meyakinkan
              (status: incomplete) karena bukti wajib berikut tidak tersedia:
              skor dasar CVSS NVD tidak tersedia/tidak valid ... Bukti yang
              hilang TIDAK diperlakukan sebagai risiko rendah; lengkapi
              metadata aset / data NVD lalu hitung ulang."
```

## Known limitations

- No EPSS/KEV/exploit intelligence and no CPE-based applicability matching
  (deferred by design); severity comes solely from the NVD base score.
- The environment factor is latent on the current sample dataset
  (`environment=unknown` on all 7 rows); it is reported in `unknown_factors`.
- Zone-derived exposure treats Internal/Restricted as non-exposed; a
  mislabeled zone would bias the exposure factor (mitigated by the explicit
  `internet_exposed` field when populated).
- Weights/thresholds are expert-designed for the prototype and should be
  calibrated against evaluation data (next pipeline phase) rather than
  treated as empirically validated.

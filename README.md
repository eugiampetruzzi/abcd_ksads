# abcd_ksads

Harmonization pipeline and codebooks for the **KSADS-COMP diagnostic data in the
Adolescent Brain Cognitive Development (ABCD) Study, release 7.0**.

The ABCD KSADS-COMP diagnoses require a chain of analytic decisions — how to read
administrative codes, whether to count past as well as current episodes, how to
combine parent and youth report, which instrument version to use, which modules
were administered when — that are individually reasonable but made differently
across studies. This repository contains the code that resolves each of those
decisions explicitly and a set of codebooks documenting the diagnosis variables.

This is the analysis code for a data-resource paper; the harmonized dataset
itself is deposited separately in the NIMH Data Archive (NDA).

## Layout

```
harmonize/    processing pipeline (Python 3)
codebooks/    variable maps and crosswalks (metadata only; no participant data)
```

### Pipeline (`harmonize/`)

| Script | Step |
|---|---|
| `01_resolve_missingness.py` | resolve every diagnosis cell to {positive, administered-negative, not-administered, no-record} |
| `02_administration_calendar.py` | module × wave × informant administration map |
| `03_category_crosswalk.py` | diagnosis → DSM-category mapping + caseness engine (status / informant / threshold / membership toggles) |
| `04_version_provenance.py` | KSADS-COMP 1.0 vs 2.0 provenance per cell |
| `06`–`08` | multiverse grid, per-construct summary, single-lever sensitivity |
| `09_missingness_error.py` | quantifies the 555-as-0 error |
| `10_correctness_anchor.py` | internal-consistency and correctness checks |
| `11_paper_numbers.py` | single machine-readable source of every reported number |
| `12_build_bids_dataset.py` | materializes the harmonized dataset in BIDS phenotype format |
| `13_technical_validation.py` | correctness, face-validity (CDC), parent–youth concordance |
| `14_module_overscreening.py` | module-level screening rates (core criteria) |
| `fig_*.py` | figures |

### Codebooks (`codebooks/`)

Structural metadata describing the 230 KSADS-COMP diagnosis variables — variable
names, labels, value labels, administrative codes, module/wave coverage, the
1.0→2.0 version crosswalk, and the diagnosis-to-category mappings. These contain
no participant-level data.

## Data

This repository contains **code and codebooks only**. The underlying ABCD 7.0
KSADS-COMP data are access-restricted through the NIMH Data Archive and are not
redistributable. Absolute data paths in the scripts reflect the analysis
environment and must be repointed to a local ABCD mirror. Resolved caseness is
produced by the Layer 3 engine; non-administration (555) is never treated as a
negative, and all prevalence is computed over the administered denominator.

## Requirements

Python 3.12 with pandas, polars, numpy, scikit-learn, scipy, pyarrow,
python-docx, and matplotlib.

Developed with the assistance of Claude Code (Anthropic).

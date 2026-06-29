# abcd_ksads

Harmonization and analysis code for the ABCD Study KSADS-COMP diagnostic data
(release 7.0). The pipeline resolves administrative-missingness codes, reconstructs
the administration calendar, crosswalks the 230 diagnosis variables to DSM
categories, and tracks instrument-version provenance. It then quantifies how
diagnostic operationalization changes both the prevalence of disorders and the
associations of caseness with demographic, psychosocial, and neuroimaging measures.

## Data access

All inputs are access-controlled and are not included in this repository. An approved
ABCD Data Use Certification is required (https://nbdc-datahub.org/data-access-process).
The pipeline expects:

- Release 7.0: the tabulated KSADS-COMP source (BIDS `rawdata/phenotype`).
- Release 5.1 `core`: `imaging/`, `culture-environment/`, and `novel-technologies/`
  (baseline predictors for the correlate analysis).

## Setup

```bash
pip install -r requirements.txt
export ABCD_70=/path/to/abcd/release-7.0
export ABCD_51=/path/to/abcd/release-5.1/core
```

Paths are centralized in `config.py`; no script hard-codes a location.

## Pipeline

```
harmonize/01_resolve_missingness.py      resolve the four missingness states
harmonize/02_administration_calendar.py  module x wave x informant administration
harmonize/03_category_crosswalk.py        diagnosis variables to DSM categories
harmonize/04_version_provenance.py        KSADS-COMP 1.0 / 2.0 flags
harmonize/06_multiverse_spec.py           prevalence under every operationalization
harmonize/07_multiverse_summary.py        per-construct fold and range summary
harmonize/08_single_lever.py              one-decision-at-a-time prevalence shifts
harmonize/09_inferential_multiverse.py    predictor x construct x spec logistic models
harmonize/10_informant_discrepancy.py     caregiver vs youth caseness and Cohen's kappa
harmonize/aux_missingness_audit.py        quantify miscoded non-administration (555)
harmonize/aux_anxiety_decomposition.py    anxiety prevalence by constituent diagnosis
harmonize/11_paper_numbers.py             collate manuscript numbers to paper_numbers.json
harmonize/12_build_bids_dataset.py        BIDS-style harmonized dataset
harmonize/13_technical_validation.py      data-descriptor validation checks
harmonize/14_module_overscreening.py      branch-skip diagnostics
harmonize/15_export_analysis_csv.py       analysis-ready CSV release

figures/   make_fig12, make_fig_informant, make_fig_bwas_style,
           make_fig_inferential, make_fig_catcalendar
tables/    build_table1_checklist, build_table_categories
```

Scripts 01 through 11 write intermediates to `harmonize/derivatives/` (git-ignored).
The figure scripts read those intermediates; `11_paper_numbers.py` collates every
in-text statistic into `paper_numbers.json`.

## Software

Python 3.12 with pandas, numpy, pyarrow, statsmodels, matplotlib, and openpyxl.

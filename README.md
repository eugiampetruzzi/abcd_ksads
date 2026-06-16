# abcd_ksads

Harmonization pipeline and codebooks for the KSADS-COMP diagnostic data in the
Adolescent Brain Cognitive Development (ABCD) Study, release 7.0.

The pipeline resolves administrative missingness codes, maps the administration
calendar across waves and informants, crosswalks the 230 diagnosis variables to
DSM categories, and builds analysis-ready caseness tables. Non-administration
(555) is never counted as a negative; prevalence is computed over the
administered denominator.

## harmonize/

Run in order. Each script reads `codebooks/` and writes to `derivatives/`;
`04` writes the dataset to `dataset/`.

| Script | Output |
|---|---|
| `01_resolve_missingness.py` | per-cell state: positive / administered_negative / not_administered / no_record |
| `02_administration_calendar.py` | module x wave x informant administration map |
| `03_category_crosswalk.py` | diagnosis to DSM-category crosswalk and caseness engine |
| `04_build_dataset.py` | current and ever-met caseness tables (parent / youth / either), sessions, and participants, with NDA identifier columns |

## codebooks/

Structural metadata only, no participant data.

| File | Contents |
|---|---|
| `ksads_variable_map.csv` | variable names, labels, value/admin codes, module, informant, layer |
| `ksads_version_crosswalk.csv` | 1.0 vs 2.0 provenance per merged variable |
| `ksads_administration_calendar.csv` | administered counts and flags by module/wave/informant |
| `ksads_category_crosswalk.csv` | diagnosis to category and broadband mapping |
| `ksads_caseness_candidates.csv` | proposed case definition per disorder |

## Data

ABCD 7.0 KSADS-COMP data are access-restricted through the NIMH Data Archive and
are not redistributable. Repoint the data path at the top of `01` and `04` to a
local ABCD mirror.

## Requirements

Python 3.12 with pandas, numpy, and pyarrow.

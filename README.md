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
The pipeline expects a number of tabular files, which are listed in the [Manifest](codebooks/phenotype_manifest.txt), which can be downloaded from the NBDC.  

**TODO**: Add download instructions

You should first create a base data directory for the project (preferably separate from the code repository).  The tabular files should be located within `rawdata/phenotype/` under that main directory.

## Setup

1) [Install the uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

2) Clone this repository: `git clone ...`

3) Set up the environment

```bash
cd abcd_ksads
uv sync
```

4) Specify the path to your ABCD base data directory in the `.env` file.  First, copy the `.env.example` file to `.env`:


```bash
cp .env.example .env
```

Then, edit the .env file and change the example path for the `ABCD_70` release to your actual path.

5) Run the full workflow:

```bash
make all
```

The results will be placed into a folder called `derivatives` within the ABCD data directory.

## Running via Docker / Apptainer

The whole workflow can be run in a container instead of a local `uv` install.
No data is baked into the image: you bind-mount your ABCD base data directory
into the container at `/data` and point `ABCD_70` at it. Inputs are read from
and all outputs (`derivatives/`, `figures/`, `tables/`, …) are written back
under that mounted directory, exactly as in a local run.

### Docker

Build the image, then run the full pipeline (replace `/path/to/ABCD/basedir`
with your host data directory):

```bash
docker build -t abcd_ksads:latest .

docker run --rm --user $(id -u):$(id -g) \
    -v /path/to/ABCD/basedir:/data -e ABCD_70=/data \
    abcd_ksads:latest all
```

`--user $(id -u):$(id -g)` makes the generated files owned by you rather than
root. Append a different target to run a single stage, e.g. `... abcd_ksads:latest figures`.

If you have set `ABCD_70` in your `.env`, the Makefile provides equivalent
wrappers:

```bash
make docker-build
make docker-run              # runs `all`
make docker-run TARGET=figures
```

### Apptainer

Build a `.sif` image from the local Docker image, then run it. Apptainer
executes as your own user (so outputs are owned correctly) and mounts the image
read-only:

```bash
apptainer build abcd_ksads.sif docker-daemon://abcd_ksads:latest

apptainer run --bind /path/to/ABCD/basedir:/data --env ABCD_70=/data \
    abcd_ksads.sif all
```

Or via the Makefile wrappers (with `ABCD_70` set in `.env`):

```bash
make apptainer-build
make apptainer-run              # runs `all`
make apptainer-run TARGET=tables
```

## Pipeline

```
harmonize/00_ingest.py                    Combine raw data files into a single parquet file
harmonize/01_resolve_missingness.py       resolve the four missingness states
harmonize/02_administration_calendar.py   module x wave x informant administration
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

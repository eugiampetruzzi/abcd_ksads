-include .env
export

.PHONY: all merge_covariates clean clean-cache ingest figures tables validate \
	coverage docker-build docker-run apptainer-build apptainer-run

# Run the test suite with a line-coverage report for the abcd_ksads package.
coverage:
	uv run pytest --cov=abcd_ksads --cov-report=term-missing

# --- Containerized runs (Docker / Apptainer) ---------------------------------
# IMAGE is the local image tag; TARGET is the pipeline target to run inside the
# container (e.g. `make docker-run TARGET=figures`). The host ${ABCD_70} data
# directory is bind-mounted to /data, and ABCD_70=/data inside the container.
IMAGE ?= abcd_ksads:latest
SIF ?= abcd_ksads.sif
DEF ?= apptainer.def
TARGET ?= all

docker-build:
	docker build -t ${IMAGE} .

docker-run:
	docker run --rm --user $$(id -u):$$(id -g) \
		-v "${ABCD_70}":/data -e ABCD_70=/data ${IMAGE} ${TARGET}

# Native build straight from the definition file (no Docker needed).
# --no-mount bind-paths skips site-configured binds (e.g. /software on HPC),
# which the minimal base image lacks and cannot auto-create during the build.
apptainer-build:
	apptainer build --fakeroot --no-mount bind-paths ${SIF} ${DEF}

apptainer-run:
	apptainer run --bind "${ABCD_70}":/data --env ABCD_70=/data ${SIF} ${TARGET}

merge_covariates:
	uv run python harmonize/merge_covariates.py

clean-cache:
	-rm ${ABCD_70}/derivatives/raw_cache/*

clean:
	-rm ${ABCD_70}/derivatives/ksads*

clean-figures:
	-rm ${ABCD_70}/figures/*

clean-tables:
	-rm ${ABCD_70}/tables/*
	

ingest: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet

${ABCD_70}/derivatives/raw_cache/phenotype.parquet:
	uv run python harmonize/00_ingest.py

${ABCD_70}/derivatives/ksads_resolution_summary.csv: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet
	uv run python harmonize/01_resolve_missingness.py

${ABCD_70}/derivatives/ksads_administration_grid.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/02_administration_calendar.py

${ABCD_70}/derivatives/ksads_caseness_sensitivity.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/03_category_crosswalk.py

${ABCD_70}/derivatives/ksads_version_audit.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/04_version_provenance.py

${ABCD_70}/derivatives/multiverse_grid.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv ${ABCD_70}/derivatives/ksads_administration_grid.csv
	uv run python harmonize/06_multiverse_spec.py

${ABCD_70}/derivatives/multiverse_summary.csv: ${ABCD_70}/derivatives/multiverse_grid.csv
	uv run python harmonize/07_multiverse_summary.py

${ABCD_70}/derivatives/single_lever.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/08_single_lever.py

${ABCD_70}/derivatives/inferential_summary.csv: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/09_inferential_multiverse.py

${ABCD_70}/derivatives/informant_concordance.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/10_informant_discrepancy.py

${ABCD_70}/derivatives/anxiety_decomposition.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/10b_aux_anxiety_decomposition.py

${ABCD_70}/derivatives/missingness_error.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/10c_aux_missingness_audit.py

${ABCD_70}/derivatives/paper_numbers.json: ${ABCD_70}/derivatives/ksads_resolution_summary.csv ${ABCD_70}/derivatives/ksads_caseness_sensitivity.csv ${ABCD_70}/derivatives/ksads_version_audit.csv ${ABCD_70}/derivatives/multiverse_summary.csv ${ABCD_70}/derivatives/single_lever.csv ${ABCD_70}/derivatives/anxiety_decomposition.csv ${ABCD_70}/derivatives/missingness_error.csv
	uv run python harmonize/11_paper_numbers.py

${ABCD_70}/abcd_ksads_harmonized/participants.tsv: ${ABCD_70}/derivatives/ksads_version_audit.csv ${ABCD_70}/derivatives/ksads_administration_grid.csv
	uv run python harmonize/12_build_bids_dataset.py

${ABCD_70}/derivatives/technical_validation_report.txt: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/13_technical_validation.py

${ABCD_70}/derivatives/module_overscreening.csv: ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/14_module_overscreening.py

${ABCD_70}/abcd_ksads_harmonized/csv/sessions.csv: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet ${ABCD_70}/abcd_ksads_harmonized/participants.tsv
	uv run python harmonize/15_export_analysis_csv.py

${ABCD_70}/figures/Figure1_2_combined.png: ${ABCD_70}/derivatives/multiverse_grid.csv
	uv run python figures/make_fig12.py

${ABCD_70}/figures/Figure_bwas_style.png: ${ABCD_70}/derivatives/inferential_summary.csv
	uv run python figures/make_fig_bwas_style.py

${ABCD_70}/figures/Figure_category_calendar.png: ${ABCD_70}/derivatives/ksads_administration_grid.csv
	uv run python figures/make_fig_catcalendar.py

${ABCD_70}/figures/Figure_inferential.png: ${ABCD_70}/derivatives/inferential_summary.csv
	uv run python figures/make_fig_inferential.py

${ABCD_70}/figures/Figure_informant_trajectories.png: ${ABCD_70}/derivatives/informant_concordance.csv
	uv run python figures/make_fig_informant.py

figures: ${ABCD_70}/figures/Figure1_2_combined.png ${ABCD_70}/figures/Figure_bwas_style.png \
${ABCD_70}/figures/Figure_category_calendar.png ${ABCD_70}/figures/Figure_inferential.png \
${ABCD_70}/figures/Figure_informant_trajectories.png

${ABCD_70}/tables/Table1_reporting_checklist.docx: tables/build_table1_checklist.py
	uv run python tables/build_table1_checklist.py

${ABCD_70}/tables/Table2_categories_subdiagnoses.docx: tables/build_table_categories.py
	uv run python tables/build_table_categories.py

tables: ${ABCD_70}/tables/Table1_reporting_checklist.docx ${ABCD_70}/tables/Table2_categories_subdiagnoses.docx

PIPELINE := ingest ${ABCD_70}/derivatives/ksads_resolution_summary.csv ${ABCD_70}/derivatives/ksads_administration_grid.csv \
${ABCD_70}/derivatives/ksads_caseness_sensitivity.csv ${ABCD_70}/derivatives/ksads_version_audit.csv \
${ABCD_70}/derivatives/multiverse_grid.csv ${ABCD_70}/derivatives/multiverse_summary.csv \
${ABCD_70}/derivatives/single_lever.csv ${ABCD_70}/derivatives/inferential_summary.csv \
${ABCD_70}/derivatives/informant_concordance.csv ${ABCD_70}/derivatives/anxiety_decomposition.csv \
${ABCD_70}/derivatives/missingness_error.csv ${ABCD_70}/derivatives/paper_numbers.json \
${ABCD_70}/abcd_ksads_harmonized/participants.tsv ${ABCD_70}/derivatives/technical_validation_report.txt \
${ABCD_70}/derivatives/module_overscreening.csv ${ABCD_70}/abcd_ksads_harmonized/csv/sessions.csv \
figures tables

# validate depends on the whole pipeline, so it runs last even under `make -j`.
validate: $(PIPELINE)
	uv run python harmonize/validate_against_orig.py

all: $(PIPELINE) validate

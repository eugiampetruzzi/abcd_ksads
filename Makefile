-include .env
export

.PHONY: all merge_covariates clean ingest

merge_covariates:
	uv run python harmonize/merge_covariates.py

clean-cache:
	-rm ${ABCD_70}/derivatives/raw_cache/*

clean:
	-rm ${ABCD_70}/subject_demographics.tsv
	-rm ${ABCD_70}/derivatives/ksads*

ingest: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet

${ABCD_70}/derivatives/raw_cache/phenotype.parquet:
	uv run python harmonize/00_ingest.py

${ABCD_70}/derivatives/ksads_resolution_summary.csv: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet
	uv run python harmonize/01_resolve_missingness.py

${ABCD_70}/derivatives/ksads_administration_grid.csv:
	uv run python harmonize/02_administration_calendar.py

${ABCD_70}/derivatives/ksads_caseness_sensitivity.csv:
	uv run python harmonize/03_category_crosswalk.py

${ABCD_70}/derivatives/ksads_version_audit.csv:
	uv run python harmonize/04_version_provenance.py

${ABCD_70}/derivatives/multiverse_grid.csv:
	uv run python harmonize/06_multiverse_spec.py

${ABCD_70}/derivatives/multiverse_summary.csv:
	uv run python harmonize/07_multiverse_summary.py

${ABCD_70}/derivatives/single_lever.csv:
	uv run python harmonize/08_single_lever.py

${ABCD_70}/derivatives/inferential_summary.csv: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet ${ABCD_70}/derivatives/ksads_resolution_summary.csv
	uv run python harmonize/09_inferential_multiverse.py

all: ingest ${ABCD_70}/derivatives/ksads_resolution_summary.csv ${ABCD_70}/derivatives/ksads_administration_grid.csv \
${ABCD_70}/derivatives/ksads_caseness_sensitivity.csv ${ABCD_70}/derivatives/ksads_version_audit.csv \
${ABCD_70}/derivatives/multiverse_grid.csv ${ABCD_70}/derivatives/multiverse_summary.csv \
${ABCD_70}/derivatives/single_lever.csv ${ABCD_70}/derivatives/inferential_summary.csv

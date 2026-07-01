-include .env
export

.PHONY: all merge_covariates clean ingest

merge_covariates:
	uv run python harmonize/merge_covariates.py

clean:
	-rm harmonize/derivatives/*
	-rm ${ABCD_70}/subject_demographics.tsv

ingest: ${ABCD_70}/derivatives/raw_cache/phenotype.parquet

${ABCD_70}/derivatives/raw_cache/phenotype.parquet:
	uv run python harmonize/00_ingest.py

harmonize/derivatives/ksads_resolution_summary.csv:
	uv run python harmonize/01_resolve_missingness.py

harmonize/derivatives/ksads_administration_grid.csv:
	uv run python harmonize/02_administration_calendar.py

harmonize/derivatives/ksads_caseness_sensitivity.csv:
	uv run python harmonize/03_category_crosswalk.py

harmonize/derivatives/ksads_version_audit.csv:
	uv run python harmonize/04_version_provenance.py

harmonize/derivatives/multiverse_grid.csv:
	uv run python harmonize/06_multiverse_spec.py

harmonize/derivatives/multiverse_summary.csv:
	uv run python harmonize/07_multiverse_summary.py

harmonize/derivatives/single_lever.csv:
	uv run python harmonize/08_single_lever.py

harmonize/derivatives/inferential_summary.csv:
	uv run python harmonize/09_inferential_multiverse.py

all: harmonize/derivatives/ksads_resolution_summary.csv harmonize/derivatives/ksads_administration_grid.csv \
harmonize/derivatives/ksads_caseness_sensitivity.csv harmonize/derivatives/ksads_version_audit.csv \
harmonize/derivatives/multiverse_grid.csv harmonize/derivatives/multiverse_summary.csv \
harmonize/derivatives/single_lever.csv harmonize/derivatives/inferential_summary.csv

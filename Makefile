-include .env
export

merge_covariates:
	uv run python harmonize/merge_covariates.py

clean:
	-rm harmonize/derivatives/*
	-rm ${ABCD_70}/subject_demographics.tsv

derivatives/ksads_resolution_summary.csv:
	uv run python harmonize/01_resolve_missingness.py

derivatives/ksads_administration_grid.csv:
	uv run python harmonize/02_administration_calendar.py

derivatives/ksads_caseness_sensitivity.csv:
	uv run python harmonize/03_category_crosswalk.py

derivatives/ksads_version_audit.csv:
	uv run python harmonize/04_version_provenance.py

resolve: derivatives/ksads_resolution_summary.csv

calendar: derivatives/ksads_administration_grid.csv

crosswalk: derivatives/ksads_caseness_sensitivity.csv

provenace: derivatives/ksads_version_audit.csv
# Changelog: substantive edits and analytic decisions

Chronological record of the substantive changes made with Claude Code. Cosmetic edits
(formatting, comments, path parameterization) are omitted.

## Harmonization / dataset
- Built the four-state missingness resolver (positive / administered-negative /
  not-administered / no-record); non-administration (555) is never recoded to a
  negative.
- Implemented the administration calendar, DSM-category crosswalk (13 categories /
  26 sub-disorders), and KSADS-COMP 1.0/2.0 version provenance.
- Updated the ever-met definition to include full remission.
- Added quantification of the 555 miscoding rate and the anxiety phobia decomposition.

## Prevalence multiverse
- Enumerated prevalence across operationalizations (timeframe x informant x grouping,
  with threshold as a separate lever).
- Restricted the headline figure and text to the three primary decisions at full
  criteria (1.9-49.9%); reported the threshold-inclusive full range separately (56.8%).

## Informant discrepancy
- Computed caregiver vs. youth caseness per category and Cohen's kappa across waves;
  restricted comparisons to modules administered to both informants.

## Inferential (correlate) analysis
- Specified predictor x construct x operationalization logistic models across five
  domains (sex, income, race/ethnicity, screen time + family conflict, six
  triple-network RSFC measures).
- Added ABCD-standard RSFC quality control to the connectivity models.
- Added study-site (and scanner + mean FD for imaging) fixed effects and family
  cluster-robust standard errors.
- Summarized stability per pair (median OR, range, share significant, sign reversal,
  significant-in-every-spec) and variance decomposition across the three axes.

## Figures
- Adopted a colorblind-safe palette throughout (Okabe-Ito).
- Figure 1: two-panel prevalence (full grid + literature operationalizations),
  three-decision framing, full criteria.
- Figure 2: informant case-split, both axes shown as positive shares.
- Figure 3: association-strength ranking; both panels share one ordering (by
  consolidated median) so each cloud sits above its range bar.

## Reproducibility / packaging
- Centralized all paths in `config.py`; removed hard-coded locations.
- Collated every manuscript number into `paper_numbers.json` from pipeline outputs.
- Removed exploratory/superseded scripts; documented run order in the README.

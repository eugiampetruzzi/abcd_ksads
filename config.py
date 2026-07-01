"""Central paths for the pipeline.

Set the four locations below to your local ABCD data, either by editing the defaults
or by exporting the matching environment variables (ABCD_70, ABCD_51,
ABCD_RAW_PHENOTYPE, ABCD_DEMOGRAPHICS). All ABCD source data are access-controlled
(an ABCD Data Use Certification is required) and are not distributed here.
"""

import os

# Release 7.0 root: tabulated KSADS-COMP source, the BIDS rawdata/phenotype tables,
# the codebooks, and the covariate files (4_ELA_final.xlsx, 5_covariates_extended.xlsx).
ABCD_70 = os.environ.get("ABCD_70", "/path/to/abcd/release-7.0")

# Release 5.1 'core' directory: imaging, culture-environment, and novel-technologies
# tables used as baseline predictors in the inferential analysis.
ABCD_51_CORE = os.environ.get("ABCD_51", "/path/to/abcd/release-5.1/core")

# BIDS-converted raw KSADS phenotype tables (input to the dataset-export step).
RAW_PHENOTYPE = os.environ.get(
    "ABCD_RAW_PHENOTYPE", os.path.join(ABCD_70, "rawdata", "phenotype")
)

# Per-participant demographics (sex), merged into the released analysis CSVs.
DEMOGRAPHICS = os.environ.get(
    "ABCD_DEMOGRAPHICS", os.path.join(ABCD_70, "subject_demographics.tsv")
)

# repo-internal (do not edit)
REPO = os.path.dirname(os.path.abspath(__file__))
HARMONIZE = os.path.join(REPO, "harmonize")
DERIV = os.path.join(HARMONIZE, "derivatives")
FIGURES = os.path.join(REPO, "figures")
TABLES = os.path.join(REPO, "tables")
CODEBOOKS = os.path.join(REPO, "codebooks")
DATASET = os.path.join(REPO, "abcd_ksads_harmonized")

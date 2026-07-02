"""Central paths for the pipeline, as a pydantic settings model.

The four data locations below come from your local ABCD install, set either by the
defaults here or by exporting the matching environment variables (``ABCD_70``,
``ABCD_51``, ``ABCD_RAW_PHENOTYPE``, ``ABCD_DEMOGRAPHICS``). All ABCD source data are
access-controlled (an ABCD Data Use Certification is required) and are not distributed
here.

The repo-internal paths are derived from this file's location and should not need to be
set.
"""

from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import dotenv

dotenv.load_dotenv()

# This file lives at <repo>/src/abcd_ksads/config.py, so the repo root is three
# directories up.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    """Resolved filesystem paths for the harmonization and analysis pipeline."""

    model_config = SettingsConfigDict(extra="ignore")

    # Release 7.0 root: tabulated KSADS-COMP source, the BIDS rawdata/phenotype tables,
    # the codebooks, and the covariate files.
    ABCD_70: Path = Field(
        default=Path("/path/to/abcd/release-7.0"),
        validation_alias="ABCD_70",
    )

    # Release 5.1 'core' directory: imaging, culture-environment, and
    # novel-technologies tables used as baseline predictors.
    ABCD_51_CORE: Path = Field(
        default=Path("/path/to/abcd/release-5.1/core"),
        validation_alias="ABCD_51",
    )

    # BIDS-converted raw KSADS phenotype tables (input to the dataset-export step).
    # Defaults to ``ABCD_70/rawdata/phenotype`` when unset.
    RAW_PHENOTYPE: Path | None = Field(
        default=None,
        validation_alias="ABCD_RAW_PHENOTYPE",
    )

    # Per-participant demographics (sex), merged into the released analysis CSVs.
    # Defaults to ``ABCD_70/subject_demographics.tsv`` when unset.
    DEMOGRAPHICS: Path | None = Field(
        default=None,
        validation_alias="ABCD_DEMOGRAPHICS",
    )

    @model_validator(mode="after")
    def _fill_derived_defaults(self) -> "Config":
        """Derive phenotype/demographics paths from ``ABCD_70`` when not given."""
        if self.RAW_PHENOTYPE is None:
            self.RAW_PHENOTYPE = self.ABCD_70 / "rawdata" / "phenotype"
        if self.DEMOGRAPHICS is None:
            self.DEMOGRAPHICS = self.ABCD_70 / "subject_demographics.tsv"
        return self

    @computed_field
    @property
    def REPO(self) -> Path:
        return _REPO_ROOT

    @computed_field
    @property
    def HARMONIZE(self) -> Path:
        return self.REPO / "harmonize"

    @computed_field
    @property
    def DERIV(self) -> Path:
        return self.ABCD_70 / "derivatives"

    @computed_field
    @property
    def FIGURES(self) -> Path:
        return self.REPO / "figures"

    @computed_field
    @property
    def FIGURES_OUT(self) -> Path:
        """Rendered figure outputs, written alongside the derivatives directory."""
        return self.DERIV.parent / "figures"

    @computed_field
    @property
    def TABLES(self) -> Path:
        return self.REPO / "tables"

    @computed_field
    @property
    def CODEBOOKS(self) -> Path:
        return self.REPO / "codebooks"

    @computed_field
    @property
    def DATASET(self) -> Path:
        return self.ABCD_70 / "abcd_ksads_harmonized"

    @computed_field
    @property
    def RAW_CACHE(self) -> Path:
        return self.DERIV / "raw_cache"

    @computed_field
    @property
    def KSADS_VARIABLE_MAP(self) -> Path:
        return self.CODEBOOKS / "ksads_variable_map.csv"

    @computed_field
    @property
    def PHENOTYPE_MANIFEST(self) -> Path:
        return self.CODEBOOKS / "phenotype_manifest.txt"


# Module-level singleton plus attribute re-exports so callers can use either
# ``from abcd_ksads.config import config`` or ``from abcd_ksads import config``.
config = Config()

ABCD_70 = config.ABCD_70
ABCD_51_CORE = config.ABCD_51_CORE
RAW_PHENOTYPE = config.RAW_PHENOTYPE
DEMOGRAPHICS = config.DEMOGRAPHICS
REPO = config.REPO
HARMONIZE = config.HARMONIZE
DERIV = config.DERIV
FIGURES = config.FIGURES
FIGURES_OUT = config.FIGURES_OUT
TABLES = config.TABLES
CODEBOOKS = config.CODEBOOKS
DATASET = config.DATASET
RAW_CACHE = config.RAW_CACHE
KSADS_VARIABLE_MAP = config.KSADS_VARIABLE_MAP
PHENOTYPE_MANIFEST = config.PHENOTYPE_MANIFEST

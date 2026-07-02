"""Build the inferential-analysis predictors from the cached phenotype table.

All predictors are read from the consolidated release-7.0 cache (phenotype.parquet)
at the baseline session. Raw values are the exact source strings, so each helper
coerces to numeric and recodes as needed before z-scoring.
"""

import numpy as np
import pandas as pd

from abcd_ksads import config
from abcd_ksads.multiverse import BASE_SES

# ---- cache column names (SOURCES) -------------------------------------------
SEX = "ab_g_stc__cohort_sex"
RACE = "ab_g_stc__cohort_ethnrace__leg"
FAMILY = "ab_g_stc__design_id__fam"
AGE = "ab_g_dyn__visit_age"
SITE = "ab_g_dyn__design_site"
INCOME = "ab_p_demo__income__hhold_001"
FAM_CONFLICT = "fc_y_fes__confl_mean"
RSFMRI_INCL = "mr_y_qc__incl__rsfmri_indicator"
SCANNER = "mr_y_adm__info__dev_manufact"
MOTION = "mr_y_qc__mot__rsfmri__mot_mean"

SCREEN_ITEMS = [f"nt_y_stq__screen__wkdy_{i:03d}" for i in range(1, 7)] + [
    f"nt_y_stq__screen__wknd_{i:03d}" for i in range(1, 7)
]

# functional-connectivity measure -> Gordon-network correlation column
FC_SOURCE = {
    "fc_dmn_within": "mr_y_rsfmri__corr__gpnet__def__def_mean",
    "fc_fpn_within": "mr_y_rsfmri__corr__gpnet__frp__frp_mean",
    "fc_sal_within": "mr_y_rsfmri__corr__gpnet__sal__sal_mean",
    "fc_dmn_fpn": "mr_y_rsfmri__corr__gpnet__def__frp_mean",
    "fc_dmn_salience": "mr_y_rsfmri__corr__gpnet__def__sal_mean",
    "fc_sal_fpn": "mr_y_rsfmri__corr__gpnet__sal__frp_mean",
}
FC_COLS = list(FC_SOURCE)

# ---- recoding tables --------------------------------------------------------
SEX_MAP = {1: 0.0, 2: 1.0}  # sex_f: Female (2) -> 1, Male (1) -> 0
RACE_MAP = {1: "Hispanic", 2: "White", 3: "Black/AA", 4: "Asian", 13: "Other/Multiracial"}
SCREEN_HOURS = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0, 6: 4.0}
INCOME_MISSING = [777, 999]  # "Decline to answer" / "Don't know"

RACE_REF = "White"
RACE_LVES = ["Black/AA", "Hispanic", "Asian", "Other/Multiracial"]


def _num(series: pd.Series) -> pd.Series:
    """Coerce a (possibly categorical/string) series to float, non-numeric -> NaN."""
    return pd.to_numeric(pd.Series(series).astype("object"), errors="coerce")


def map_sex(series: pd.Series) -> pd.Series:
    return _num(series).map(SEX_MAP)


def map_race(series: pd.Series) -> pd.Series:
    return _num(series).map(RACE_MAP)


def recode_income(series: pd.Series) -> pd.Series:
    values = _num(series)
    return values.mask(values.isin(INCOME_MISSING))


def screen_hours(wide: pd.DataFrame) -> pd.Series:
    """Sum the 12 screen-time items after mapping each ordinal code to hours."""
    hours = [_num(wide[item]).map(SCREEN_HOURS) for item in SCREEN_ITEMS]
    return pd.concat(hours, axis=1).sum(axis=1, min_count=1)


def compute_fc_qc(series: pd.Series) -> pd.Series:
    """rsfMRI QC pass = ABCD recommended-for-inclusion indicator == 1."""
    return _num(series) == 1


def zscore(series: pd.Series) -> pd.Series:
    values = _num(series)
    return (values - values.mean()) / values.std()


def derive_predictors(wide: pd.DataFrame) -> pd.DataFrame:
    """Turn a baseline wide slice (indexed by participant_id) into analysis predictors."""
    d = pd.DataFrame(index=wide.index)
    d["sex_f"] = map_sex(wide[SEX])
    d["Race"] = map_race(wide[RACE])
    d["family_id"] = wide[FAMILY].astype("object")
    d["site"] = wide[SITE].astype("object")
    d["scanner"] = wide[SCANNER].astype("object")
    d["age_z"] = zscore(wide[AGE])
    d["income_z"] = zscore(recode_income(wide[INCOME]))
    d["screentime"] = screen_hours(wide)
    d["screentime_z"] = zscore(d["screentime"])
    d["fam_conflict_z"] = zscore(wide[FAM_CONFLICT])
    d["mean_fd"] = _num(wide[MOTION])
    d["fc_qc_pass"] = compute_fc_qc(wide[RSFMRI_INCL])

    for fc, src in FC_SOURCE.items():
        d[fc] = _num(wide[src])
    # apply ABCD rsfMRI QC: blank FC for participants failing inclusion
    d.loc[~d["fc_qc_pass"], FC_COLS] = np.nan

    # standardize motion over QC-passing participants (the ones with usable FC)
    fd_pass = d.loc[d["fc_qc_pass"], "mean_fd"]
    d["mean_fd_z"] = (d["mean_fd"] - fd_pass.mean()) / fd_pass.std()
    for fc in FC_COLS:
        d[fc + "_z"] = zscore(d[fc])
    return d


def load_predictors(cache_path=None, base_ses=BASE_SES) -> pd.DataFrame:
    """Read the required columns from the cache at baseline and derive predictors."""
    cache_path = cache_path or (config.RAW_CACHE / "phenotype.parquet")
    columns = (
        ["participant_id", "session_id", SEX, RACE, FAMILY, AGE, SITE, INCOME,
         FAM_CONFLICT, RSFMRI_INCL, SCANNER, MOTION]
        + SCREEN_ITEMS
        + list(FC_SOURCE.values())
    )
    wide = pd.read_parquet(cache_path, columns=columns)
    base = (
        wide[wide["session_id"] == base_ses]
        .drop_duplicates("participant_id")
        .set_index("participant_id")
    )
    return derive_predictors(base)

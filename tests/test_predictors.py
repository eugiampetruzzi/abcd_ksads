"""Tests for building inferential predictors from the cached phenotype table."""

import numpy as np
import pandas as pd
import pytest

from abcd_ksads import predictors as pr


def test_map_sex_female_is_one():
    out = pr.map_sex(pd.Series(["1.0", "2.0", "", np.nan]))
    assert out.iloc[0] == 0.0  # Male
    assert out.iloc[1] == 1.0  # Female
    assert out.iloc[2:].isna().all()


def test_map_race_codes_to_labels():
    out = pr.map_race(pd.Series(["1.0", "2.0", "3.0", "4.0", "13.0", "99.0"]))
    assert out.tolist()[:5] == ["Hispanic", "White", "Black/AA", "Asian", "Other/Multiracial"]
    assert pd.isna(out.iloc[5])


def test_recode_income_sentinels_to_nan():
    out = pr.recode_income(pd.Series(["7.0", "777.0", "999.0", "1.0"]))
    assert out.iloc[0] == 7.0
    assert out.iloc[3] == 1.0
    assert out.iloc[1:3].isna().all()


def test_screen_hours_maps_codes_and_sums():
    # all 12 items at code 3 (=1 hour) -> 12 hours; all code 0 -> 0 hours
    data = {item: ["3.0", "0.0"] for item in pr.SCREEN_ITEMS}
    out = pr.screen_hours(pd.DataFrame(data))
    assert out.tolist() == [12.0, 0.0]


def test_screen_hours_all_missing_is_nan():
    data = {item: [np.nan] for item in pr.SCREEN_ITEMS}
    out = pr.screen_hours(pd.DataFrame(data))
    assert pd.isna(out.iloc[0])


def test_compute_fc_qc_indicator_only():
    out = pr.compute_fc_qc(pd.Series(["1", "0", np.nan]))
    assert out.tolist() == [True, False, False]


def test_zscore_centers_and_scales():
    out = pr.zscore(pd.Series([1.0, 2.0, 3.0]))
    assert out.tolist() == [-1.0, 0.0, 1.0]


@pytest.fixture
def sample_wide():
    """Baseline wide slice with all SOURCES columns for four participants."""
    rows = {
        pr.SEX: ["2.0", "1.0", "2.0", "1.0"],
        pr.RACE: ["2.0", "3.0", "1.0", "4.0"],
        pr.FAMILY: ["F1", "F2", "F3", "F4"],
        pr.AGE: ["10.3", "11.1", "9.8", "10.0"],
        pr.SITE: ["1.0", "2.0", "1.0", "3.0"],
        pr.INCOME: ["7.0", "777.0", "3.0", "999.0"],
        pr.FAM_CONFLICT: ["1.5", "2.0", "0.5", "1.0"],
        pr.RSFMRI_INCL: ["1", "1", "1", "0"],
        pr.SCANNER: ["3.0", "1.0", "3.0", "2.0"],
        pr.MOTION: ["0.2", "0.3", "0.15", "0.9"],
    }
    for item in pr.SCREEN_ITEMS:
        rows[item] = ["3.0", "0.0", "6.0", "2.0"]
    for i, src in enumerate(pr.FC_SOURCE.values()):
        rows[src] = [f"0.{i}1", f"0.{i}2", f"0.{i}3", f"0.{i}4"]
    return pd.DataFrame(rows, index=pd.Index(["p1", "p2", "p3", "p4"], name="participant_id"))


def test_derive_predictors_has_expected_columns(sample_wide):
    P = pr.derive_predictors(sample_wide)
    expected = {
        "sex_f", "Race", "family_id", "age_z", "income_z", "site", "scanner",
        "screentime_z", "fam_conflict_z", "mean_fd_z", "fc_qc_pass",
    } | {fc + "_z" for fc in pr.FC_COLS}
    assert expected <= set(P.columns)


def test_derive_predictors_sex_and_race(sample_wide):
    P = pr.derive_predictors(sample_wide)
    assert P["sex_f"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert P["Race"].tolist() == ["White", "Black/AA", "Hispanic", "Asian"]


def test_derive_predictors_income_sentinels_missing(sample_wide):
    P = pr.derive_predictors(sample_wide)
    # p2 (777) and p4 (999) have missing income -> missing income_z
    assert P.loc["p2", "income_z"] != P.loc["p2", "income_z"]  # NaN
    assert P.loc["p4", "income_z"] != P.loc["p4", "income_z"]


def test_derive_predictors_screentime_hours(sample_wide):
    P = pr.derive_predictors(sample_wide)
    # p1: 12 items x 1h = 12; p3: 12 x 4h = 48; p4: 12 x 0.5h = 6
    assert P["screentime"].tolist() == [12.0, 0.0, 48.0, 6.0]


def test_derive_predictors_blanks_fc_when_qc_fails(sample_wide):
    P = pr.derive_predictors(sample_wide)
    assert P["fc_qc_pass"].tolist() == [True, True, True, False]
    # p4 fails QC -> its FC z-scores are blanked
    assert P.loc["p4", [fc + "_z" for fc in pr.FC_COLS]].isna().all()
    # a passing participant keeps FC
    assert P.loc["p1", "fc_dmn_within_z"] == P.loc["p1", "fc_dmn_within_z"]

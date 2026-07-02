"""Tests for the analysis-CSV interview-age derivation."""

import pandas as pd

from abcd_ksads import export as ex


def _wide(rows):
    return pd.DataFrame(
        rows, columns=["participant_id", "session_id", "mh_p_ksads__dep_age", "mh_y_ksads__dep_age"]
    )


def test_interview_ages_youngest_wins_and_converts_to_months():
    # parent reports 10.5 yr, youth reports 9.0 yr -> youngest (9.0) wins
    wide = _wide([("p1", "ses-00A", "10.5", "9.0")])
    out = ex.interview_ages(wide).set_index("participant_id")
    assert out.loc["p1", "interview_age"] == 108     # 9.0 * 12
    assert out.loc["p1", "ksads_version"] == "1.0"    # ses-00A is a 1.0 wave


def test_interview_ages_version_tag_and_missing_ages_dropped():
    wide = _wide([
        ("p1", "ses-04A", "12.0", "x"),      # youth non-numeric -> parent 12.0 used
        ("p2", "ses-00A", "y", "z"),         # both non-numeric -> dropped
    ])
    out = ex.interview_ages(wide).set_index("participant_id")
    assert "p2" not in out.index
    assert out.loc["p1", "interview_age"] == 144      # 12.0 * 12
    assert out.loc["p1", "ksads_version"] == "2.0"    # ses-04A is a 2.0 wave

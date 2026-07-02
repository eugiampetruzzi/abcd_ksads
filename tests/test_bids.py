"""Tests for the BIDS caseness builders (either-informant combination, wave counts)."""

import pandas as pd

from abcd_ksads import bids


def _caseness(rows):
    return pd.DataFrame(
        rows, columns=["participant_id", "session_id", "category", "status"]
    )


def test_combine_either_takes_max_rank_across_informants():
    cp = _caseness([
        ("p1", "ses-00A", "Depression", "positive"),
        ("p2", "ses-00A", "Depression", "administered_negative"),
    ])
    cy = _caseness([
        ("p1", "ses-00A", "Depression", "administered_negative"),
        ("p2", "ses-00A", "Depression", "not_administered"),
    ])
    out = bids.combine_either(cp, cy).set_index("participant_id")["status"]
    assert out["p1"] == "positive"                 # max(positive, administered_negative)
    assert out["p2"] == "administered_negative"     # max(administered_negative, not_administered)


def test_count_admin_waves_counts_administered_sessions_per_informant():
    res = pd.DataFrame(
        [
            ("p1", "ses-00A", "parent", "positive"),
            ("p1", "ses-02A", "parent", "administered_negative"),
            ("p1", "ses-00A", "youth", "not_administered"),   # not administered -> not counted
            ("p2", "ses-00A", "parent", "positive"),
        ],
        columns=["participant_id", "session_id", "informant", "resolved"],
    )
    out = bids.count_admin_waves(res).set_index("participant_id")
    assert out.loc["p1", "n_waves_parent_kSADS"] == 2
    assert out.loc["p1", "n_waves_youth_kSADS"] == 0
    assert out.loc["p2", "n_waves_parent_kSADS"] == 1


def test_caseness_wide_stacks_informants_and_either_is_max():
    cw = pd.DataFrame(
        [
            ("v_dep_p", "parent", "dep", "present", "Depression", 0),
            ("v_dep_y", "youth", "dep", "present", "Depression", 0),
        ],
        columns=["variable", "informant", "module", "status_layer",
                 "category", "is_subthreshold"],
    )
    base = pd.DataFrame(
        [
            ("p1", "ses-00A", "v_dep_p", "positive"),
            ("p1", "ses-00A", "v_dep_y", "administered_negative"),
        ],
        columns=["participant_id", "session_id", "variable", "resolved"],
    )
    wide = bids.caseness_wide(base, cw, "current", disorder_cats=["Depression"])
    assert set(wide.informant) == {"parent", "youth", "either"}
    assert "Depression" in wide.columns
    either = wide[wide.informant == "either"].iloc[0]
    assert either.Depression == "positive"          # parent-positive wins

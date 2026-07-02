"""Tests for KSADS-COMP version tagging and the pre-switch audit."""

import pandas as pd

from abcd_ksads import version as ver


def _resolved(rows):
    return pd.DataFrame(
        rows, columns=["participant_id", "session_id", "variable", "resolved"]
    )


# ---- tag_versions -----------------------------------------------------------


def test_tag_versions_labels_waves_and_flags_invalid():
    r = _resolved([
        ("p1", "ses-00A", "v_2only", "positive"),          # 2.0-only under 1.0 -> invalid
        ("p1", "ses-04A", "v_2only", "positive"),          # 2.0-only under 2.0 -> valid
        ("p1", "ses-00A", "v_normal", "administered_negative"),  # not 2.0-only -> valid
    ])
    out = ver.tag_versions(r, two_zero_only_vars={"v_2only"})
    assert out.ksads_version.tolist() == ["1.0", "2.0", "1.0"]
    assert out.two_zero_only.tolist() == [True, True, False]
    assert out.version_valid.tolist() == [False, True, True]


def test_tag_versions_does_not_mutate_input():
    r = _resolved([("p1", "ses-00A", "v", "positive")])
    ver.tag_versions(r, two_zero_only_vars=set())
    assert "ksads_version" not in r.columns


# ---- audit_pre_switch -------------------------------------------------------


def test_audit_pre_switch_flags_administered_2only_cells_under_v1():
    r = ver.tag_versions(
        _resolved([
            ("p1", "ses-00A", "v_2only", "positive"),           # administered pre-switch
            ("p2", "ses-00A", "v_2only", "administered_negative"),
            ("p3", "ses-02A", "v_2only", "not_administered"),   # not administered -> excluded
            ("p4", "ses-04A", "v_2only", "positive"),           # post-switch -> excluded
        ]),
        two_zero_only_vars={"v_2only"},
    )
    audit = ver.audit_pre_switch(r)
    # only the ses-00A row has administered (positive+neg) cells > 0
    assert audit.session_id.tolist() == ["ses-00A"]
    assert audit.iloc[0].administered_pre_switch == 2


def test_audit_pre_switch_empty_when_no_2only_before_switch():
    r = ver.tag_versions(
        _resolved([("p1", "ses-04A", "v_2only", "positive")]),
        two_zero_only_vars={"v_2only"},
    )
    assert len(ver.audit_pre_switch(r)) == 0

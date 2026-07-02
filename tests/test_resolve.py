"""Tests for resolving KSADS diagnosis cells from the consolidated wide cache."""

import numpy as np
import pandas as pd
import pytest

from abcd_ksads import resolve

# Two synthetic diagnosis variables with metadata mirroring the KSADS map. v_mix has
# an empty ``disorder`` so its label is used as the fallback.
VAR_META = {
    "v_pos": {
        "variable": "v_pos", "informant": "parent", "module": "adhd",
        "status": "current", "disorder": "ADHD", "label": "ADHD label",
    },
    "v_mix": {
        "variable": "v_mix", "informant": "youth", "module": "dep",
        "status": "ever_met", "disorder": "", "label": "Dep label",
    },
}


@pytest.fixture
def wide():
    """A wide cache slice: coded floats-as-strings, blanks, absent (NaN), and an
    out-of-scope session — value columns dictionary-encoded like the real cache."""
    df = pd.DataFrame(
        {
            "participant_id": ["s1", "s2", "s3", "s4", "s5"],
            "session_id": ["ses-00A", "ses-00A", "ses-00A", "ses-00A", "ses-99Z"],
            "v_pos": ["1.0", "0.0", "", np.nan, "1.0"],
            "v_mix": ["0.0", "555.0", "1.0", "0.0", "1.0"],
        }
    )
    for c in ("v_pos", "v_mix"):
        df[c] = df[c].astype("category")
    return df


def _resolved(long, participant, variable):
    row = long[(long.participant_id == participant) & (long.variable == variable)]
    return None if row.empty else row.iloc[0]["resolved"]


def test_coded_values_map_to_resolved_states(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    assert _resolved(long, "s1", "v_pos") == "positive"
    assert _resolved(long, "s2", "v_pos") == "administered_negative"
    assert _resolved(long, "s2", "v_mix") == "not_administered"


def test_blank_maps_to_no_record(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    assert _resolved(long, "s3", "v_pos") == "no_record"


def test_absent_cell_is_dropped(wide):
    # s4/v_pos is NaN in the cache (pair not in that variable's source file) -> no row
    long = resolve.resolve_wide(wide, VAR_META)
    assert _resolved(long, "s4", "v_pos") is None
    # but s4/v_mix (present, coded 0.0) is kept
    assert _resolved(long, "s4", "v_mix") == "administered_negative"


def test_out_of_scope_session_is_dropped(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    assert (long.session_id == "ses-99Z").sum() == 0


def test_row_count_matches_kept_cells(wide):
    # v_pos: s1,s2,s3 kept (s4 absent, s5 session) = 3; v_mix: s1..s4 kept = 4
    long = resolve.resolve_wide(wide, VAR_META)
    assert len(long) == 7


def test_disorder_falls_back_to_label(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    vmix = long[long.variable == "v_mix"]
    assert (vmix["disorder"] == "Dep label").all()


def test_resolved_is_ordered_categorical(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    assert isinstance(long["resolved"].dtype, pd.CategoricalDtype)
    assert list(long["resolved"].cat.categories) == resolve.RESOLVED


def test_build_summary_counts_per_variable_session(wide):
    long = resolve.resolve_wide(wide, VAR_META)
    summ = resolve.build_summary(long, VAR_META).set_index(["variable", "session_id"])
    pos = summ.loc[("v_pos", "ses-00A")]
    assert (pos["n_positive"], pos["n_administered_negative"], pos["n_no_record"]) == (1, 1, 1)
    mix = summ.loc[("v_mix", "ses-00A")]
    assert (mix["n_positive"], mix["n_administered_negative"], mix["n_not_administered"]) == (1, 2, 1)


def test_load_diagnosis_metadata_filters_diagnosis_layer(tmp_path):
    csv_path = tmp_path / "map.csv"
    csv_path.write_text(
        "variable,layer,informant,module,status,disorder,label\n"
        "v_pos,diagnosis,parent,adhd,current,ADHD,ADHD label\n"
        "v_meta,meta,parent,adhd,,,\n"
    )
    meta = resolve.load_diagnosis_metadata(csv_path)
    assert set(meta) == {"v_pos"}
    assert meta["v_pos"]["module"] == "adhd"

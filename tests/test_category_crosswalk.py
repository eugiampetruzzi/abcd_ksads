"""Tests for the DSM-category crosswalk and caseness engine."""

import pandas as pd
import pytest

from abcd_ksads import category_crosswalk as cc


def _cw(rows):
    """A crosswalk frame with the columns build_caseness consumes."""
    return pd.DataFrame(
        rows,
        columns=["variable", "informant", "module", "status_layer",
                 "category", "is_subthreshold"],
    )


def _resolved(rows):
    return pd.DataFrame(
        rows, columns=["participant_id", "session_id", "variable", "resolved"]
    )


def _status(df, participant, category="Depression"):
    """The single caseness status for one participant/category."""
    sub = df[(df.participant_id == participant) & (df.category == category)]
    assert len(sub) == 1
    return sub.status.iloc[0]


# ---- build_caseness: constituent aggregation --------------------------------


def test_caseness_positive_if_any_constituent_positive():
    # Anxiety has two constituents; one positive, one negative -> category positive.
    cw = _cw([
        ("v_panic", "parent", "panic", "present", "Anxiety", 0),
        ("v_gad", "parent", "gad", "present", "Anxiety", 0),
    ])
    resolved = _resolved([
        ("P1", "ses-00A", "v_panic", "positive"),
        ("P1", "ses-00A", "v_gad", "administered_negative"),
    ])
    out = cc.build_caseness(resolved, cw, status_set="current", informant="parent")
    assert _status(out, "P1", "Anxiety") == "positive"


def test_caseness_administered_negative_when_all_constituents_negative():
    cw = _cw([("v_dep", "parent", "dep", "present", "Depression", 0)])
    resolved = _resolved([("P1", "ses-00A", "v_dep", "administered_negative")])
    out = cc.build_caseness(resolved, cw, status_set="current", informant="parent")
    assert _status(out, "P1") == "administered_negative"


# ---- build_caseness: status_set lever ---------------------------------------


def test_caseness_status_set_current_excludes_past_layer():
    # A case positive only on the 'past' layer: negative under current, positive under ever_met.
    cw = _cw([
        ("v_dep_present", "parent", "dep", "present", "Depression", 0),
        ("v_dep_past", "parent", "dep", "past", "Depression", 0),
    ])
    resolved = _resolved([
        ("P1", "ses-00A", "v_dep_present", "administered_negative"),
        ("P1", "ses-00A", "v_dep_past", "positive"),
    ])
    cur = cc.build_caseness(resolved, cw, status_set="current", informant="parent")
    ever = cc.build_caseness(resolved, cw, status_set="ever_met", informant="parent")
    assert _status(cur, "P1") == "administered_negative"
    assert _status(ever, "P1") == "positive"


# ---- build_caseness: subthreshold lever -------------------------------------


def test_caseness_subthreshold_excluded_by_default_included_on_request():
    cw = _cw([
        ("v_dep_full", "parent", "dep", "present", "Depression", 0),
        ("v_dep_sub", "parent", "dep", "present", "Depression", 1),
    ])
    resolved = _resolved([
        ("P1", "ses-00A", "v_dep_full", "administered_negative"),
        ("P1", "ses-00A", "v_dep_sub", "positive"),
    ])
    default = cc.build_caseness(resolved, cw, status_set="current", informant="parent")
    withsub = cc.build_caseness(
        resolved, cw, status_set="current", informant="parent", include_subthreshold=True
    )
    assert _status(default, "P1") == "administered_negative"
    assert _status(withsub, "P1") == "positive"


# ---- build_caseness: informant filter ---------------------------------------


def test_caseness_informant_filter_selects_matching_rows():
    cw = _cw([
        ("v_dep_p", "parent", "dep", "present", "Depression", 0),
        ("v_dep_y", "youth", "dep", "present", "Depression", 0),
    ])
    resolved = _resolved([
        ("P1", "ses-00A", "v_dep_p", "positive"),
        ("P1", "ses-00A", "v_dep_y", "administered_negative"),
    ])
    par = cc.build_caseness(resolved, cw, status_set="current", informant="parent")
    yth = cc.build_caseness(resolved, cw, status_set="current", informant="youth")
    assert _status(par, "P1") == "positive"
    assert _status(yth, "P1") == "administered_negative"


# ---- build_caseness: 'both' informant branch --------------------------------


def test_caseness_both_requires_positive_on_both_informants():
    cw = _cw([
        ("v_dep_p", "parent", "dep", "present", "Depression", 0),
        ("v_dep_y", "youth", "dep", "present", "Depression", 0),
    ])
    resolved = _resolved([
        # P1: positive on both -> positive
        ("P1", "ses-00A", "v_dep_p", "positive"),
        ("P1", "ses-00A", "v_dep_y", "positive"),
        # P2: positive parent, negative youth -> administered_negative (not both)
        ("P2", "ses-00A", "v_dep_p", "positive"),
        ("P2", "ses-00A", "v_dep_y", "administered_negative"),
        # P3: not administered on either -> not_administered
        ("P3", "ses-00A", "v_dep_p", "not_administered"),
        ("P3", "ses-00A", "v_dep_y", "not_administered"),
    ])
    out = cc.build_caseness(resolved, cw, status_set="current", informant="both")
    assert _status(out, "P1") == "positive"
    assert _status(out, "P2") == "administered_negative"
    assert _status(out, "P3") == "not_administered"


# ---- build_crosswalk --------------------------------------------------------


def _write_map(tmp_path, rows):
    path = tmp_path / "map.csv"
    pd.DataFrame(
        rows, columns=["variable", "informant", "module", "layer", "status", "label"]
    ).to_csv(path, index=False)
    return path


def test_build_crosswalk_flags_subthreshold_and_maps_category(tmp_path, monkeypatch):
    path = _write_map(tmp_path, [
        ("v1", "parent", "dep", "diagnosis", "present", "Major Depressive Disorder"),
        ("v2", "parent", "dep", "diagnosis", "present", "Other Specified Depressive Disorder"),
        ("v3", "parent", "dep", "diagnosis", "present", "Unspecified Depressive Disorder"),
        ("v4", "parent", "dep", "symptom", "present", "a symptom, not a diagnosis"),
    ])
    monkeypatch.setattr(cc.config, "KSADS_VARIABLE_MAP", path)
    out = cc.build_crosswalk()
    assert list(out.columns) == cc.COLUMNS
    assert set(out.variable) == {"v1", "v2", "v3"}  # non-diagnosis layer dropped
    flags = dict(zip(out.variable, out.is_subthreshold))
    assert flags == {"v1": 0, "v2": 1, "v3": 1}
    assert (out.category == "Depression").all()


def test_build_crosswalk_raises_on_unmapped_module(tmp_path, monkeypatch):
    path = _write_map(tmp_path, [
        ("v1", "parent", "not_a_module", "diagnosis", "present", "Mystery Disorder"),
    ])
    monkeypatch.setattr(cc.config, "KSADS_VARIABLE_MAP", path)
    with pytest.raises(ValueError, match="not_a_module"):
        cc.build_crosswalk()

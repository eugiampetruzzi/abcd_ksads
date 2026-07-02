"""Tests for the multiverse prevalence primitives."""

import numpy as np
import pandas as pd
import pytest

from abcd_ksads import multiverse as mv


def _caseness(rows):
    """A caseness frame as build_caseness would emit (single session implied)."""
    return pd.DataFrame(rows, columns=["participant_id", "category", "status"])


# ---- prevalence -------------------------------------------------------------


def test_prevalence_over_administered_denominator():
    stat = pd.Series(["positive", "positive", "administered_negative", "not_administered"])
    pct, n_num, n_den = mv.prevalence(stat)
    assert n_den == 3  # not_administered excluded
    assert n_num == 2
    assert pct == pytest.approx(100 * 2 / 3)


def test_prevalence_none_and_empty_are_nan():
    for stat in (None, pd.Series([], dtype=object)):
        pct, n_num, n_den = mv.prevalence(stat)
        assert np.isnan(pct) and n_num == 0 and n_den == 0


def test_prevalence_zero_denominator_is_nan():
    pct, n_num, n_den = mv.prevalence(pd.Series(["not_administered", "not_administered"]))
    assert np.isnan(pct) and n_num == 0 and n_den == 0


# ---- _agg -------------------------------------------------------------------


def test_agg_takes_max_status_across_categories():
    cobj = _caseness([
        ("P1", "ADHD", "administered_negative"),
        ("P1", "ODD", "positive"),
        ("P2", "ADHD", "administered_negative"),
        ("P2", "ODD", "administered_negative"),
    ])
    out = mv._agg(cobj, ["ADHD", "ODD", "Conduct"])
    assert out["P1"] == "positive"          # any constituent positive -> positive
    assert out["P2"] == "administered_negative"


def test_agg_empty_when_no_matching_categories():
    cobj = _caseness([("P1", "Depression", "positive")])
    assert len(mv._agg(cobj, ["Nonexistent"])) == 0


# ---- construct_status: informant combination --------------------------------


def _cache(parent, youth, key=("ever_met", False, "phobia_in")):
    return {key: {"parent": parent, "youth": youth}}


def test_construct_status_parent_and_youth_passthrough():
    parent = _caseness([("P1", "Depression", "positive"),
                        ("P2", "Depression", "positive")])
    youth = _caseness([("P1", "Depression", "positive"),
                       ("P2", "Depression", "administered_negative")])
    cache = _cache(parent, youth)
    ps = mv.construct_status(cache, "depression", "ever_met", "parent", False, "phobia_in")
    ys = mv.construct_status(cache, "depression", "ever_met", "youth", False, "phobia_in")
    assert ps.to_dict() == {"P1": "positive", "P2": "positive"}
    assert ys.to_dict() == {"P1": "positive", "P2": "administered_negative"}


def test_construct_status_either_is_max_both_requires_agreement():
    parent = _caseness([("P1", "Depression", "positive"),
                        ("P2", "Depression", "positive")])
    youth = _caseness([("P1", "Depression", "positive"),
                       ("P2", "Depression", "administered_negative")])
    cache = _cache(parent, youth)
    either = mv.construct_status(cache, "depression", "ever_met", "either", False, "phobia_in")
    both = mv.construct_status(cache, "depression", "ever_met", "both", False, "phobia_in")
    # either: P1 positive|positive, P2 positive|neg -> both positive
    assert either.to_dict() == {"P1": "positive", "P2": "positive"}
    # both: only P1 is positive on both informants
    assert both.to_dict() == {"P1": "positive", "P2": "administered_negative"}


def test_construct_status_multi_category_construct():
    # 'externalizing' aggregates ADHD/ODD/Conduct.
    parent = _caseness([("P1", "ADHD", "administered_negative"),
                        ("P1", "Conduct", "positive")])
    youth = _caseness([("P1", "ADHD", "administered_negative"),
                       ("P1", "Conduct", "administered_negative")])
    cache = _cache(parent, youth)
    ps = mv.construct_status(cache, "externalizing", "ever_met", "parent", False, "phobia_in")
    assert ps["P1"] == "positive"


# ---- _phobia_crosswalk ------------------------------------------------------


def test_phobia_crosswalk_out_drops_phobia_in_keeps_all():
    cw = pd.DataFrame({"module": ["phobia", "gad"], "category": ["Anxiety", "Anxiety"]})
    assert set(mv._phobia_crosswalk(cw, "phobia_out").module) == {"gad"}
    assert set(mv._phobia_crosswalk(cw, "phobia_in").module) == {"phobia", "gad"}


# ---- informant_validity -----------------------------------------------------


def test_informant_validity_uses_baseline_administration_only():
    cw = pd.DataFrame({
        "module": ["dep", "dep"],
        "category": ["Depression", "Depression"],
        "informant": ["parent", "youth"],
    })
    cal = pd.DataFrame({
        "session_id": ["ses-00A", "ses-00A", "ses-02A"],
        "status": ["administered", "not_administered", "administered"],
        "informant": ["parent", "youth", "youth"],
        "module": ["dep", "dep", "dep"],
    })
    valid = mv.informant_validity(cw, cal)
    # parent administered at baseline; youth only administered at a later wave -> not valid
    assert valid["depression"] == {
        "parent": True, "youth": False, "either": True, "both": False
    }


# ---- build_primitive_cache (orchestration smoke test) -----------------------


# ---- summarize_multiverse ---------------------------------------------------


def test_summarize_multiverse_fold_range_and_instability():
    grid = pd.DataFrame({
        "construct": ["A", "A", "A", "B", "B"],
        "prevalence_pct": [2.0, 4.0, 8.0, 0.05, 5.0],
    })
    summ = mv.summarize_multiverse(grid).set_index("construct")
    assert summ.loc["A", "fold_range"] == 4.0        # 8 / 2
    assert summ.loc["A", "pp_span"] == 6.0
    assert bool(summ.loc["A", "unstable_fold"]) is False
    assert bool(summ.loc["B", "unstable_fold"]) is True   # min 0.05 < TINY
    # sorted by fold_range descending -> B (fold 100) first
    assert mv.summarize_multiverse(grid).construct.iloc[0] == "B"


# ---- build_multiverse_grid --------------------------------------------------


def _valid(**flags):
    return {"parent": False, "youth": False, "either": False, "both": False, **flags}


def test_build_multiverse_grid_gates_informants_and_phobia(monkeypatch):
    monkeypatch.setattr(
        mv, "construct_status",
        lambda *a, **k: pd.Series(["positive"] * 5 + ["administered_negative"] * 45),
    )
    monkeypatch.setattr(mv, "prevalence", lambda stat: (10.0, 5, 50))
    cats_for = {"depression": ["Depression"], "anxiety": ["Anxiety"]}
    valid = {
        "depression": _valid(parent=True, either=True),          # youth/both invalid
        "anxiety": _valid(parent=True, youth=True, either=True, both=True),
    }
    grid, skipped = mv.build_multiverse_grid(cache={}, valid=valid, cats_for=cats_for)
    # depression: 2 valid informants x 2 statuses x 2 thresholds x 1 phobia = 8
    # anxiety:    4 informants x 2 statuses x 2 thresholds x 2 phobias      = 32
    assert len(grid) == 40
    assert skipped == 4                                          # depression youth/both x 2 statuses
    # phobia_out only exists for anxiety
    assert set(grid[grid.phobia == "phobia_out"].construct) == {"anxiety"}
    assert set(grid.threshold) == {"full", "with_subthreshold"}


def test_build_multiverse_grid_skips_empty_denominator(monkeypatch):
    monkeypatch.setattr(mv, "construct_status", lambda *a, **k: pd.Series(["not_administered"]))
    monkeypatch.setattr(mv, "prevalence", lambda stat: (float("nan"), 0, 0))
    valid = {"depression": _valid(parent=True)}
    grid, skipped = mv.build_multiverse_grid(
        cache={}, valid=valid, cats_for={"depression": ["Depression"]}
    )
    assert len(grid) == 0
    assert skipped > 0


# ---- single_lever_table -----------------------------------------------------


def test_single_lever_table_deltas_and_ordering(monkeypatch):
    def fake_cs(cache, con, status, informant, subthr, phobia):
        if status == "ever_met":
            k = 20
        elif informant == "youth":
            k = 5
        elif informant == "either":
            k = 15
        elif subthr:
            k = 12
        elif phobia == "phobia_out":
            k = 8
        else:
            k = 10  # base
        return pd.Series(["positive"] * k + ["administered_negative"] * (100 - k))

    monkeypatch.setattr(mv, "construct_status", fake_cs)  # real prevalence -> pct == k
    df = mv.single_lever_table(cache={})
    assert df.iloc[0].lever.startswith("base")
    assert df.iloc[0].prevalence_pct == 10.0 and df.iloc[0].delta_pts == 0.0
    deltas = dict(zip(df.lever, df.delta_pts))
    assert deltas["current -> ever-met"] == 10.0
    assert deltas["parent -> youth-only"] == -5.0
    assert deltas["anxiety: drop phobia"] == -2.0
    # body ordered by absolute delta -> the +10 flip is the first non-base row
    assert df.iloc[1].lever == "current -> ever-met"


def test_build_primitive_cache_has_expected_keys():
    cw = pd.DataFrame(
        [("v_dep_p", "parent", "dep", "present", "Depression", 0),
         ("v_dep_y", "youth", "dep", "present", "Depression", 0)],
        columns=["variable", "informant", "module", "status_layer",
                 "category", "is_subthreshold"],
    )
    base = pd.DataFrame(
        [("P1", "ses-00A", "v_dep_p", "positive"),
         ("P1", "ses-00A", "v_dep_y", "administered_negative")],
        columns=["participant_id", "session_id", "variable", "resolved"],
    )
    cache = mv.build_primitive_cache(base, cw)
    expected = {(s, t, p)
                for s in ("current", "ever_met")
                for t in (False, True)
                for p in ("phobia_in", "phobia_out")}
    assert set(cache) == expected
    assert set(cache[("current", False, "phobia_in")]) == {"parent", "youth"}

"""Tests for module over-screening prevalence."""

import pandas as pd

from abcd_ksads import overscreening as ov


def _rows(rows):
    return pd.DataFrame(
        rows, columns=["participant_id", "module", "status_layer", "resolved"]
    )


# ---- prevalence_over_assessed -----------------------------------------------


def test_prevalence_over_assessed_excludes_not_administered():
    d = _rows([
        ("p1", "dep", "present", "positive"),
        ("p2", "dep", "present", "administered_negative"),
        ("p3", "dep", "present", "not_administered"),   # not in the denominator
    ])
    npos, nadm, pct = ov.prevalence_over_assessed(d)
    assert npos == 1 and nadm == 2
    assert pct == 50.0


def test_prevalence_over_assessed_empty_is_zero():
    d = _rows([("p1", "dep", "present", "not_administered")])
    assert ov.prevalence_over_assessed(d) == (0, 0, 0.0)


# ---- module_overscreening ---------------------------------------------------


def test_module_overscreening_sorted_high_to_low():
    base = _rows([
        # phobia: 2/2 = 100%
        ("p1", "phobia", "present", "positive"),
        ("p2", "phobia", "present", "positive"),
        # adhd: 1/2 = 50%
        ("p1", "adhd", "present", "positive"),
        ("p2", "adhd", "present", "administered_negative"),
    ])
    tab = ov.module_overscreening(base, labels={"phobia": "Specific phobia", "adhd": "ADHD"},
                                  epi={"phobia": "~5", "adhd": "~7-9"})
    assert tab.module.tolist() == ["phobia", "adhd"]   # descending prevalence
    assert tab.iloc[0].present_core_pct == 100.0
    assert tab.iloc[1].present_core_pct == 50.0
    assert tab.iloc[0].approx_childhood_pct == "~5"


# ---- depression_breakdown ---------------------------------------------------


def test_depression_breakdown_present_past_ratio():
    base = _rows([
        ("p1", "dep", "present", "positive"),
        ("p2", "dep", "present", "administered_negative"),
        ("p1", "dep", "past", "positive"),
        ("p2", "dep", "past", "positive"),
    ])
    sb = ov.depression_breakdown(base).iloc[0]
    assert sb.n_present == 1 and sb.present_pct == 50.0
    assert sb.n_past == 2 and sb.past_pct == 100.0
    assert sb.past_to_present_ratio == 2.0

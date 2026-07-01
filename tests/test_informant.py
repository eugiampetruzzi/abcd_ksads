"""Tests for informant prevalence and parent-youth concordance."""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from abcd_ksads import informant


def test_prevalence_counts_over_administered_denominator():
    s = pd.Series(["positive", "positive", "administered_negative", "not_administered"])
    out = informant.prevalence(s)
    assert out["n_positive"] == 2
    assert out["n_denominator"] == 3  # not_administered excluded
    assert out["prevalence_pct"] == 100 * 2 / 3


def test_prevalence_empty_denominator_is_nan():
    s = pd.Series(["not_administered", "not_administered"])
    out = informant.prevalence(s)
    assert out["n_denominator"] == 0
    assert np.isnan(out["prevalence_pct"])


def _paired(both, ponly, yonly, neither):
    """Build parent/youth per-person Series with the given 2x2 counts."""
    p, y, idx = [], [], []
    n = 0
    for label, (pv, yv), k in [
        ("both", ("positive", "positive"), both),
        ("ponly", ("positive", "administered_negative"), ponly),
        ("yonly", ("administered_negative", "positive"), yonly),
        ("neither", ("administered_negative", "administered_negative"), neither),
    ]:
        for _ in range(k):
            idx.append(f"s{n}")
            p.append(pv)
            y.append(yv)
            n += 1
    ps = pd.Series(p, index=idx)
    ys = pd.Series(y, index=idx)
    return ps, ys


def test_concordance_2x2_counts():
    ps, ys = _paired(both=3, ponly=1, yonly=1, neither=5)
    out = informant.concordance(ps, ys)
    assert out["n_both_admin"] == 10
    assert out["both_pos"] == 3
    assert out["parent_only"] == 1
    assert out["youth_only"] == 1
    assert out["union_pos"] == 5


def test_concordance_kappa_matches_analytic_and_sklearn():
    ps, ys = _paired(both=3, ponly=1, yonly=1, neither=5)
    out = informant.concordance(ps, ys)
    # analytic Cohen's kappa for this table = 0.5833...
    assert abs(out["kappa"] - 0.5833333333333333) < 1e-9
    pb = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    yb = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    assert abs(out["kappa"] - cohen_kappa_score(pb, yb)) < 1e-12


def test_concordance_excludes_not_administered_and_unpaired():
    ps, ys = _paired(both=3, ponly=1, yonly=1, neither=5)
    # add pairs that must be dropped: one not-administered on each side, one unpaired
    ps = pd.concat([ps, pd.Series({"x1": "not_administered", "x2": "positive", "x3": "positive"})])
    ys = pd.concat([ys, pd.Series({"x1": "positive", "x2": "not_administered"})])  # x3 absent from ys
    out = informant.concordance(ps, ys)
    assert out["n_both_admin"] == 10  # only the fully-paired, both-administered rows count


def test_concordance_no_overlap_returns_none():
    ps = pd.Series({"a": "positive", "b": "administered_negative"})
    ys = pd.Series({"c": "positive", "d": "positive"})
    assert informant.concordance(ps, ys) is None

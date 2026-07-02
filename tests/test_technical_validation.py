"""Tests for the technical-validation checks."""

import numpy as np
import pandas as pd

from abcd_ksads import technical_validation as tv


# ---- correctness_by_category ------------------------------------------------


def test_correctness_by_category_totals_and_match_flag():
    res_pos = {"v1": 5, "v2": 3, "v3": 10}
    raw_one = {"v1": 5, "v2": 4, "v3": 10}          # v2 differs
    cw_cat = {"v1": "A", "v2": "A", "v3": "B"}
    df = tv.correctness_by_category(res_pos, raw_one, cw_cat).set_index("category")
    assert df.loc["A", "n_resolved_positive"] == 8 and df.loc["A", "n_raw_value1"] == 9
    assert bool(df.loc["A", "match"]) is False
    assert bool(df.loc["B", "match"]) is True


# ---- caseness_prevalence ----------------------------------------------------


def test_caseness_prevalence_over_assessed_denominator():
    caseness = pd.DataFrame(
        [
            ("p1", "Depression", "positive"),
            ("p2", "Depression", "administered_negative"),
            ("p3", "Depression", "not_administered"),   # excluded from denominator
        ],
        columns=["participant_id", "category", "status"],
    )
    assert tv.caseness_prevalence(caseness, ["Depression"]) == 50.0


def test_caseness_prevalence_nan_when_none_assessed():
    # every row not_administered -> empty assessed denominator -> NaN, not a divide error
    caseness = pd.DataFrame(
        [
            ("p1", "Depression", "not_administered"),
            ("p2", "Depression", "not_administered"),
        ],
        columns=["participant_id", "category", "status"],
    )
    assert np.isnan(tv.caseness_prevalence(caseness, ["Depression"]))


# ---- concordance_kappa ------------------------------------------------------


def _cy(n_pos, n_neg):
    parts = [f"p{i}" for i in range(n_pos + n_neg)]
    status = ["positive"] * n_pos + ["administered_negative"] * n_neg
    return pd.DataFrame({"participant_id": parts, "category": "Depression", "status": status})


def test_concordance_kappa_perfect_agreement():
    cp = _cy(30, 30)
    row = tv.concordance_kappa(cp, cp.copy(), "Depression")
    assert row["n_both_assessed"] == 60
    assert row["both_positive"] == 30
    assert row["cohen_kappa"] == 1.0


def test_concordance_kappa_nan_when_too_few_assessed():
    cp = _cy(1, 1)
    row = tv.concordance_kappa(cp, cp.copy(), "Depression")
    assert np.isnan(row["cohen_kappa"])
    assert row["n_both_assessed"] == 2

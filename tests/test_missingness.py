"""Tests for the 555-as-0 missingness deflation statistics."""

import pandas as pd

from abcd_ksads import missingness as ms


def _resolved(states):
    return pd.DataFrame({"resolved": states})


def test_missingness_error_computes_correct_error_and_fold():
    # 10 positive, 30 administered_negative (assessed=40), 60 not_administered (all=100)
    d = _resolved(["positive"] * 10 + ["administered_negative"] * 30
                  + ["not_administered"] * 60)
    out = ms.missingness_error(d)
    assert out["n_positive"] == 10
    assert out["n_administered"] == 40
    assert out["n_all_personwaves"] == 100
    assert out["fabricated_personwaves"] == 60
    assert out["prevalence_correct_pct"] == 25.0    # 10/40
    assert out["prevalence_error_pct"] == 10.0       # 10/100
    assert out["fold_deflation"] == 2.5              # 25 / 10 == all/assessed


def test_missingness_error_no_deflation_when_all_assessed():
    d = _resolved(["positive"] * 10 + ["administered_negative"] * 90)
    out = ms.missingness_error(d)
    assert out["prevalence_correct_pct"] == out["prevalence_error_pct"]
    assert out["fold_deflation"] == 1.0
    assert out["fabricated_personwaves"] == 0

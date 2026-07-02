"""Tests for collating reported paper numbers."""

import numpy as np
import pandas as pd

from abcd_ksads import paper_numbers as pn


def _inputs():
    rs = pd.DataFrame({
        "n_positive": [10], "n_administered_negative": [30],
        "n_not_administered": [40], "n_no_record": [20],
    })  # 100 cells total
    cw = pd.DataFrame({
        "variable": ["v1", "v2", "v3"],
        "category": ["A", "A", "B"],
        "is_subthreshold": [0, 1, 0],
    })
    msum = pd.DataFrame({
        "construct": ["any-disorder", "suicidality"],
        "n_specs": [12, 8],
        "prev_min": [10.0, 0.05], "prev_max": [40.0, 5.0],
        "fold_range": [4.0, 100.0], "pp_span": [30.0, 4.95],
        "prev_median": [20.0, 1.0],
        "unstable_fold": [False, True],   # suicidality off a near-zero base
    })
    lev = pd.DataFrame({
        "lever": ["base (current, parent, full, phobia-in)", "current -> ever-met",
                  "parent -> either"],
        "prevalence_pct": [15.0, 25.0, 20.0],
        "delta_pts": [0.0, 10.0, 5.0],
    })
    anx = pd.DataFrame({
        "sub": ["Specific phobia", "ANY (without phobia)", "ANY (with phobia)"],
        "prevalence_pct": [5.0, 8.0, 12.0],
    })
    miss = pd.Series({
        "prevalence_correct_pct": 25.0, "prevalence_error_pct": 10.0,
        "fold_deflation": 2.5, "fabricated_personwaves": 60,
    })
    ver = pd.DataFrame({
        "ksads_version": ["1.0", "1.0", "2.0"],
        "version_valid": [False, True, True],
    })
    return rs, cw, msum, lev, anx, miss, ver


def test_collate_numbers_aggregates_and_headlines():
    out = pn.collate_numbers(*_inputs())
    # cell-share aggregates
    assert out["n_cells_total"] == 100
    assert out["pct_positive"] == 10.0
    assert out["pct_555"] == 40.0
    # crosswalk counts
    assert out["n_diagnosis_vars"] == 3
    assert out["n_categories"] == 2
    assert out["n_subthreshold"] == 1
    # headline max fold picks the largest fold overall (suicidality, flagged unstable)
    assert out["multiverse"]["headline_max_fold"]["construct"] == "suicidality"
    assert out["multiverse"]["headline_max_fold"]["unstable"] is True
    # max *stable* fold excludes the near-zero-base construct
    assert out["multiverse"]["headline_max_stable_fold"]["construct"] == "any-disorder"


def test_collate_numbers_single_lever_anxiety_and_version():
    out = pn.collate_numbers(*_inputs())
    assert out["single_lever"] == {"ever_met": 10.0, "either": 5.0, "base_prevalence": 15.0}
    assert out["anxiety"]["fold"] == 1.5           # 12 / 8
    assert out["anxiety"]["phobia_only"] == 5.0
    assert out["version"] == {"cells_v1": 2, "cells_v2": 1, "two_zero_only_under_one": 1}


def test_num_coerces_nan_to_none():
    assert pn._num(np.float64("nan")) is None
    assert pn._num(np.int64(3)) == 3

"""Tests for the anxiety-construct decomposition."""

import pandas as pd

from abcd_ksads import anxiety as ax


def _rows(rows):
    return pd.DataFrame(rows, columns=["participant_id", "module", "resolved"])


def test_decompose_anxiety_per_sub_and_cumulative_with_without_phobia():
    # gad: p1 positive; phobia: p2 positive; both assessed on both modules
    base = _rows([
        ("p1", "gad", "positive"),
        ("p2", "gad", "administered_negative"),
        ("p1", "phobia", "administered_negative"),
        ("p2", "phobia", "positive"),
    ])
    subs = [("gad", "GAD"), ("phobia", "Specific phobia")]
    dec, cum, any_with, any_without, n_assessed = ax.decompose_anxiety(base, subs=subs)

    by = dict(zip(dec["sub"], dec.prevalence_pct))
    assert by["GAD"] == 50.0 and by["Specific phobia"] == 50.0
    assert n_assessed == 2                       # union assessed = {p1, p2}
    # any-anxiety with phobia = both p1 and p2 positive somewhere -> 100%
    assert any_with == 100.0
    # excluding phobia, only p1 (gad) is positive -> 50%
    assert any_without == 50.0
    # cumulative is monotonic and ends at any_with
    assert cum[-1] == any_with and cum[0] <= cum[-1]

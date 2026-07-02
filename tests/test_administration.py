"""Tests for the administration-calendar classification and cadence logic."""

import pandas as pd

from abcd_ksads import administration as adm


# ---- classify ---------------------------------------------------------------


def test_classify_administered_when_coverage_meets_threshold():
    # administered / in_release = (5+5)/(5+5+0) = 1.0 >= 0.5
    assert adm.classify(pos=5, neg=5, not_admin=0, no_rec=0) == "administered"


def test_classify_not_administered_when_all_555():
    # in_release>0 but administered==0 -> not_administered
    assert adm.classify(pos=0, neg=0, not_admin=20, no_rec=0) == "not_administered"


def test_classify_absent_when_no_cells_in_release():
    # only no_record present -> module/variable not in the release at this wave
    assert adm.classify(pos=0, neg=0, not_admin=0, no_rec=30) == "absent"


def test_classify_partial_coverage_counts_as_administered():
    # administered>0 but below threshold -> still "administered" (non-trivial coverage)
    assert adm.classify(pos=1, neg=0, not_admin=9, no_rec=0) == "administered"


# ---- administration_flags ---------------------------------------------------


def _status(**by_session):
    return pd.Series(by_session)


def test_flags_never_administered():
    st = _status(**{s: "absent" for s in adm.SESSIONS})
    assert adm.administration_flags(st) == "never_administered"


def test_flags_added_after_baseline():
    st = _status(**{s: "absent" for s in adm.SESSIONS})
    st["ses-02A"] = "administered"
    st["ses-04A"] = "administered"
    st["ses-06A"] = "administered"
    assert adm.administration_flags(st) == "added@02A"


def test_flags_dropped_before_final_wave():
    st = _status(**{s: "absent" for s in adm.SESSIONS})
    st["ses-00A"] = "administered"
    st["ses-02A"] = "administered"   # stops before the ses-06A endpoint
    assert adm.administration_flags(st) == "dropped_after@02A"


def test_flags_intermittent_hole_between_even_waves():
    st = _status(**{s: "absent" for s in adm.SESSIONS})
    st["ses-00A"] = "administered"
    st["ses-04A"] = "administered"   # ses-02A missing between endpoints
    st["ses-06A"] = "administered"
    assert "intermittent" in adm.administration_flags(st)


def test_flags_clean_full_administration_has_no_notes():
    st = _status(**{s: "administered" for s in adm.SESSIONS})
    assert adm.administration_flags(st) == ""


# ---- build_calendar ---------------------------------------------------------


def _summary_rows(rows):
    cols = ["informant", "module", "session_id", "status_layer",
            "n_positive", "n_administered_negative", "n_not_administered", "n_no_record"]
    return pd.DataFrame(rows, columns=cols)


def test_build_calendar_produces_long_and_grid():
    # one module administered at baseline, not-administered (all 555) at ses-02A
    summary = _summary_rows([
        ("parent", "dep", "ses-00A", "present", 5, 15, 0, 0),
        ("parent", "dep", "ses-02A", "present", 0, 0, 20, 0),
    ])
    long, grid = adm.build_calendar(summary)
    base = long[long.session_id == "ses-00A"].iloc[0]
    assert base.status == "administered"
    assert base.n_administered == 20                      # 5 + 15
    assert long[long.session_id == "ses-02A"].iloc[0].status == "not_administered"
    # grid has one row per (informant, module) with the session columns present
    assert len(grid) == 1
    assert grid.iloc[0]["ses-00A"] == "X" and grid.iloc[0]["ses-02A"] == "."

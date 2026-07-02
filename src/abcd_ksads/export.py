"""Per-session interview age for the analysis-ready CSV export.

Interview age is taken from the KSADS interview metadata in the cache: the youngest
reported age (parent or youth) wins per participant-session, converted years -> months.
The ``15_export_analysis_csv.py`` script wires this to the CSVs.
"""

import pandas as pd

V1 = {"ses-00A", "ses-01A", "ses-02A"}  # KSADS-COMP 1.0 waves
# youngest reported age wins; parent then youth KSADS dep interview age
AGE_COLS = {"mh_p": "mh_p_ksads__dep_age", "mh_y": "mh_y_ksads__dep_age"}


def interview_ages(wide, age_cols=AGE_COLS, v1_waves=V1):
    """participant x session interview_age (months) + ksads_version, youngest age wins."""
    def age_rows(pref):
        d = wide[["participant_id", "session_id", age_cols[pref]]].copy()
        d.columns = ["participant_id", "session_id", "age_yr"]
        return d

    ad = pd.concat([age_rows(k) for k in age_cols], ignore_index=True)
    ad["age_yr"] = pd.to_numeric(ad["age_yr"].astype("object"), errors="coerce")
    ad = ad.dropna(subset=["age_yr"]).sort_values("age_yr")
    ad = ad.drop_duplicates(["participant_id", "session_id"], keep="first")
    ad["interview_age"] = (ad["age_yr"] * 12).round().astype("Int64")
    ad["ksads_version"] = ad["session_id"].apply(lambda s: "1.0" if s in v1_waves else "2.0")
    return ad[["participant_id", "session_id", "interview_age", "ksads_version"]]

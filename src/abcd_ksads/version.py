"""KSADS-COMP version provenance (1.0 vs 2.0) and version validity.

The instrument switched from 1.0 to 2.0 at the 3-year follow-up (ses-03A). A value
is version-invalid if it records a 2.0-only diagnosis under a 1.0 wave. The
``04_version_provenance.py`` script wires this to the resolved cache and CSVs.
"""

import pandas as pd

V1_WAVES = {"ses-00A", "ses-01A", "ses-02A"}  # KSADS-COMP 1.0
V2_SWITCH = "ses-03A"  # 2.0 from here on


def tag_versions(resolved, two_zero_only_vars, v1_waves=V1_WAVES):
    """Add ksads_version, two_zero_only, and version_valid columns to a resolved frame.

    version_valid is False only for a 2.0-only diagnosis recorded under a 1.0 wave."""
    out = resolved.copy()
    out["ksads_version"] = out.session_id.map(lambda s: "1.0" if s in v1_waves else "2.0")
    out["two_zero_only"] = out.variable.isin(set(two_zero_only_vars))
    out["version_valid"] = ~(out.two_zero_only & (out.ksads_version == "1.0"))
    return out


def audit_pre_switch(resolved, v1_waves=V1_WAVES):
    """Documented 2.0-only diagnoses that nonetheless have administered cells under 1.0.

    Surfaces the documentation-vs-release discrepancy: rows are variable x session with
    a positive count of administered (positive + administered_negative) cells pre-switch.
    """
    pre = resolved[resolved.two_zero_only & resolved.session_id.isin(v1_waves)]
    audit = (
        pre.groupby(["variable", "session_id"])["resolved"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("positive", "administered_negative", "not_administered", "no_record"):
        if col not in audit:
            audit[col] = 0
    audit["administered_pre_switch"] = audit.positive + audit.administered_negative
    return audit[audit.administered_pre_switch > 0]

#!/usr/bin/env python3
import csv
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DERIV = os.path.join(HERE, "derivatives")
XWALK = os.path.join(config.CODEBOOKS, "ksads_version_crosswalk.csv")

V1_WAVES = {"ses-00A", "ses-01A", "ses-02A"}  # KSADS-COMP 1.0
V2_SWITCH = "ses-03A"  # 2.0 from here on


def main():
    resolved = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    two_oh_only = {
        r["merged_var"]
        for r in csv.DictReader(open(XWALK))
        if r["two_zero_only_flag"].strip()
    }

    resolved["ksads_version"] = resolved.session_id.map(
        lambda s: "1.0" if s in V1_WAVES else "2.0"
    )
    resolved["two_zero_only"] = resolved.variable.isin(two_oh_only)
    # a value is version-valid unless it is a 2.0-only diagnosis recorded under 1.0
    resolved["version_valid"] = ~(
        resolved.two_zero_only & (resolved.ksads_version == "1.0")
    )

    out = os.path.join(DERIV, "ksads_resolved_versioned.parquet")
    resolved.to_parquet(out, index=False)

    # audit: do the documented 2.0-only diagnoses actually have administered cells
    # before the switch? (documentation-vs-release discrepancy)
    pre = resolved[resolved.two_zero_only & resolved.session_id.isin(V1_WAVES)]
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
    audit = audit[audit.administered_pre_switch > 0]
    audit.to_csv(os.path.join(DERIV, "ksads_version_audit.csv"), index=False)

    n = len(resolved)
    print("Layer 4 version provenance")
    print(
        f"  cells tagged 1.0: {int((resolved.ksads_version == '1.0').sum()):,}  "
        f"2.0: {int((resolved.ksads_version == '2.0').sum()):,}"
    )
    print(
        f"  2.0-only diagnosis variables: {len(two_oh_only)} "
        f"({resolved.two_zero_only.sum():,} cells)"
    )
    n_invalid = int((~resolved.version_valid).sum())
    print(f"  version-invalid cells (2.0-only recorded under 1.0): {n_invalid:,}")
    if len(audit):
        print(
            f"\n  DISCREPANCY: {audit.variable.nunique()} documented 2.0-only diagnoses "
            f"have administered cells BEFORE the switch:"
        )
        for v, sub in audit.groupby("variable"):
            waves = ", ".join(
                f"{r.session_id[-3:]}(n={int(r.administered_pre_switch)})"
                for _, r in sub.iterrows()
            )
            print(f"    {v}: {waves}")
        print(
            "  -> release notes call these 2.0-only; the BIDS exposes pre-switch values."
        )
    else:
        print(
            "  No 2.0-only diagnoses administered before the switch (matches the notes)."
        )
    print(f"\nWrote {out}")
    print(f"Wrote {DERIV}/ksads_version_audit.csv")


if __name__ == "__main__":
    main()

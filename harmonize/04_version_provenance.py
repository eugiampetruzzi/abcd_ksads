#!/usr/bin/env python3
"""KSADS-COMP version provenance; the tagging/audit logic lives in abcd_ksads.version."""

import csv
import os

import pandas as pd

from abcd_ksads import config
from abcd_ksads.version import audit_pre_switch, tag_versions

XWALK = os.path.join(config.CODEBOOKS, "ksads_version_crosswalk.csv")


def main():
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    two_oh_only = {
        r["merged_var"]
        for r in csv.DictReader(open(XWALK))
        if r["two_zero_only_flag"].strip()
    }

    resolved = tag_versions(resolved, two_oh_only)
    out = config.DERIV / "ksads_resolved_versioned.parquet"
    resolved.to_parquet(out, index=False)

    audit = audit_pre_switch(resolved)
    audit.to_csv(config.DERIV / "ksads_version_audit.csv", index=False)

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
    print(f"Wrote {config.DERIV.as_posix()}/ksads_version_audit.csv")


if __name__ == "__main__":
    main()

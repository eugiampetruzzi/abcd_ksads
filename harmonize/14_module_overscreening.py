#!/usr/bin/env python3
"""Module over-screening: baseline parent present-diagnosis prevalence by module; the
prevalence logic lives in abcd_ksads.overscreening."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.overscreening import depression_breakdown, module_overscreening


def main():
    cw = build_crosswalk()
    sub = cw.set_index("variable")["is_subthreshold"].to_dict()
    r = pd.read_parquet(
        config.DERIV / "ksads_resolved_long.parquet",
        columns=[
            "participant_id", "session_id", "informant", "module",
            "variable", "status_layer", "resolved",
        ],
    )
    for c in ["session_id", "informant", "module", "variable", "status_layer", "resolved"]:
        r[c] = r[c].astype(str)
    r["issub"] = r.variable.map(sub).fillna(0)
    base = r[(r.session_id == "ses-00A") & (r.informant == "parent") & (r.issub == 0)]

    tab = module_overscreening(base)
    tab.to_csv(config.DERIV / "module_overscreening.csv", index=False)
    print("Core-criteria baseline parent PRESENT prevalence (subthreshold excluded):\n")
    print(tab.to_string(index=False))

    sb = depression_breakdown(base)
    sb.to_csv(config.DERIV / "status_depression_breakdown.csv", index=False)
    print("\nDepression status breakdown (core criteria, baseline, parent):")
    print(sb.to_string(index=False))
    print("\nWrote module_overscreening.csv and status_depression_breakdown.csv")


if __name__ == "__main__":
    main()

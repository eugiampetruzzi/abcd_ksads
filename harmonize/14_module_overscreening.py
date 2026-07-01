#!/usr/bin/env python3
"""Module over-screening: baseline parent present-diagnosis prevalence by module."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk

# approximate childhood (≈9-10 yr) prevalence context, for flagging only
EPI = {
    "phobia": "~5",
    "ocd": "1-2",
    "bpd": "<1",
    "psych": "<0.5",
    "adhd": "~7-9",
    "odd": "~3-5",
    "cond": "~1-2",
    "dep": "<1 (current)",
    "gad": "~1",
    "socanx": "~1-2",
    "panic": "<1",
    "ptsd": "~1",
}
LABEL = {
    "phobia": "Specific phobia",
    "ocd": "OCD",
    "bpd": "Bipolar",
    "psych": "Psychotic disorders",
    "adhd": "ADHD",
    "odd": "ODD",
    "cond": "Conduct",
    "dep": "Depression (MDD/PDD)",
    "gad": "GAD",
    "socanx": "Social anxiety",
    "panic": "Panic",
    "ptsd": "PTSD",
}


def main():
    cw = build_crosswalk()
    sub = cw.set_index("variable")["is_subthreshold"].to_dict()
    r = pd.read_parquet(
        config.DERIV / "ksads_resolved_long.parquet",
        columns=[
            "participant_id",
            "session_id",
            "informant",
            "module",
            "variable",
            "status_layer",
            "resolved",
        ],
    )
    for c in [
        "session_id",
        "informant",
        "module",
        "variable",
        "status_layer",
        "resolved",
    ]:
        r[c] = r[c].astype(str)
    r["issub"] = r.variable.map(sub).fillna(0)
    base = r[(r.session_id == "ses-00A") & (r.informant == "parent") & (r.issub == 0)]

    def prev(d):
        pos = set(d[d.resolved == "positive"].participant_id)
        adm = set(
            d[d.resolved.isin(["positive", "administered_negative"])].participant_id
        )
        return len(pos), len(adm), (100 * len(pos) / len(adm) if adm else 0.0)

    rows = []
    for mod in LABEL:
        npos, nadm, pct = prev(
            base[(base.module == mod) & (base.status_layer == "present")]
        )
        rows.append(
            {
                "disorder": LABEL[mod],
                "module": mod,
                "present_core_pct": round(pct, 2),
                "n_positive": npos,
                "n_administered": nadm,
                "approx_childhood_pct": EPI.get(mod, ""),
            }
        )
    tab = pd.DataFrame(rows).sort_values("present_core_pct", ascending=False)
    tab.to_csv(config.DERIV / "module_overscreening.csv", index=False)
    print("Core-criteria baseline parent PRESENT prevalence (subthreshold excluded):\n")
    print(tab.to_string(index=False))

    # depression current vs past, core only (MDD + PDD; exclude unspecified)
    dep = base[base.module == "dep"]
    pn, _, pp = prev(dep[dep.status_layer == "present"])
    an, _, ap = prev(dep[dep.status_layer == "past"])
    sb = pd.DataFrame(
        [
            {
                "construct": "Depression (core MDD/PDD)",
                "present_pct": round(pp, 2),
                "n_present": pn,
                "past_pct": round(ap, 2),
                "n_past": an,
                "past_to_present_ratio": round(an / pn, 1) if pn else None,
            }
        ]
    )
    sb.to_csv(config.DERIV / "status_depression_breakdown.csv", index=False)
    print("\nDepression status breakdown (core criteria, baseline, parent):")
    print(sb.to_string(index=False))
    print("\nWrote module_overscreening.csv and status_depression_breakdown.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Informant discrepancy: caregiver vs youth prevalence and concordance by category."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk, build_caseness
from abcd_ksads.multiverse import _agg
from abcd_ksads.informant import prevalence, concordance

# Categories assessed in both the caregiver and youth interviews.
CATS = [
    "Depression",
    "Anxiety",
    "Bipolar",
    "DMDD",
    "Conduct",
    "Eating",
    "OCD",
    "PTSD",
    "Suicidality",
]
SESSIONS = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]  # full battery, both informants


def per_person(caseobj, cat):
    return _agg(
        caseobj, [cat]
    )  # Series: participant_id -> positive/administered_negative/not_administered


def main():
    cw = build_crosswalk()
    # Fair informant comparison: keep only modules assessed in BOTH interviews,
    # so e.g. Anxiety is compared on its shared modules (gad/socanx/panic), not
    # the caregiver-only agoraphobia/separation/phobia modules.
    both_mods = set(cw[cw.informant == "parent"].module) & set(
        cw[cw.informant == "youth"].module
    )
    cw = cw[cw.module.isin(both_mods)].copy()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    prev_rows, conc_rows = [], []
    for ses in SESSIONS:
        base = resolved[resolved.session_id == ses].copy()
        if base.empty:
            continue
        cp = build_caseness(
            base,
            cw,
            status_set="current",
            include_subthreshold=False,
            informant="parent",
        )
        cy = build_caseness(
            base,
            cw,
            status_set="current",
            include_subthreshold=False,
            informant="youth",
        )
        for cat in CATS:
            ps = per_person(cp, cat)
            ys = per_person(cy, cat)
            for inf, s in (("parent", ps), ("youth", ys)):
                p = prevalence(s)
                prev_rows.append(
                    {
                        "session": ses,
                        "category": cat,
                        "informant": inf,
                        "prevalence_pct": p["prevalence_pct"],
                        "n_positive": p["n_positive"],
                        "n_denominator": p["n_denominator"],
                    }
                )
            conc = concordance(ps, ys)
            if conc is None:
                continue
            conc_rows.append({"session": ses, "category": cat, **conc})

    pd.DataFrame(prev_rows).to_csv(
        config.DERIV / "informant_prevalence.csv", index=False
    )
    C = pd.DataFrame(conc_rows)
    C.to_csv(config.DERIV / "informant_concordance.csv", index=False)

    base = C[C.session == "ses-00A"].copy()
    print("Baseline (ses-00A) informant discrepancy by category:")
    print(
        f"{'category':12}{'kappa':>7}{'parent+':>9}{'youth+':>8}{'both':>6}{'P-only':>8}{'Y-only':>8}"
    )
    for _, r in base.iterrows():
        print(
            f"{r.category:12}{r.kappa:>7.2f}{r.both_pos + r.parent_only:>9}{r.both_pos + r.youth_only:>8}"
            f"{r.both_pos:>6}{r.parent_only:>8}{r.youth_only:>8}"
        )
    print(f"\nmean baseline kappa: {base.kappa.mean():.3f}")
    print("Wrote informant_prevalence.csv and informant_concordance.csv")


if __name__ == "__main__":
    main()

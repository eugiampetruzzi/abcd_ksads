#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
from abcd_ksads import config
from abcd_ksads.category_crosswalk import (
    build_crosswalk,
    build_caseness,
)

# module -> (DSM category, [broadband dimensions])
CATEGORY = {
    "dep": ("Depression", ["Internalizing", "Mood"]),
    "bpd": ("Bipolar", ["Mood"]),
    "dmdd": ("DMDD", ["Mood"]),
    "gad": ("Anxiety", ["Internalizing"]),
    "sepanx": ("Anxiety", ["Internalizing"]),
    "socanx": ("Anxiety", ["Internalizing"]),
    "panic": ("Anxiety", ["Internalizing"]),
    "agor": ("Anxiety", ["Internalizing"]),
    "phobia": ("Anxiety", ["Internalizing"]),
    "ocd": ("OCD", ["Internalizing"]),
    "ptsd": ("PTSD", ["Internalizing"]),
    "adhd": ("ADHD", ["Externalizing", "Neurodevelopmental"]),
    "odd": ("ODD", ["Externalizing"]),
    "cond": ("Conduct", ["Externalizing"]),
    "asd": ("Autism", ["Neurodevelopmental"]),
    "tic": ("Tic", ["Neurodevelopmental"]),
    "ed": ("Eating", ["Other"]),
    "psych": ("Psychosis", ["Other"]),
    "sleep": ("Sleep", ["Other"]),
    "suic": ("Suicidality", ["Other"]),
    "hom": ("Homicidality", ["Other"]),
}

FULL = ["present", "past", "partial_remission"]
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]


if __name__ == "__main__":
    #    main()

    # def main():
    cw = build_crosswalk()
    print(
        f"Crosswalk: {len(cw)} diagnosis variables -> {cw.category.nunique()} categories "
        f"({cw.is_subthreshold.sum()} subthreshold)."
    )

    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in [
        "session_id",
        "variable",
        "resolved",
        "informant",
        "module",
        "status_layer",
    ]:
        if c in resolved:
            resolved[c] = resolved[c].astype(str)
    resolved = resolved[resolved.session_id.isin(EVEN)].copy()

    CONFIGS = [
        ("current", False, "parent"),
        ("ever_met", False, "parent"),
        ("ever_met", True, "parent"),
        ("current", False, "youth"),
        ("ever_met", False, "either"),
        ("ever_met", False, "both"),
    ]
    CATS = ["Depression", "Anxiety", "ADHD", "ODD", "Conduct"]
    rows = []
    for status_set, subthr, inf in CONFIGS:
        if inf == "either":
            cp = build_caseness(
                resolved,
                cw,
                status_set=status_set,
                include_subthreshold=subthr,
                informant="parent",
            )
            cy = build_caseness(
                resolved,
                cw,
                status_set=status_set,
                include_subthreshold=subthr,
                informant="youth",
            )
            rank = {"positive": 3, "administered_negative": 2, "not_administered": 1}
            c = pd.concat([cp, cy])
            c["rk"] = c.status.map(rank)
            c = (
                c.groupby(["participant_id", "session_id", "category"])["rk"]
                .max()
                .reset_index()
            )
            c["status"] = c.rk.map(
                {3: "positive", 2: "administered_negative", 1: "not_administered"}
            )
        else:
            c = build_caseness(
                resolved,
                cw,
                status_set=status_set,
                include_subthreshold=subthr,
                informant=inf,
            )
        # ever-met prevalence across waves: positive at any even wave / administered at any wave
        for cat in CATS:
            cc = c[c.category == cat]
            ppl = cc.groupby("participant_id")["status"].agg(
                lambda s: (
                    "positive"
                    if (s == "positive").any()
                    else (
                        "administered_negative"
                        if (s == "administered_negative").any()
                        else "not_administered"
                    )
                )
            )
            n_admin = (ppl != "not_administered").sum()
            n_pos = (ppl == "positive").sum()
            rows.append(
                {
                    "status_set": status_set,
                    "subthreshold": subthr,
                    "informant": inf,
                    "category": cat,
                    "n_assessed": int(n_admin),
                    "n_positive": int(n_pos),
                    "prevalence_pct": round(100 * n_pos / n_admin, 2)
                    if n_admin
                    else np.nan,
                }
            )
    sens = pd.DataFrame(rows)
    sens.to_csv(config.DERIV / "ksads_caseness_sensitivity.csv", index=False)

    print("\nLifetime (any even wave) prevalence by operationalization:")
    piv = sens.pivot_table(
        index=["status_set", "subthreshold", "informant"],
        columns="category",
        values="prevalence_pct",
    )
    print(piv[CATS].to_string())
    print(f"\nWrote {config.DERIV.as_posix()}/ksads_category_crosswalk.csv")
    print(f"Wrote {config.DERIV.as_posix()}/ksads_caseness_sensitivity.csv")

#!/usr/bin/env python3
import csv
import numpy as np
import pandas as pd
from abcd_ksads import config

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


def build_crosswalk():
    rows = [
        r
        for r in csv.DictReader(open(config.KSADS_VARIABLE_MAP))
        if r["layer"] == "diagnosis"
    ]
    out = []
    for r in rows:
        cat, bands = CATEGORY.get(r["module"], ("Unmapped", []))
        lab = r["label"].lower()
        out.append(
            {
                "variable": r["variable"],
                "informant": r["informant"],
                "module": r["module"],
                "status_layer": r["status"],
                "category": cat,
                "broadband": "|".join(bands),
                "is_subthreshold": int(
                    "other specified" in lab or "unspecified" in lab
                ),
            }
        )
    cw = pd.DataFrame(out)
    cw.to_csv(config.DERIV / "ksads_category_crosswalk.csv", index=False)
    return cw


def build_caseness(
    resolved,
    crosswalk,
    *,
    status_set="ever_met",
    include_subthreshold=False,
    informant="parent",
):
    """participant x session x category caseness honoring resolved states.

    Returns long df: participant_id, session_id, category, status in
    {positive, administered_negative, not_administered}.
    """
    statuses = FULL if status_set == "ever_met" else ["present"]
    cw = crosswalk[crosswalk.status_layer.isin(statuses)].copy()
    if not include_subthreshold:
        cw = cw[cw.is_subthreshold == 0]
    if informant in ("parent", "youth"):
        cw = cw[cw.informant == informant]
    keep = set(cw.variable)

    r = resolved[resolved.variable.isin(keep)][
        ["participant_id", "session_id", "variable", "resolved"]
    ].merge(cw[["variable", "category", "informant"]], on="variable", how="inner")
    # rank resolved states so the max over constituents = category state
    rank = {
        "positive": 3,
        "administered_negative": 2,
        "not_administered": 1,
        "no_record": 0,
    }
    r["rk"] = r.resolved.map(rank)

    if informant == "both":
        # require positive on BOTH informants; collapse per informant first
        per = (
            r.groupby(["participant_id", "session_id", "category", "informant"])["rk"]
            .max()
            .reset_index()
        )
        piv = per.pivot_table(
            index=["participant_id", "session_id", "category"],
            columns="informant",
            values="rk",
            fill_value=0,
        )
        both_pos = (piv.get("parent", 0) == 3) & (piv.get("youth", 0) == 3)
        admin = piv.max(axis=1) >= 2
        st = np.where(
            both_pos,
            "positive",
            np.where(admin, "administered_negative", "not_administered"),
        )
        res = piv.reset_index()[["participant_id", "session_id", "category"]].copy()
        res["status"] = st
        return res

    g = (
        r.groupby(["participant_id", "session_id", "category"])["rk"]
        .max()
        .reset_index()
    )
    inv = {
        3: "positive",
        2: "administered_negative",
        1: "not_administered",
        0: "not_administered",
    }
    g["status"] = g["rk"].map(inv)
    return g[["participant_id", "session_id", "category", "status"]]

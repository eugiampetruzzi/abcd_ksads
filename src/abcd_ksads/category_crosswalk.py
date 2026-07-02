#!/usr/bin/env python3
import numpy as np
import pandas as pd
from abcd_ksads import config

# KSADS module -> DSM category
CATEGORY = {
    "dep": "Depression",
    "bpd": "Bipolar",
    "dmdd": "DMDD",
    "gad": "Anxiety",
    "sepanx": "Anxiety",
    "socanx": "Anxiety",
    "panic": "Anxiety",
    "agor": "Anxiety",
    "phobia": "Anxiety",
    "ocd": "OCD",
    "ptsd": "PTSD",
    "adhd": "ADHD",
    "odd": "ODD",
    "cond": "Conduct",
    "asd": "Autism",
    "tic": "Tic",
    "ed": "Eating",
    "psych": "Psychosis",
    "sleep": "Sleep",
    "suic": "Suicidality",
    "hom": "Homicidality",
}

COLUMNS = [
    "variable",
    "informant",
    "module",
    "status_layer",
    "category",
    "is_subthreshold",
]

FULL = ["present", "past", "partial_remission"]
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]


def build_crosswalk():
    cw = pd.read_csv(config.KSADS_VARIABLE_MAP)
    cw = cw[cw.layer == "diagnosis"].copy()
    cw["category"] = cw.module.map(CATEGORY)
    missing = sorted(cw.loc[cw.category.isna(), "module"].unique())
    if missing:
        raise ValueError(f"modules missing from CATEGORY: {missing}")
    label = cw.label.str.lower()
    cw["is_subthreshold"] = (
        label.str.contains("other specified", regex=False, na=False)
        | label.str.contains("unspecified", regex=False, na=False)
    ).astype(int)
    return cw.rename(columns={"status": "status_layer"})[COLUMNS]


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

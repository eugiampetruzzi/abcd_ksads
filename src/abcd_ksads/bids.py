"""Build the analysis-ready category caseness tables for the harmonized BIDS dataset.

The ``either`` informant is the recommended default: a category is positive if the
parent or youth is positive (max over the resolved-state rank). The
``12_build_bids_dataset.py`` script wires these to the TSVs and JSON sidecars.
"""

import pandas as pd

from abcd_ksads.category_crosswalk import build_caseness

RANK = {"positive": 3, "administered_negative": 2, "not_administered": 1}
INV = {3: "positive", 2: "administered_negative", 1: "not_administered"}
DISORDER_CATS = [
    "Depression", "Anxiety", "ADHD", "ODD", "Conduct", "Bipolar", "DMDD",
    "OCD", "PTSD", "Autism", "Tic", "Eating", "Psychosis",
]


def combine_either(cp, cy):
    """Combine parent and youth caseness: category positive if either informant is.

    Takes the max resolved-state rank per participant x session x category."""
    c = pd.concat([cp, cy])
    c = c.assign(rk=c.status.map(RANK))
    c = c.groupby(["participant_id", "session_id", "category"])["rk"].max().reset_index()
    c["status"] = c.rk.map(INV)
    return c[["participant_id", "session_id", "category", "status"]]


def caseness_informant(base, cw, status_set, informant):
    """Category caseness for one informant, where 'either' = parent-or-youth positive."""
    if informant == "either":
        cp = build_caseness(base, cw, status_set=status_set,
                            include_subthreshold=False, informant="parent")
        cy = build_caseness(base, cw, status_set=status_set,
                            include_subthreshold=False, informant="youth")
        return combine_either(cp, cy)
    return build_caseness(base, cw, status_set=status_set,
                          include_subthreshold=False, informant=informant)


def caseness_wide(base, cw, status_set, disorder_cats=DISORDER_CATS,
                  informants=("parent", "youth", "either")):
    """Wide caseness (one column per disorder category) stacked over informants.

    Categories a participant was not assessed on are filled with not_administered."""
    frames = []
    for inf in informants:
        c = caseness_informant(base, cw, status_set, inf)
        c = c[c.category.isin(disorder_cats)]
        wide = c.pivot_table(
            index=["participant_id", "session_id"],
            columns="category",
            values="status",
            aggfunc="first",
        ).reset_index()
        for cat in disorder_cats:
            if cat not in wide:
                wide[cat] = "not_administered"
            wide[cat] = wide[cat].fillna("not_administered")
        wide.insert(2, "informant", inf)
        frames.append(wide[["participant_id", "session_id", "informant"] + list(disorder_cats)])
    return pd.concat(frames, ignore_index=True)


def count_admin_waves(res):
    """Per participant, the number of waves with an administered parent/youth KSADS."""
    adm = res[res.resolved.isin(["positive", "administered_negative"])]
    nwav = (
        adm.groupby(["participant_id", "informant"])["session_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("parent", "youth"):
        if col not in nwav:
            nwav[col] = 0
    parts = nwav.rename(
        columns={"parent": "n_waves_parent_kSADS", "youth": "n_waves_youth_kSADS"}
    )
    return parts[["participant_id", "n_waves_parent_kSADS", "n_waves_youth_kSADS"]]

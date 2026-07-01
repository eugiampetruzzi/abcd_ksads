#!/usr/bin/env python3
import numpy as np
import pandas as pd

from abcd_ksads.category_crosswalk import build_caseness


CATS_FOR = {
    "depression": ["Depression"],
    "anxiety": ["Anxiety"],
    "externalizing": ["ADHD", "ODD", "Conduct"],
    "ADHD": ["ADHD"],
    "ODD": ["ODD"],
    "conduct": ["Conduct"],
    "eating": ["Eating"],
    "suicidality": ["Suicidality"],
    "any-disorder": [
        "Depression",
        "Anxiety",
        "ADHD",
        "ODD",
        "Conduct",
        "Bipolar",
        "DMDD",
        "OCD",
        "PTSD",
        "Autism",
        "Tic",
        "Eating",
        "Psychosis",
    ],
}
RANK = {"positive": 3, "administered_negative": 2, "not_administered": 1}
INV = {3: "positive", 2: "administered_negative", 1: "not_administered"}
BASE_SES = "ses-00A"


def _phobia_crosswalk(cw, phobia):
    if phobia == "phobia_out":
        return cw[cw.module != "phobia"].copy()
    return cw


def build_primitive_cache(base, cw):
    """Precompute parent and youth caseness for each (status, threshold, phobia)."""
    cache = {}
    for status_set in ("current", "ever_met"):
        for subthr in (False, True):
            for phobia in ("phobia_in", "phobia_out"):
                cwp = _phobia_crosswalk(cw, phobia)
                cache[(status_set, subthr, phobia)] = {
                    inf: build_caseness(
                        base,
                        cwp,
                        status_set=status_set,
                        include_subthreshold=subthr,
                        informant=inf,
                    )
                    for inf in ("parent", "youth")
                }
    return cache


def _agg(cobj, cats):
    c = cobj[cobj.category.isin(cats)]
    if c.empty:
        return pd.Series(dtype=object)
    rk = c.assign(rk=c.status.map(RANK)).groupby("participant_id")["rk"].max()
    return rk.map(INV)


def construct_status(cache, construct, status_set, informant, subthr, phobia):
    prim = cache[(status_set, subthr, phobia)]
    cats = CATS_FOR[construct]
    ps = _agg(prim["parent"], cats)
    ys = _agg(prim["youth"], cats)
    if informant == "parent":
        return ps
    if informant == "youth":
        return ys
    df = pd.concat([ps.map(RANK).rename("p"), ys.map(RANK).rename("y")], axis=1)
    if informant == "either":
        return df.max(axis=1).map(INV)
    # both: positive only if both informants positive; administered if either; else none
    both = np.where((df.p == 3) & (df.y == 3), 3, np.where(df.max(axis=1) >= 2, 2, 1))
    return pd.Series(both, index=df.index).map(INV)


def prevalence(stat):
    if stat is None or len(stat) == 0:
        return np.nan, 0, 0
    n_den = int((stat != "not_administered").sum())
    n_num = int((stat == "positive").sum())
    return (100 * n_num / n_den if n_den else np.nan), n_num, n_den


def informant_validity(cw, cal):
    """parent/youth module availability per construct at baseline."""
    adm = cal[(cal.session_id == BASE_SES) & (cal.status == "administered")]
    adm_p = set(adm[adm.informant == "parent"].module)
    adm_y = set(adm[adm.informant == "youth"].module)
    valid = {}
    for con, cats in CATS_FOR.items():
        mods_p = set(cw[(cw.category.isin(cats)) & (cw.informant == "parent")].module)
        mods_y = set(cw[(cw.category.isin(cats)) & (cw.informant == "youth")].module)
        vp = bool(mods_p & adm_p)
        vy = bool(mods_y & adm_y)
        valid[con] = {"parent": vp, "youth": vy, "either": vp or vy, "both": vp and vy}
    return valid

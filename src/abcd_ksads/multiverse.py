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


TINY = 0.1  # min-prevalence floor below which a fold-range is flagged unstable


def build_multiverse_grid(cache, valid, cats_for=CATS_FOR):
    """Enumerate every valid operationalization and its baseline prevalence.

    Varies construct x timeframe x informant x threshold x (phobia, anxiety only);
    skips informant/construct pairs whose modules were never administered, and specs
    with an empty administered denominator. Returns ``(grid, n_skipped)``."""
    rows, skipped, sid = [], 0, 0
    for con in cats_for:
        phobia_levels = ("phobia_in", "phobia_out") if con == "anxiety" else ("phobia_in",)
        for status_set in ("current", "ever_met"):
            for informant in ("parent", "youth", "either", "both"):
                if not valid[con][informant]:
                    skipped += 1
                    continue
                for subthr in (False, True):
                    for phobia in phobia_levels:
                        stat = construct_status(
                            cache, con, status_set, informant, subthr, phobia
                        )
                        prev, num, den = prevalence(stat)
                        if den == 0:
                            skipped += 1
                            continue
                        sid += 1
                        rows.append(
                            {
                                "construct": con,
                                "status": status_set,
                                "informant": informant,
                                "threshold": "with_subthreshold" if subthr else "full",
                                "phobia": phobia,
                                "window": "single_wave_baseline",
                                "spec_id": sid,
                                "prevalence_pct": round(prev, 3),
                                "n_numerator": num,
                                "n_denominator": den,
                            }
                        )
    return pd.DataFrame(rows), skipped


def summarize_multiverse(grid, tiny=TINY):
    """Per-construct spread of prevalence across specifications (fold-range et al.)."""
    rows = []
    for con, sub in grid.groupby("construct"):
        p = sub.prevalence_pct
        lo, hi = p.min(), p.max()
        fold = hi / lo if lo > 0 else np.inf
        rows.append(
            {
                "construct": con,
                "n_specs": len(sub),
                "prev_min": round(lo, 3),
                "prev_max": round(hi, 3),
                "fold_range": round(fold, 1) if np.isfinite(fold) else np.inf,
                "pp_span": round(hi - lo, 2),
                "prev_median": round(p.median(), 3),
                "prev_iqr_low": round(p.quantile(0.25), 3),
                "prev_iqr_high": round(p.quantile(0.75), 3),
                "unstable_fold": bool(lo < tiny),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_range", ascending=False)


DEFAULT_LEVER_BASE = dict(status="current", informant="parent", subthr=False, phobia="phobia_in")
DEFAULT_LEVERS = [
    ("current -> ever-met", dict(status="ever_met")),
    ("parent -> youth-only", dict(informant="youth")),
    ("parent -> either", dict(informant="either")),
    ("+ subthreshold dx", dict(subthr=True)),
    ("anxiety: drop phobia", dict(phobia="phobia_out")),
]


def single_lever_table(cache, construct="any-disorder", base_cfg=None, flips=None):
    """One-decision-at-a-time prevalence shifts from a base operationalization.

    Flips each lever singly from ``base_cfg`` and reports the prevalence delta,
    ordered by absolute effect. Returns a DataFrame (base row first)."""
    base_cfg = base_cfg or DEFAULT_LEVER_BASE
    flips = flips or DEFAULT_LEVERS

    def prev(status, informant, subthr, phobia):
        stat = construct_status(cache, construct, status, informant, subthr, phobia)
        return prevalence(stat)[0]

    base_prev = prev(**base_cfg)
    rows = [
        {
            "lever": "base (current, parent, full, phobia-in)",
            "prevalence_pct": round(base_prev, 3),
            "delta_pts": 0.0,
        }
    ]
    for name, override in flips:
        p = prev(**dict(base_cfg, **override))
        rows.append(
            {
                "lever": name,
                "prevalence_pct": round(p, 3),
                "delta_pts": round(p - base_prev, 3),
            }
        )
    df = pd.DataFrame(rows)
    body = df.iloc[1:].reindex(df.iloc[1:].delta_pts.abs().sort_values(ascending=False).index)
    return pd.concat([df.iloc[[0]], body], ignore_index=True)

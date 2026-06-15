#!/usr/bin/env python3
"""Multiverse specification grid (release 7.0, self-contained).

Crosses every defensible operationalization axis and computes baseline prevalence
for each construct over the administered denominator only (positive +
administered_negative; not_administered cells are excluded from numerator and
denominator). Caseness comes from the Layer 3 engine; validity of informant
levels is checked against the Layer 2 administration calendar.

Axes: status {current, ever_met}; informant {parent, youth, either, both};
threshold {full, with_subthreshold}; phobia {phobia_in, phobia_out} (anxiety
only); window held at baseline (ses-00A).

Output: derivatives/multiverse_grid.csv
Importable helpers (cache, construct_status, prevalence) are reused by 07-10.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")


def _load(f):
    spec = importlib.util.spec_from_file_location(f[:-3].replace(".", "_"),
                                                  os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


L3 = _load("03_category_crosswalk.py")

CATS_FOR = {
    "depression":    ["Depression"],
    "anxiety":       ["Anxiety"],
    "externalizing": ["ADHD", "ODD", "Conduct"],
    "ADHD":          ["ADHD"],
    "ODD":           ["ODD"],
    "conduct":       ["Conduct"],
    "any-disorder":  ["Depression", "Anxiety", "ADHD", "ODD", "Conduct", "Bipolar",
                      "DMDD", "OCD", "PTSD", "Autism", "Tic", "Eating", "Psychosis"],
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
                    inf: L3.build_caseness(base, cwp, status_set=status_set,
                                           include_subthreshold=subthr, informant=inf)
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
    both = np.where((df.p == 3) & (df.y == 3), 3,
                    np.where(df.max(axis=1) >= 2, 2, 1))
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
        valid[con] = {"parent": vp, "youth": vy,
                      "either": vp or vy, "both": vp and vy}
    return valid


def main():
    cw = L3.build_crosswalk()
    cal = pd.read_csv(os.path.join(DERIV, "ksads_administration_calendar.csv"))
    resolved = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()

    cache = build_primitive_cache(base, cw)
    valid = informant_validity(cw, cal)

    rows, skipped, sid = [], 0, 0
    for con in CATS_FOR:
        phobia_levels = ("phobia_in", "phobia_out") if con == "anxiety" else ("phobia_in",)
        for status_set in ("current", "ever_met"):
            for informant in ("parent", "youth", "either", "both"):
                if not valid[con][informant]:
                    skipped += 1
                    continue
                for subthr in (False, True):
                    for phobia in phobia_levels:
                        stat = construct_status(cache, con, status_set,
                                                informant, subthr, phobia)
                        prev, num, den = prevalence(stat)
                        if den == 0:
                            skipped += 1
                            continue
                        sid += 1
                        rows.append({
                            "construct": con, "status": status_set,
                            "informant": informant,
                            "threshold": "with_subthreshold" if subthr else "full",
                            "phobia": phobia, "window": "single_wave_baseline",
                            "spec_id": sid, "prevalence_pct": round(prev, 3),
                            "n_numerator": num, "n_denominator": den})
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(DERIV, "multiverse_grid.csv"), index=False)

    print(f"Enumerated {len(grid)} valid specifications "
          f"({skipped} impossible cells skipped).\n")
    hdr = f"{'construct':14} {'n':>4} {'min%':>7} {'max%':>7} {'fold':>6} {'median%':>8} {'IQR%':>14}"
    print(hdr); print("-" * len(hdr))
    for con, sub in grid.groupby("construct"):
        p = sub.prevalence_pct
        q1, q3 = p.quantile(.25), p.quantile(.75)
        fold = p.max() / p.min() if p.min() > 0 else np.inf
        print(f"{con:14} {len(sub):>4} {p.min():>7.2f} {p.max():>7.2f} "
              f"{fold:>6.1f} {p.median():>8.2f} {q1:>6.2f}-{q3:<7.2f}")
    print(f"\nWrote {DERIV}/multiverse_grid.csv")


if __name__ == "__main__":
    main()

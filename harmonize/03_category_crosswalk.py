#!/usr/bin/env python3
"""Pipeline layer 3: diagnosis -> DSM-category crosswalk + toggle engine.

Maps the 230 diagnosis variables to DSM categories and broadband dimensions, then
materializes participant x wave x category caseness under explicit, configurable
rules. The same resolved states from layer 1 are carried through, so a category is
only "negative" when its constituent diagnoses were actually administered; if none
were administered the category is not_administered (never silently 0).

Toggles (every prior ABCD paper makes these choices implicitly):
    status_set            current  -> {present}
                          ever_met -> {present, past, partial_remission}
    include_subthreshold  include / exclude "other specified" + "unspecified" dx
    informant             parent / youth / either (positive if either) / both

Outputs:
    derivatives/ksads_category_crosswalk.csv   variable -> category map
    derivatives/ksads_caseness_sensitivity.csv prevalence under toggle settings
    build_caseness(...) is importable for downstream analytic datasets.
"""
import csv
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
MAP = os.path.join(os.path.dirname(HERE), "codebooks", "ksads_variable_map.csv")

# module -> (DSM category, [broadband dimensions])
CATEGORY = {
    "dep":    ("Depression",        ["Internalizing", "Mood"]),
    "bpd":    ("Bipolar",           ["Mood"]),
    "dmdd":   ("DMDD",              ["Mood"]),
    "gad":    ("Anxiety",           ["Internalizing"]),
    "sepanx": ("Anxiety",           ["Internalizing"]),
    "socanx": ("Anxiety",           ["Internalizing"]),
    "panic":  ("Anxiety",           ["Internalizing"]),
    "agor":   ("Anxiety",           ["Internalizing"]),
    "phobia": ("Anxiety",           ["Internalizing"]),
    "ocd":    ("OCD",               ["Internalizing"]),
    "ptsd":   ("PTSD",              ["Internalizing"]),
    "adhd":   ("ADHD",              ["Externalizing", "Neurodevelopmental"]),
    "odd":    ("ODD",               ["Externalizing"]),
    "cond":   ("Conduct",           ["Externalizing"]),
    "asd":    ("Autism",            ["Neurodevelopmental"]),
    "tic":    ("Tic",               ["Neurodevelopmental"]),
    "ed":     ("Eating",            ["Other"]),
    "psych":  ("Psychosis",         ["Other"]),
    "sleep":  ("Sleep",             ["Other"]),
    "suic":   ("Suicidality",       ["Other"]),
    "hom":    ("Homicidality",      ["Other"]),
}

FULL = ["present", "past", "partial_remission"]
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]


def build_crosswalk():
    rows = [r for r in csv.DictReader(open(MAP)) if r["layer"] == "diagnosis"]
    out = []
    for r in rows:
        cat, bands = CATEGORY.get(r["module"], ("Unmapped", []))
        lab = r["label"].lower()
        out.append({
            "variable": r["variable"], "informant": r["informant"],
            "module": r["module"], "status_layer": r["status"],
            "category": cat, "broadband": "|".join(bands),
            "is_subthreshold": int("other specified" in lab or "unspecified" in lab),
        })
    cw = pd.DataFrame(out)
    cw.to_csv(os.path.join(DERIV, "ksads_category_crosswalk.csv"), index=False)
    return cw


def build_caseness(resolved, crosswalk, *, status_set="ever_met",
                   include_subthreshold=False, informant="parent"):
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

    r = (resolved[resolved.variable.isin(keep)]
         [["participant_id", "session_id", "variable", "resolved"]]
         .merge(cw[["variable", "category", "informant"]], on="variable", how="inner"))
    # rank resolved states so the max over constituents = category state
    rank = {"positive": 3, "administered_negative": 2, "not_administered": 1, "no_record": 0}
    r["rk"] = r.resolved.map(rank)

    if informant == "both":
        # require positive on BOTH informants; collapse per informant first
        per = (r.groupby(["participant_id", "session_id", "category", "informant"])
                 ["rk"].max().reset_index())
        piv = per.pivot_table(index=["participant_id", "session_id", "category"],
                              columns="informant", values="rk", fill_value=0)
        both_pos = (piv.get("parent", 0) == 3) & (piv.get("youth", 0) == 3)
        admin = (piv.max(axis=1) >= 2)
        st = np.where(both_pos, "positive",
                      np.where(admin, "administered_negative", "not_administered"))
        res = piv.reset_index()[["participant_id", "session_id", "category"]].copy()
        res["status"] = st
        return res

    g = (r.groupby(["participant_id", "session_id", "category"])["rk"]
           .max().reset_index())
    inv = {3: "positive", 2: "administered_negative", 1: "not_administered", 0: "not_administered"}
    g["status"] = g["rk"].map(inv)
    return g[["participant_id", "session_id", "category", "status"]]


def main():
    cw = build_crosswalk()
    print(f"Crosswalk: {len(cw)} diagnosis variables -> {cw.category.nunique()} categories "
          f"({cw.is_subthreshold.sum()} subthreshold).")

    resolved = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "variable", "resolved", "informant", "module", "status_layer"]:
        if c in resolved:
            resolved[c] = resolved[c].astype(str)
    resolved = resolved[resolved.session_id.isin(EVEN)].copy()

    CONFIGS = [
        ("current",  False, "parent"),
        ("ever_met", False, "parent"),
        ("ever_met", True,  "parent"),
        ("current",  False, "youth"),
        ("ever_met", False, "either"),
        ("ever_met", False, "both"),
    ]
    CATS = ["Depression", "Anxiety", "ADHD", "ODD", "Conduct"]
    rows = []
    for status_set, subthr, inf in CONFIGS:
        if inf == "either":
            cp = build_caseness(resolved, cw, status_set=status_set,
                                include_subthreshold=subthr, informant="parent")
            cy = build_caseness(resolved, cw, status_set=status_set,
                                include_subthreshold=subthr, informant="youth")
            rank = {"positive": 3, "administered_negative": 2, "not_administered": 1}
            c = pd.concat([cp, cy]); c["rk"] = c.status.map(rank)
            c = c.groupby(["participant_id", "session_id", "category"])["rk"].max().reset_index()
            c["status"] = c.rk.map({3: "positive", 2: "administered_negative", 1: "not_administered"})
        else:
            c = build_caseness(resolved, cw, status_set=status_set,
                               include_subthreshold=subthr, informant=inf)
        # ever-met prevalence across waves: positive at any even wave / administered at any wave
        for cat in CATS:
            cc = c[c.category == cat]
            ppl = cc.groupby("participant_id")["status"].agg(
                lambda s: "positive" if (s == "positive").any()
                else ("administered_negative" if (s == "administered_negative").any()
                      else "not_administered"))
            n_admin = (ppl != "not_administered").sum()
            n_pos = (ppl == "positive").sum()
            rows.append({
                "status_set": status_set, "subthreshold": subthr, "informant": inf,
                "category": cat, "n_assessed": int(n_admin), "n_positive": int(n_pos),
                "prevalence_pct": round(100 * n_pos / n_admin, 2) if n_admin else np.nan,
            })
    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(DERIV, "ksads_caseness_sensitivity.csv"), index=False)

    print("\nLifetime (any even wave) prevalence by operationalization:")
    piv = sens.pivot_table(index=["status_set", "subthreshold", "informant"],
                           columns="category", values="prevalence_pct")
    print(piv[CATS].to_string())
    print(f"\nWrote {DERIV}/ksads_category_crosswalk.csv")
    print(f"Wrote {DERIV}/ksads_caseness_sensitivity.csv")


if __name__ == "__main__":
    main()

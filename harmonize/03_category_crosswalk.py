import csv
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, "codebooks", "ksads_variable_map.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivatives")
os.makedirs(OUT, exist_ok=True)

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
RANK = {"positive": 3, "administered_negative": 2, "not_administered": 1, "no_record": 0}
INV = {3: "positive", 2: "administered_negative", 1: "not_administered", 0: "not_administered"}


def build_crosswalk():
    rows = [r for r in csv.DictReader(open(MAP)) if r["layer"] == "diagnosis"]
    out = []
    for r in rows:
        cat, bands = CATEGORY.get(r["module"], ("Unmapped", []))
        lab = r["label"].lower()
        out.append({"variable": r["variable"], "informant": r["informant"],
                    "module": r["module"], "status_layer": r["status"],
                    "category": cat, "broadband": "|".join(bands),
                    "is_subthreshold": int("other specified" in lab or "unspecified" in lab)})
    return pd.DataFrame(out)


def build_caseness(resolved, cw, status_set="ever_met",
                   include_subthreshold=False, informant="parent"):
    c = cw[cw.status_layer.isin(FULL if status_set == "ever_met" else ["present"])]
    if not include_subthreshold:
        c = c[c.is_subthreshold == 0]
    if informant in ("parent", "youth"):
        c = c[c.informant == informant]
    r = (resolved[resolved.variable.isin(set(c.variable))]
         [["participant_id", "session_id", "variable", "resolved"]]
         .merge(c[["variable", "category", "informant"]], on="variable"))
    r["rk"] = r.resolved.map(RANK)
    if informant == "both":
        per = r.groupby(["participant_id", "session_id", "category", "informant"])["rk"].max().reset_index()
        piv = per.pivot_table(index=["participant_id", "session_id", "category"],
                              columns="informant", values="rk", fill_value=0)
        both = (piv.get("parent", 0) == 3) & (piv.get("youth", 0) == 3)
        adm = piv.max(axis=1) >= 2
        res = piv.reset_index()[["participant_id", "session_id", "category"]]
        res["status"] = np.where(both, "positive",
                                 np.where(adm, "administered_negative", "not_administered"))
        return res
    g = r.groupby(["participant_id", "session_id", "category"])["rk"].max().reset_index()
    g["status"] = g.rk.map(INV)
    return g[["participant_id", "session_id", "category", "status"]]


if __name__ == "__main__":
    build_crosswalk().to_csv(os.path.join(OUT, "ksads_category_crosswalk.csv"), index=False)

#!/usr/bin/env python3
"""01_resolve_missingness.py — Pipeline layer 1: canonical missingness resolver.

The ABCD 7.0 KSADS-COMP release stores every diagnosis as 0 / 1 / 555 / blank.
These encode four semantically distinct states that are routinely collapsed to a
0/1 "caseness" flag, silently converting not-administered records into healthy
controls. This module resolves every participant x session x diagnosis cell into
an explicit status, the foundation every downstream layer builds on.

Resolved status (diagnosis layer):
    positive              value == 1   (met criteria)
    administered_negative value == 0   (assessed, criteria not met)
    not_administered      value == 555 (module/wave not given to this participant)
    no_record             value blank/NaN, or no row for that participant x session

(888 = branch-skipped exists only in the ITEM layer, not the diagnosis layer, so
it is not produced here; it is handled by the symptom-layer resolver.)

Inputs:
    rawdata/phenotype/mh_{p,y}_ksads__<module>.tsv
    abcd_ksads/outputs/ksads_variable_map.csv   (diagnosis-layer rows)

Outputs:
    harmonize/derivatives/ksads_resolved_long.parquet
        participant_id, session_id, informant, module, status_layer, variable,
        disorder, resolved   (resolved is an ordered categorical)
    harmonize/derivatives/ksads_resolution_summary.csv
        per variable x session: n_positive, n_admin_neg, n_not_admin, n_no_record
"""
import csv
import os

import numpy as np
import pandas as pd

KS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(KS, "rawdata", "phenotype")
MAP = os.path.join(KS, "abcd_ksads", "codebooks", "ksads_variable_map.csv")
DERIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivatives")
os.makedirs(DERIV, exist_ok=True)

SESSIONS = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
            "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
RESOLVED = ["positive", "administered_negative", "not_administered", "no_record"]
VALUE_MAP = {"1": "positive", "0": "administered_negative", "555": "not_administered"}


def load_diagnosis_vars():
    rows = [r for r in csv.DictReader(open(MAP)) if r["layer"] == "diagnosis"]
    by_file = {}
    for r in rows:
        pref = "mh_p" if r["informant"] == "parent" else "mh_y"
        fname = f"{pref}_ksads__{r['module']}.tsv"
        by_file.setdefault(fname, []).append(r)
    return rows, by_file


def resolve():
    rows, by_file = load_diagnosis_vars()
    meta = {r["variable"]: r for r in rows}
    long_parts = []
    summary_rows = []

    for fname, vrows in sorted(by_file.items()):
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            continue
        wanted = [r["variable"] for r in vrows]
        avail = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        use = [v for v in wanted if v in avail]
        if not use:
            continue
        df = pd.read_csv(path, sep="\t", usecols=["participant_id", "session_id"] + use,
                         dtype=str)
        df = df[df["session_id"].isin(SESSIONS)]

        long = df.melt(id_vars=["participant_id", "session_id"], value_vars=use,
                       var_name="variable", value_name="raw")
        long["resolved"] = long["raw"].map(VALUE_MAP).fillna("no_record")
        m = long["variable"].map(meta)
        long["informant"] = long["variable"].map(lambda v: meta[v]["informant"])
        long["module"] = long["variable"].map(lambda v: meta[v]["module"])
        long["status_layer"] = long["variable"].map(lambda v: meta[v]["status"])
        long["disorder"] = long["variable"].map(
            lambda v: meta[v]["disorder"] or meta[v]["label"])
        long = long.drop(columns="raw")
        long_parts.append(long)

        # summary per variable x session
        g = (long.groupby(["variable", "session_id", "resolved"]).size()
                  .unstack("resolved", fill_value=0).reset_index())
        for col in RESOLVED:
            if col not in g:
                g[col] = 0
        for _, rr in g.iterrows():
            summary_rows.append({
                "variable": rr["variable"], "session_id": rr["session_id"],
                "informant": meta[rr["variable"]]["informant"],
                "module": meta[rr["variable"]]["module"],
                "status_layer": meta[rr["variable"]]["status"],
                "n_positive": int(rr["positive"]),
                "n_administered_negative": int(rr["administered_negative"]),
                "n_not_administered": int(rr["not_administered"]),
                "n_no_record": int(rr["no_record"]),
            })

    long = pd.concat(long_parts, ignore_index=True)
    long["resolved"] = pd.Categorical(long["resolved"], categories=RESOLVED, ordered=True)
    for c in ["session_id", "informant", "module", "status_layer", "variable"]:
        long[c] = long[c].astype("category")
    out_long = os.path.join(DERIV, "ksads_resolved_long.parquet")
    long.to_parquet(out_long, index=False)

    summ = pd.DataFrame(summary_rows).sort_values(["informant", "module", "variable", "session_id"])
    out_summ = os.path.join(DERIV, "ksads_resolution_summary.csv")
    summ.to_csv(out_summ, index=False)

    # report
    tot = long["resolved"].value_counts()
    n = len(long)
    print(f"Resolved {n:,} participant x session x diagnosis cells "
          f"({long['variable'].nunique()} diagnosis variables, {len(SESSIONS)} sessions).")
    for s in RESOLVED:
        print(f"  {s:24} {int(tot.get(s,0)):>12,}  ({100*tot.get(s,0)/n:5.1f}%)")
    print(f"\nWrote {out_long}")
    print(f"Wrote {out_summ}")
    return long


if __name__ == "__main__":
    resolve()

import csv
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
       "ABCD 7.0/KSADS/rawdata/phenotype")
MAP = os.path.join(ROOT, "codebooks", "ksads_variable_map.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivatives")
os.makedirs(OUT, exist_ok=True)

SESSIONS = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
            "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
VALUE = {"1": "positive", "0": "administered_negative", "555": "not_administered"}

rows = [r for r in csv.DictReader(open(MAP)) if r["layer"] == "diagnosis"]
meta = {r["variable"]: r for r in rows}
by_file = {}
for r in rows:
    pref = "mh_p" if r["informant"] == "parent" else "mh_y"
    by_file.setdefault(f"{pref}_ksads__{r['module']}.tsv", []).append(r["variable"])

parts = []
for fname, vrows in sorted(by_file.items()):
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        continue
    avail = pd.read_csv(path, sep="\t", nrows=0).columns
    use = [v for v in vrows if v in avail]
    if not use:
        continue
    df = pd.read_csv(path, sep="\t", usecols=["participant_id", "session_id"] + use, dtype=str)
    df = df[df["session_id"].isin(SESSIONS)]
    long = df.melt(["participant_id", "session_id"], use, "variable", "raw")
    long["resolved"] = long["raw"].map(VALUE).fillna("no_record")
    long["informant"] = long["variable"].map(lambda v: meta[v]["informant"])
    long["module"] = long["variable"].map(lambda v: meta[v]["module"])
    long["status_layer"] = long["variable"].map(lambda v: meta[v]["status"])
    parts.append(long.drop(columns="raw"))

out = pd.concat(parts, ignore_index=True)
for c in ["session_id", "informant", "module", "status_layer", "variable", "resolved"]:
    out[c] = out[c].astype("category")
out.to_parquet(os.path.join(OUT, "ksads_resolved_long.parquet"), index=False)

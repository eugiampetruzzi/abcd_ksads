#!/usr/bin/env python3
import csv
import os
import dotenv
import pandas as pd

from abcd_ksads import config


dotenv.load_dotenv()

SESSIONS = [
    "ses-00A",
    "ses-01A",
    "ses-02A",
    "ses-03A",
    "ses-04A",
    "ses-05A",
    "ses-06A",
    "ses-07A",
]


def load_diagnosis_vars(config):
    rows = [
        r
        for r in csv.DictReader(open(config.KSADS_VARIABLE_MAP))
        if r["layer"] == "diagnosis"
    ]
    file_metadata = {}
    for r in rows:
        pref = "mh_p" if r["informant"] == "parent" else "mh_y"
        fname = f"{pref}_ksads__{r['module']}.tsv"
        file_metadata.setdefault(fname, []).append(r)
    return rows, file_metadata


def load_df_with_wanted_variables(path, file_variable_metadata):
    wanted = [r["variable"] for r in file_variable_metadata]
    avail = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    vars_to_use = [v for v in wanted if v in avail]
    if not vars_to_use:
        return None
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["participant_id", "session_id"] + vars_to_use,
    )
    return df[df["session_id"].isin(SESSIONS)], vars_to_use


def resolve():
    RAW = config.RAW_PHENOTYPE
    DERIV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derivatives")
    os.makedirs(DERIV, exist_ok=True)

    RESOLVED = ["positive", "administered_negative", "not_administered", "no_record"]
    VALUE_MAP = {
        "1": "positive",
        "0": "administered_negative",
        "555": "not_administered",
        1: "positive",
        0: "administered_negative",
        555: "not_administered",
    }

    rows, file_metadata = load_diagnosis_vars(config)
    all_var_metadata = {r["variable"]: r for r in rows}
    long_parts = []
    summary_rows = []

    for fname, file_variable_metadata in sorted(file_metadata.items()):
        path = os.path.join(RAW, fname)
        if not os.path.exists(path):
            print(f"{fname} not found, skipping")
            continue

        df, vars_to_use = load_df_with_wanted_variables(path, file_variable_metadata)

        long = df.melt(
            id_vars=["participant_id", "session_id"],
            value_vars=vars_to_use,
            var_name="variable",
            value_name="raw",
        )
        long["resolved"] = long["raw"].map(VALUE_MAP).fillna("no_record")
        # m = long["variable"].map(all_var_metadata)
        long["informant"] = long["variable"].map(
            lambda v: all_var_metadata[v]["informant"]
        )
        long["module"] = long["variable"].map(lambda v: all_var_metadata[v]["module"])
        long["status_layer"] = long["variable"].map(
            lambda v: all_var_metadata[v]["status"]
        )
        long["disorder"] = long["variable"].map(
            lambda v: all_var_metadata[v]["disorder"] or all_var_metadata[v]["label"]
        )
        long = long.drop(columns="raw")
        long_parts.append(long)

        # summary per variable x session
        g = (
            long.groupby(["variable", "session_id", "resolved"])
            .size()
            .unstack("resolved", fill_value=0)
            .reset_index()
        )
        for col in RESOLVED:
            if col not in g:
                g[col] = 0
        for _, rr in g.iterrows():
            summary_rows.append(
                {
                    "variable": rr["variable"],
                    "session_id": rr["session_id"],
                    "informant": all_var_metadata[rr["variable"]]["informant"],
                    "module": all_var_metadata[rr["variable"]]["module"],
                    "status_layer": all_var_metadata[rr["variable"]]["status"],
                    "n_positive": int(rr["positive"]),
                    "n_administered_negative": int(rr["administered_negative"]),
                    "n_not_administered": int(rr["not_administered"]),
                    "n_no_record": int(rr["no_record"]),
                }
            )

    long = pd.concat(long_parts, ignore_index=True)
    long["resolved"] = pd.Categorical(
        long["resolved"], categories=RESOLVED, ordered=True
    )
    for c in ["session_id", "informant", "module", "status_layer", "variable"]:
        long[c] = long[c].astype("category")
    out_long = os.path.join(DERIV, "ksads_resolved_long.parquet")
    long.to_parquet(out_long, index=False)

    summ = pd.DataFrame(summary_rows).sort_values(
        ["informant", "module", "variable", "session_id"]
    )
    out_summ = os.path.join(DERIV, "ksads_resolution_summary.csv")
    summ.to_csv(out_summ, index=False)

    # report
    tot = long["resolved"].value_counts()
    n = len(long)
    print(
        f"Resolved {n:,} participant x session x diagnosis cells "
        f"({long['variable'].nunique()} diagnosis variables, {len(SESSIONS)} sessions)."
    )
    for s in RESOLVED:
        print(f"  {s:24} {int(tot.get(s, 0)):>12,}  ({100 * tot.get(s, 0) / n:5.1f}%)")
    print(f"\nWrote {out_long}")
    print(f"Wrote {out_summ}")
    # return long


if __name__ == "__main__":
    resolve()

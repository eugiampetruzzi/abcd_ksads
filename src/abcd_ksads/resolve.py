"""Resolve KSADS diagnosis cells into missingness states.

Operates on the consolidated wide cache (``phenotype.parquet``): each diagnosis
variable is a column whose values are the exact source strings. A value is resolved
to one of four states; cells that are absent from a variable's source table appear as
``NaN`` in the cache and are dropped (they were never rows in the original per-file
output), while present-but-blank cells (``""``) resolve to ``no_record``.
"""

import csv
from pathlib import Path

import pandas as pd

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
RESOLVED = ["positive", "administered_negative", "not_administered", "no_record"]
# Codes are stored as float-formatted strings ("0.0", "1.0", "555.0"); coercing to
# numeric lets these integer keys match (1.0 == 1).
VALUE_MAP = {1: "positive", 0: "administered_negative", 555: "not_administered"}

_IDS = ["participant_id", "session_id"]


def load_diagnosis_metadata(map_path: Path) -> dict:
    """Return ``{variable: row}`` for the diagnosis-layer rows of the variable map."""
    rows = [r for r in csv.DictReader(open(map_path)) if r["layer"] == "diagnosis"]
    return {r["variable"]: r for r in rows}


def resolve_values(raw: pd.Series) -> pd.Series:
    """Map raw diagnosis values to resolved states; anything unrecognized is no_record."""
    numeric = pd.to_numeric(raw, errors="coerce")
    return numeric.map(VALUE_MAP).fillna("no_record")


def resolve_wide(wide: pd.DataFrame, var_metadata: dict, sessions=SESSIONS) -> pd.DataFrame:
    """Melt the diagnosis columns of a wide cache slice into resolved long form."""
    wide = wide[wide["session_id"].isin(sessions)]
    value_vars = [v for v in var_metadata if v in wide.columns]

    subset = wide[_IDS + value_vars].copy()
    for v in value_vars:  # drop categorical encoding so melt yields plain values
        subset[v] = subset[v].astype("object")
    long = subset.melt(id_vars=_IDS, value_vars=value_vars, var_name="variable", value_name="raw")

    # NaN == pair absent from that variable's source table -> not a real observation
    long = long.dropna(subset=["raw"])
    long["resolved"] = resolve_values(long["raw"])
    long["informant"] = long["variable"].map(lambda v: var_metadata[v]["informant"])
    long["module"] = long["variable"].map(lambda v: var_metadata[v]["module"])
    long["status_layer"] = long["variable"].map(lambda v: var_metadata[v]["status"])
    long["disorder"] = long["variable"].map(
        lambda v: var_metadata[v]["disorder"] or var_metadata[v]["label"]
    )
    long = long.drop(columns="raw")

    long["resolved"] = pd.Categorical(long["resolved"], categories=RESOLVED, ordered=True)
    for c in ["session_id", "informant", "module", "status_layer", "variable"]:
        long[c] = long[c].astype("category")
    return long.reset_index(drop=True)


def build_summary(long: pd.DataFrame, var_metadata: dict) -> pd.DataFrame:
    """Per variable x session counts of each resolved state."""
    g = (
        long.groupby(["variable", "session_id", "resolved"], observed=True)
        .size()
        .unstack("resolved", fill_value=0)
        .reset_index()
    )
    for col in RESOLVED:
        if col not in g:
            g[col] = 0

    rows = []
    for _, rr in g.iterrows():
        meta = var_metadata[rr["variable"]]
        rows.append(
            {
                "variable": rr["variable"],
                "session_id": rr["session_id"],
                "informant": meta["informant"],
                "module": meta["module"],
                "status_layer": meta["status"],
                "n_positive": int(rr["positive"]),
                "n_administered_negative": int(rr["administered_negative"]),
                "n_not_administered": int(rr["not_administered"]),
                "n_no_record": int(rr["no_record"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["informant", "module", "variable", "session_id"])

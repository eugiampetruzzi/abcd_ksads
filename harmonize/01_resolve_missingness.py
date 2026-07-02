#!/usr/bin/env python3
"""Resolve KSADS diagnosis cells into missingness states from the raw cache.

Reads the diagnosis columns of the consolidated phenotype.parquet (built by
harmonize/00_ingest.py) and writes the resolved long table and per-variable summary.
"""

import pandas as pd
import pyarrow.parquet as pq

from abcd_ksads import config
from abcd_ksads.resolve import (
    SESSIONS,
    RESOLVED,
    load_diagnosis_metadata,
    resolve_wide,
    build_summary,
)


def resolve():
    var_metadata = load_diagnosis_metadata(config.KSADS_VARIABLE_MAP)

    source = config.RAW_CACHE / "phenotype.parquet"
    available = set(pq.ParquetFile(source).schema.names)
    columns = ["participant_id", "session_id"] + [
        v for v in var_metadata if v in available
    ]
    wide = pd.read_parquet(source, columns=columns)

    long = resolve_wide(wide, var_metadata)
    out_long = config.DERIV / "ksads_resolved_long.parquet"
    long.to_parquet(out_long, index=False)

    summ = build_summary(long, var_metadata)
    out_summ = config.DERIV / "ksads_resolution_summary.csv"
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


if __name__ == "__main__":
    resolve()

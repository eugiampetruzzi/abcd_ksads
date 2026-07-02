#!/usr/bin/env python3
"""Collect the summary numbers reported in the paper into paper_numbers.json; the
collation logic lives in abcd_ksads.paper_numbers."""

import json

import pandas as pd

from abcd_ksads import config
from abcd_ksads.paper_numbers import collate_numbers


def main():
    rs = pd.read_csv(config.DERIV / "ksads_resolution_summary.csv")
    cw = pd.read_csv(config.DERIV / "ksads_category_crosswalk.csv")
    msum = pd.read_csv(config.DERIV / "multiverse_summary.csv")
    lev = pd.read_csv(config.DERIV / "single_lever.csv")
    anx = pd.read_csv(config.DERIV / "anxiety_decomposition.csv")
    miss = pd.read_csv(config.DERIV / "missingness_error.csv").iloc[0]
    ver = pd.read_parquet(
        config.DERIV / "ksads_resolved_versioned.parquet",
        columns=["ksads_version", "version_valid"],
    )

    out = collate_numbers(rs, cw, msum, lev, anx, miss, ver)

    path = config.DERIV / "paper_numbers.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

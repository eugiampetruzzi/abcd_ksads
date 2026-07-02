#!/usr/bin/env python3
"""One-decision-at-a-time prevalence shifts; the lever logic lives in
abcd_ksads.multiverse (single_lever_table)."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.multiverse import BASE_SES, build_primitive_cache, single_lever_table


def main():
    cw = build_crosswalk()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()
    cache = build_primitive_cache(base, cw)

    df = single_lever_table(cache)
    df.to_csv(config.DERIV / "single_lever.csv", index=False)

    print(df.to_string(index=False))
    print(f"\nWrote {config.DERIV.as_posix()}/single_lever.csv")


if __name__ == "__main__":
    main()

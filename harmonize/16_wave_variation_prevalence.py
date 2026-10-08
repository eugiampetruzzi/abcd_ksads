#!/usr/bin/env python3
"""Prevalence multiverse repeated at each diagnostic assessment (ses-00A..ses-06A)."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import EVEN, build_crosswalk
from abcd_ksads.multiverse import build_multiverse_grid, build_primitive_cache, informant_validity


def main():
    cw = build_crosswalk()
    cal = pd.read_csv(config.DERIV / "ksads_administration_calendar.csv")
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    grids = []
    for ses in EVEN:
        base = resolved[resolved.session_id == ses].copy()
        grid, _ = build_multiverse_grid(build_primitive_cache(base, cw), informant_validity(cw, cal, ses))
        grids.append(grid.assign(window=ses))
    grid = pd.concat(grids, ignore_index=True).rename(columns={"window": "wave"})
    grid.to_csv(config.DERIV / "wave_variation_grid.csv", index=False)

    print(grid.groupby(["wave", "construct"]).prevalence_pct.agg(["size", "min", "max", "median"]).round(2).to_string())
    print(f"\nWrote {config.DERIV.as_posix()}/wave_variation_grid.csv")


if __name__ == "__main__":
    main()

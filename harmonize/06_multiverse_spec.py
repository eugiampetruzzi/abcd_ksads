#!/usr/bin/env python3
"""Enumerate the prevalence multiverse; the enumeration logic lives in
abcd_ksads.multiverse (build_multiverse_grid)."""

import numpy as np
import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.multiverse import (
    BASE_SES,
    build_multiverse_grid,
    build_primitive_cache,
    informant_validity,
)


def main():
    cw = build_crosswalk()
    cal = pd.read_csv(config.DERIV / "ksads_administration_calendar.csv")
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()

    cache = build_primitive_cache(base, cw)
    valid = informant_validity(cw, cal)
    grid, skipped = build_multiverse_grid(cache, valid)
    grid.to_csv(config.DERIV / "multiverse_grid.csv", index=False)

    print(
        f"Enumerated {len(grid)} valid specifications "
        f"({skipped} impossible cells skipped).\n"
    )
    hdr = f"{'construct':14} {'n':>4} {'min%':>7} {'max%':>7} {'fold':>6} {'median%':>8} {'IQR%':>14}"
    print(hdr)
    print("-" * len(hdr))
    for con, sub in grid.groupby("construct"):
        p = sub.prevalence_pct
        q1, q3 = p.quantile(0.25), p.quantile(0.75)
        fold = p.max() / p.min() if p.min() > 0 else np.inf
        print(
            f"{con:14} {len(sub):>4} {p.min():>7.2f} {p.max():>7.2f} "
            f"{fold:>6.1f} {p.median():>8.2f} {q1:>6.2f}-{q3:<7.2f}"
        )
    print(f"\nWrote {config.DERIV.as_posix()}/multiverse_grid.csv")


if __name__ == "__main__":
    main()

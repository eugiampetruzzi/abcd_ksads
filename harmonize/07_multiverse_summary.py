#!/usr/bin/env python3
"""Per-construct fold-range summary of the prevalence multiverse; the summary logic
lives in abcd_ksads.multiverse (summarize_multiverse)."""

import numpy as np
import pandas as pd

from abcd_ksads import config
from abcd_ksads.multiverse import summarize_multiverse


def main():
    grid = pd.read_csv(config.DERIV / "multiverse_grid.csv")
    summ = summarize_multiverse(grid)
    summ.to_csv(config.DERIV / "multiverse_summary.csv", index=False)

    print(summ.to_string(index=False))

    anyd = summ[summ.construct == "any-disorder"].iloc[0]
    # robust max fold = largest fold among constructs whose min prevalence is not near-zero
    stable = summ[~summ.unstable_fold]
    rmax = stable.loc[stable.fold_range.idxmax()]
    # raw max fold across all constructs (may be off a tiny base)
    raw = summ.loc[summ.fold_range.replace(np.inf, 1e9).idxmax()]

    print("\nHEADLINE")
    print(
        f"  any-disorder: {anyd.prev_min:.1f}% to {anyd.prev_max:.1f}% "
        f"({anyd.fold_range:.0f}-fold) across {int(anyd.n_specs)} specifications"
    )
    print(
        f"  max stable single-construct fold: {rmax.construct} "
        f"{rmax.prev_min:.1f}-{rmax.prev_max:.1f}% ({rmax.fold_range:.0f}-fold)"
    )
    print(
        f"  most extreme construct (flagged, near-zero base): {raw.construct} "
        f"{raw.prev_min:.2f}-{raw.prev_max:.1f}% "
        f"(fold off a {raw.prev_min:.2f}% base; pp-span {raw.pp_span:.1f} points)"
    )
    print(f"\nWrote {config.DERIV.as_posix()}/multiverse_summary.csv")


if __name__ == "__main__":
    main()

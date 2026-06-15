#!/usr/bin/env python3
"""Headline multiverse statistics from multiverse_grid.csv.

Emits per-construct min/max/fold/median/IQR, and the two headline numbers the
abstract needs: the any-disorder range (intuitive) and the maximum single-construct
fold-range. Guards against divide-by-tiny: any construct with min prevalence < 0.1%
is flagged and its absolute percentage-point span is reported alongside the fold.

Output: derivatives/multiverse_summary.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
TINY = 0.1   # min-prevalence floor below which fold-range is flagged unstable


def main():
    grid = pd.read_csv(os.path.join(DERIV, "multiverse_grid.csv"))
    rows = []
    for con, sub in grid.groupby("construct"):
        p = sub.prevalence_pct
        lo, hi = p.min(), p.max()
        fold = hi / lo if lo > 0 else np.inf
        rows.append({
            "construct": con, "n_specs": len(sub),
            "prev_min": round(lo, 3), "prev_max": round(hi, 3),
            "fold_range": round(fold, 1) if np.isfinite(fold) else np.inf,
            "pp_span": round(hi - lo, 2),
            "prev_median": round(p.median(), 3),
            "prev_iqr_low": round(p.quantile(.25), 3),
            "prev_iqr_high": round(p.quantile(.75), 3),
            "unstable_fold": bool(lo < TINY),
        })
    summ = pd.DataFrame(rows).sort_values("fold_range", ascending=False)
    summ.to_csv(os.path.join(DERIV, "multiverse_summary.csv"), index=False)

    print(summ.to_string(index=False))

    anyd = summ[summ.construct == "any-disorder"].iloc[0]
    # robust max fold = largest fold among constructs whose min prevalence is not near-zero
    stable = summ[~summ.unstable_fold]
    rmax = stable.loc[stable.fold_range.idxmax()]
    # raw max fold across all constructs (may be off a tiny base)
    raw = summ.loc[summ.fold_range.replace(np.inf, 1e9).idxmax()]

    print("\nHEADLINE")
    print(f"  any-disorder: {anyd.prev_min:.1f}% to {anyd.prev_max:.1f}% "
          f"({anyd.fold_range:.0f}-fold) across {int(anyd.n_specs)} specifications")
    print(f"  max stable single-construct fold: {rmax.construct} "
          f"{rmax.prev_min:.1f}-{rmax.prev_max:.1f}% ({rmax.fold_range:.0f}-fold)")
    print(f"  most extreme construct (flagged, near-zero base): {raw.construct} "
          f"{raw.prev_min:.2f}-{raw.prev_max:.1f}% "
          f"(fold off a {raw.prev_min:.2f}% base; pp-span {raw.pp_span:.1f} points)")
    print(f"\nWrote {DERIV}/multiverse_summary.csv")


if __name__ == "__main__":
    main()

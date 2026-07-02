#!/usr/bin/env python3
"""Quantify the 555-as-0 missingness error for parent MDD present-diagnosis; the
computation lives in abcd_ksads.missingness."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.missingness import missingness_error

VAR = "mh_p_ksads__dep__mdd__pres_dx"


def main():
    r = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    r["variable"] = r["variable"].astype(str)
    r["resolved"] = r["resolved"].astype(str)
    d = r[r.variable == VAR]

    stats = missingness_error(d)
    out = pd.DataFrame([{"variable": VAR, **stats}])
    out.to_csv(config.DERIV / "missingness_error.csv", index=False)

    print("555-as-0 error (parent MDD present, person-wave over 8 sessions):")
    print(
        f"  correct (555 excluded):    {stats['prevalence_correct_pct']:.3f}%  "
        f"({stats['n_positive']:,}/{stats['n_administered']:,})"
    )
    print(
        f"  error   (555 -> 0):        {stats['prevalence_error_pct']:.3f}%  "
        f"({stats['n_positive']:,}/{stats['n_all_personwaves']:,})"
    )
    print(f"  fold-deflation:            {stats['fold_deflation']:.2f}x")
    print(
        f"  fabricated assessed-neg person-waves: {stats['fabricated_personwaves']:,}"
    )
    print(f"\nWrote {config.DERIV.as_posix()}/missingness_error.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
VAR = "mh_p_ksads__dep__mdd__pres_dx"


def main():
    r = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    r["variable"] = r["variable"].astype(str)
    r["resolved"] = r["resolved"].astype(str)
    d = r[r.variable == VAR]

    n_pos = int((d.resolved == "positive").sum())
    n_assessed = int(d.resolved.isin(["positive", "administered_negative"]).sum())
    n_all = int(len(d))            # every person-wave with a row (incl. 555 and no_record)
    n_not_admin = int((d.resolved == "not_administered").sum())

    correct = 100 * n_pos / n_assessed
    error = 100 * n_pos / n_all
    fold = correct / error

    out = pd.DataFrame([{
        "variable": VAR, "n_positive": n_pos,
        "n_administered": n_assessed, "n_all_personwaves": n_all,
        "prevalence_correct_pct": round(correct, 3),
        "prevalence_error_pct": round(error, 3),
        "fold_deflation": round(fold, 2),
        "fabricated_personwaves": n_not_admin,
    }])
    out.to_csv(os.path.join(DERIV, "missingness_error.csv"), index=False)

    print("555-as-0 error (parent MDD present, person-wave over 8 sessions):")
    print(f"  correct (555 excluded):    {correct:.3f}%  ({n_pos:,}/{n_assessed:,})")
    print(f"  error   (555 -> 0):        {error:.3f}%  ({n_pos:,}/{n_all:,})")
    print(f"  fold-deflation:            {fold:.2f}x")
    print(f"  fabricated assessed-neg person-waves: {n_not_admin:,}")
    print(f"\nWrote {DERIV}/missingness_error.csv")


if __name__ == "__main__":
    main()

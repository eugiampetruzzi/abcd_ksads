#!/usr/bin/env python3
"""Inferential multiverse: predictors are read from the consolidated phenotype cache.

The required predictor columns (SOURCES) and their recoding into analysis variables
live in abcd_ksads.predictors; the modeling logic lives in abcd_ksads.inferential.
This script wires those to the cached diagnosis outcomes and writes the specification
grid and its per-pair summary.
"""

import warnings

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.inferential import build_specs, summarize_specs
from abcd_ksads.multiverse import BASE_SES, build_primitive_cache
from abcd_ksads.predictors import load_predictors

warnings.filterwarnings("ignore")


def main():
    cw = build_crosswalk()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()
    cache = build_primitive_cache(base, cw)
    P = load_predictors()

    res = build_specs(P, cache)
    res.to_csv(config.DERIV / "inferential_specs.csv", index=False)

    S = summarize_specs(res)
    S.to_csv(config.DERIV / "inferential_summary.csv", index=False)

    n_pairs = len(S)
    print(
        f"Fit {len(res.dropna(subset=['OR']))} specifications across {n_pairs} predictor x construct pairs."
    )
    print(f"  specifications significant (p<.05): {100 * res.sig.mean():.1f}%")
    print(f"  pairs with >=1 significant spec:    {100 * S.any_sig.mean():.1f}%")
    print(f"  pairs significant in ALL specs:     {100 * S.all_sig.mean():.1f}%")
    print(f"  pairs that flip OR sign:            {100 * S.sign_flip.mean():.1f}%")
    print()
    print("Mean variance share by axis (eta^2 of logOR):")
    for ax in ["status", "informant"]:
        print(f"  {ax:11}: {S[f'eta2_{ax}'].mean():.2f}")
    print(
        f"\nWrote inferential_specs.csv ({len(res)} rows) and inferential_summary.csv ({n_pairs} pairs)"
    )


if __name__ == "__main__":
    main()

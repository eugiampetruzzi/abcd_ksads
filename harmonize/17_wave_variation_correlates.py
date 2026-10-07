#!/usr/bin/env python3
"""Inferential multiverse repeated at each diagnostic assessment (ses-00A..ses-06A).

Predictors and diagnoses are drawn from the same assessment. Screen time and household
income are not repeated with a comparable instrument after ses-01A / ses-00A, so their
models are skipped automatically at follow-up (no usable rows).
"""

import warnings

import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import EVEN, build_crosswalk
from abcd_ksads.inferential import build_specs, summarize_specs
from abcd_ksads.multiverse import build_primitive_cache
from abcd_ksads.predictors import load_predictors

warnings.filterwarnings("ignore")


def main():
    cw = build_crosswalk()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    specs, summ = [], []
    for ses in EVEN:
        cache = build_primitive_cache(resolved[resolved.session_id == ses].copy(), cw)
        res = build_specs(load_predictors(base_ses=ses), cache)
        specs.append(res.assign(wave=ses))
        summ.append(summarize_specs(res).assign(wave=ses))
    pd.concat(specs, ignore_index=True).to_csv(config.DERIV / "wave_variation_specs.csv", index=False)
    S = pd.concat(summ, ignore_index=True)
    S.to_csv(config.DERIV / "wave_variation_correlates.csv", index=False)

    print(S.groupby(["wave", "bucket"]).agg(pairs=("sign_flip", "size"), sign_flip=("sign_flip", "sum"),
                                             all_sig=("all_sig", "sum")).to_string())
    print(f"\nWrote {config.DERIV.as_posix()}/wave_variation_correlates.csv")


if __name__ == "__main__":
    main()

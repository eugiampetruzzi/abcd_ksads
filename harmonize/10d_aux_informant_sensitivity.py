#!/usr/bin/env python3
"""Informant sensitivity analyses.

1. Prevalence and baseline correlate models restricted to modules administered to
   BOTH informants, so caregiver and youth are compared on the same diagnoses.
2. From inferential_specs.csv: sign reversals significant in both directions, and
   timeframe vs informant variance shares using caregiver and youth specs only.
"""

import warnings

import numpy as np
import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import EVEN, build_crosswalk
from abcd_ksads.inferential import build_specs, eta2
from abcd_ksads.multiverse import (BASE_SES, build_multiverse_grid, build_primitive_cache,
                                   informant_validity, shared_module_crosswalk)
from abcd_ksads.predictors import load_predictors

warnings.filterwarnings("ignore")
KEYS = ["construct", "predictor"]


def robustness(res):
    """Per pair: significant reversal, and caregiver/youth 2 x 2 eta^2 (NaN if incomplete)."""
    rows = []
    for (con, pred), sub in res.dropna(subset=["OR"]).groupby(KEYS):
        cy = sub[sub.informant.isin(["parent", "youth"])]
        e = eta2(cy) if len(cy) == 4 and cy.status.nunique() == 2 else {"eta2_status": np.nan, "eta2_informant": np.nan}
        rows.append({"construct": con, "predictor": pred,
                     "sign_flip": sub.OR.min() < 1 < sub.OR.max(),
                     "sig_reversal": (sub.sig & (sub.OR > 1)).any() and (sub.sig & (sub.OR < 1)).any(),
                     "or_ratio": sub.OR.max() / sub.OR.min(),
                     "eta2_status_cy": e["eta2_status"], "eta2_informant_cy": e["eta2_informant"]})
    return pd.DataFrame(rows)


def report(label, R):
    print(f"{label}: {len(R)} pairs, sign flips {R.sign_flip.sum()}, significant reversals {R.sig_reversal.sum()}, "
          f"eta2 timeframe {R.eta2_status_cy.mean():.2f} vs informant {R.eta2_informant_cy.mean():.2f}")


def main():
    cw_full = build_crosswalk()
    cal = pd.read_csv(config.DERIV / "ksads_administration_calendar.csv")
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)

    # 1a. prevalence on shared modules at each assessment
    grids = []
    for ses in EVEN:
        cw = shared_module_crosswalk(cw_full, cal, ses)
        base = resolved[resolved.session_id == ses].copy()
        grid, _ = build_multiverse_grid(build_primitive_cache(base, cw), informant_validity(cw, cal, ses))
        grids.append(grid.assign(window=ses))
    pd.concat(grids, ignore_index=True).rename(columns={"window": "wave"}).to_csv(
        config.DERIV / "informant_shared_modules_grid.csv", index=False)

    # 1b. baseline correlate models on shared modules
    cw = shared_module_crosswalk(cw_full, cal, BASE_SES)
    cache = build_primitive_cache(resolved[resolved.session_id == BASE_SES].copy(), cw)
    constructs = [("anxiety", "Anxiety"), ("any-disorder", "Any disorder"),
                  ("depression", "Depression"), ("suicidality", "Suicidality")]
    shared = build_specs(load_predictors(), cache, constructs=constructs)
    shared.to_csv(config.DERIV / "inferential_specs_shared.csv", index=False)

    # 2. robustness summaries
    R_main = robustness(pd.read_csv(config.DERIV / "inferential_specs.csv"))
    R_shared = robustness(shared)
    R_main.to_csv(config.DERIV / "inferential_robustness.csv", index=False)
    R_shared.to_csv(config.DERIV / "inferential_robustness_shared.csv", index=False)
    report("All modules", R_main)
    report("Shared modules", R_shared)
    print(R_main[R_main.sig_reversal][KEYS].to_string(index=False))


if __name__ == "__main__":
    main()

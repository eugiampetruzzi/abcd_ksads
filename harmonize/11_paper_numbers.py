#!/usr/bin/env python3
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")


def _num(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


def main():
    rs = pd.read_csv(os.path.join(DERIV, "ksads_resolution_summary.csv"))
    tot = rs[["n_positive", "n_administered_negative",
              "n_not_administered", "n_no_record"]].sum()
    n_cells = int(tot.sum())

    cw = pd.read_csv(os.path.join(DERIV, "ksads_category_crosswalk.csv"))
    msum = pd.read_csv(os.path.join(DERIV, "multiverse_summary.csv"))
    lev = pd.read_csv(os.path.join(DERIV, "single_lever.csv"))
    anx = pd.read_csv(os.path.join(DERIV, "anxiety_decomposition.csv"))
    miss = pd.read_csv(os.path.join(DERIV, "missingness_error.csv")).iloc[0]
    ver = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_versioned.parquet"),
                          columns=["ksads_version", "version_valid"])

    mv = {}
    for _, r in msum.iterrows():
        mv[r.construct.replace("-", "_")] = {
            "n_specs": int(r.n_specs), "min": _num(r.prev_min), "max": _num(r.prev_max),
            "fold": _num(r.fold_range), "pp_span": _num(r.pp_span),
            "median": _num(r.prev_median), "unstable_fold": bool(r.unstable_fold)}
    stable = msum[~msum.unstable_fold]
    raw = msum.loc[msum.fold_range.replace(np.inf, 1e9).idxmax()]
    mv["headline_any_disorder"] = mv["any_disorder"]
    mv["headline_max_fold"] = {"value": _num(raw.fold_range), "construct": raw.construct,
                               "min": _num(raw.prev_min), "max": _num(raw.prev_max),
                               "unstable": bool(raw.unstable_fold),
                               "pp_span": _num(raw.pp_span)}
    mv["headline_max_stable_fold"] = {
        "value": _num(stable.fold_range.max()),
        "construct": stable.loc[stable.fold_range.idxmax()].construct}

    levmap = {"current -> ever-met": "ever_met", "parent -> youth-only": "youth_only",
              "parent -> either": "either", "+ subthreshold dx": "subthreshold",
              "anxiety: drop phobia": "phobia_out"}
    single = {levmap[r.lever]: _num(r.delta_pts) for _, r in lev.iterrows()
              if r.lever in levmap}
    single["base_prevalence"] = _num(lev.iloc[0].prevalence_pct)

    def anx_get(name):
        return _num(anx.loc[anx["sub"] == name, "prevalence_pct"].iloc[0])
    any_in, any_out = anx_get("ANY (with phobia)"), anx_get("ANY (without phobia)")

    out = {
        "n_cells_total": n_cells,
        "pct_positive": round(100 * tot.n_positive / n_cells, 2),
        "pct_administered_negative": round(100 * tot.n_administered_negative / n_cells, 2),
        "pct_555": round(100 * tot.n_not_administered / n_cells, 2),
        "pct_no_record": round(100 * tot.n_no_record / n_cells, 2),
        "n_diagnosis_vars": int(cw.shape[0]),
        "n_categories": int(cw.category.nunique()),
        "n_subthreshold": int(cw.is_subthreshold.sum()),
        "multiverse": mv,
        "single_lever": single,
        "anxiety": {
            "phobia_only": anx_get("Specific phobia"),
            "others_combined": any_out,
            "any_in": any_in, "any_out": any_out,
            "fold": round(any_in / any_out, 1) if any_out else None},
        "missingness_555": {
            "correct_pct": _num(miss.prevalence_correct_pct),
            "error_pct": _num(miss.prevalence_error_pct),
            "fold": _num(miss.fold_deflation),
            "fabricated_personwaves": int(miss.fabricated_personwaves)},
        "version": {
            "cells_v1": int((ver.ksads_version.astype(str) == "1.0").sum()),
            "cells_v2": int((ver.ksads_version.astype(str) == "2.0").sum()),
            "two_zero_only_under_one": int((~ver.version_valid).sum())},
    }
    path = os.path.join(DERIV, "paper_numbers.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
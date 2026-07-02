"""Collate the summary numbers reported in the paper into a single dict.

Pulls headline figures from the resolution summary, crosswalk, multiverse summary,
single-lever table, anxiety decomposition, missingness audit, and version cache. The
``11_paper_numbers.py`` script wires this to the CSVs and writes paper_numbers.json.
"""

import numpy as np

# single-lever row label -> json key
LEVER_KEYS = {
    "current -> ever-met": "ever_met",
    "parent -> youth-only": "youth_only",
    "parent -> either": "either",
    "+ subthreshold dx": "subthreshold",
    "anxiety: drop phobia": "phobia_out",
}


def _num(x):
    """JSON-friendly scalar: ints stay ints, floats round to 4 dp (NaN -> None)."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else round(float(x), 4)
    return x


def _multiverse(msum):
    mv = {}
    for _, r in msum.iterrows():
        mv[r.construct.replace("-", "_")] = {
            "n_specs": int(r.n_specs),
            "min": _num(r.prev_min),
            "max": _num(r.prev_max),
            "fold": _num(r.fold_range),
            "pp_span": _num(r.pp_span),
            "median": _num(r.prev_median),
            "unstable_fold": bool(r.unstable_fold),
        }
    stable = msum[~msum.unstable_fold]
    raw = msum.loc[msum.fold_range.replace(np.inf, 1e9).idxmax()]
    mv["headline_any_disorder"] = mv["any_disorder"]
    mv["headline_max_fold"] = {
        "value": _num(raw.fold_range),
        "construct": raw.construct,
        "min": _num(raw.prev_min),
        "max": _num(raw.prev_max),
        "unstable": bool(raw.unstable_fold),
        "pp_span": _num(raw.pp_span),
    }
    mv["headline_max_stable_fold"] = {
        "value": _num(stable.fold_range.max()),
        "construct": stable.loc[stable.fold_range.idxmax()].construct,
    }
    return mv


def collate_numbers(rs, cw, msum, lev, anx, miss, ver):
    """Assemble the reported-numbers dict from the pipeline output tables."""
    tot = rs[
        ["n_positive", "n_administered_negative", "n_not_administered", "n_no_record"]
    ].sum()
    n_cells = int(tot.sum())

    single = {
        LEVER_KEYS[r.lever]: _num(r.delta_pts)
        for _, r in lev.iterrows()
        if r.lever in LEVER_KEYS
    }
    single["base_prevalence"] = _num(lev.iloc[0].prevalence_pct)

    def anx_get(name):
        return _num(anx.loc[anx["sub"] == name, "prevalence_pct"].iloc[0])

    any_in, any_out = anx_get("ANY (with phobia)"), anx_get("ANY (without phobia)")

    return {
        "n_cells_total": n_cells,
        "pct_positive": round(100 * tot.n_positive / n_cells, 2),
        "pct_administered_negative": round(100 * tot.n_administered_negative / n_cells, 2),
        "pct_555": round(100 * tot.n_not_administered / n_cells, 2),
        "pct_no_record": round(100 * tot.n_no_record / n_cells, 2),
        "n_diagnosis_vars": int(cw.shape[0]),
        "n_categories": int(cw.category.nunique()),
        "n_subthreshold": int(cw.is_subthreshold.sum()),
        "multiverse": _multiverse(msum),
        "single_lever": single,
        "anxiety": {
            "phobia_only": anx_get("Specific phobia"),
            "others_combined": any_out,
            "any_in": any_in,
            "any_out": any_out,
            "fold": round(any_in / any_out, 1) if any_out else None,
        },
        "missingness_555": {
            "correct_pct": _num(miss.prevalence_correct_pct),
            "error_pct": _num(miss.prevalence_error_pct),
            "fold": _num(miss.fold_deflation),
            "fabricated_personwaves": int(miss.fabricated_personwaves),
        },
        "version": {
            "cells_v1": int((ver.ksads_version.astype(str) == "1.0").sum()),
            "cells_v2": int((ver.ksads_version.astype(str) == "2.0").sum()),
            "two_zero_only_under_one": int((~ver.version_valid).sum()),
        },
    }

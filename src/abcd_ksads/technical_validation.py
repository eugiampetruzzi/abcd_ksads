"""Technical-validation checks for the harmonized dataset.

Three independent checks: (1) correctness -- resolved positives must equal the raw
ABCD value-1 counts; (2) face validity -- default caseness prevalence vs published
CDC rates; (3) parent-youth concordance (Cohen's kappa). The
``13_technical_validation.py`` script wires these to the cache and CSVs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def correctness_by_category(res_pos, raw_one, cw_cat):
    """Per DSM category, compare resolved-positive counts to raw value-1 counts.

    ``res_pos`` / ``raw_one`` map variable -> count; ``cw_cat`` maps variable ->
    category. A category matches when the two totals are equal."""
    corr = []
    for cat in sorted(set(cw_cat.values())):
        vs = [v for v, c in cw_cat.items() if c == cat]
        nres = sum(res_pos.get(v, 0) for v in vs)
        nraw = sum(raw_one.get(v, 0) for v in vs)
        corr.append(
            {
                "category": cat,
                "n_resolved_positive": nres,
                "n_raw_value1": nraw,
                "match": nres == nraw,
            }
        )
    return pd.DataFrame(corr)


def caseness_prevalence(caseness, cats):
    """Prevalence (%) of being positive on any of ``cats`` over the assessed denominator."""
    c = caseness[caseness.category.isin(cats)]
    piv = c.pivot_table(
        index="participant_id", columns="category", values="status", aggfunc="first"
    )
    pos = (piv == "positive").any(axis=1)
    adm = (piv.notna() & (piv != "not_administered")).any(axis=1)
    den = int(adm.sum())
    return (100 * pos.sum() / den) if den else np.nan


def concordance_kappa(cp, cy, cat, min_n=50):
    """Parent-youth agreement on one category: 2x2 counts and Cohen's kappa.

    Restricts to participants assessed by both informants; kappa is NaN when fewer
    than ``min_n`` such participants exist."""
    p = cp[cp.category == cat].set_index("participant_id")["status"]
    y = cy[cy.category == cat].set_index("participant_id")["status"]
    m = pd.DataFrame({"p": p, "y": y}).dropna()
    m = m[(m.p != "not_administered") & (m.y != "not_administered")]
    pb = (m.p == "positive").astype(int)
    yb = (m.y == "positive").astype(int)
    k = cohen_kappa_score(pb, yb) if len(m) > min_n else np.nan
    return {
        "category": cat,
        "n_both_assessed": len(m),
        "parent_pos_pct": round(100 * pb.mean(), 2),
        "youth_pos_pct": round(100 * yb.mean(), 2),
        "cohen_kappa": round(k, 3),
        "both_positive": int(((pb == 1) & (yb == 1)).sum()),
    }

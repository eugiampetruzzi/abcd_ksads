"""Informant (caregiver vs youth) prevalence and concordance for KSADS caseness.

Operates on per-person status Series (participant_id -> positive /
administered_negative / not_administered), so the statistics are testable without
touching the pipeline's file I/O. Cohen's kappa uses scikit-learn, matching the
technical-validation script.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def prevalence(status: pd.Series) -> dict:
    """Positive count, administered denominator, and prevalence % for one informant."""
    den = int((status != "not_administered").sum())
    pos = int((status == "positive").sum())
    return {
        "n_positive": pos,
        "n_denominator": den,
        "prevalence_pct": (100 * pos / den) if den else np.nan,
    }


def _both_administered(parent: pd.Series, youth: pd.Series) -> pd.DataFrame:
    """Participants administered in BOTH interviews (paired, non-missing on each side)."""
    df = pd.concat([parent.rename("p"), youth.rename("y")], axis=1).dropna()
    return df[(df.p != "not_administered") & (df.y != "not_administered")]


def concordance(parent: pd.Series, youth: pd.Series):
    """Parent-youth agreement over participants administered both interviews.

    Returns the 2x2 breakdown and Cohen's kappa, or ``None`` if no participant was
    administered in both.
    """
    df = _both_administered(parent, youth)
    if df.empty:
        return None
    pb = (df.p == "positive").astype(int)
    yb = (df.y == "positive").astype(int)
    both = int(((pb == 1) & (yb == 1)).sum())
    parent_only = int(((pb == 1) & (yb == 0)).sum())
    youth_only = int(((pb == 0) & (yb == 1)).sum())
    return {
        "n_both_admin": len(df),
        "both_pos": both,
        "parent_only": parent_only,
        "youth_only": youth_only,
        "union_pos": both + parent_only + youth_only,
        "kappa": cohen_kappa_score(pb, yb),
    }

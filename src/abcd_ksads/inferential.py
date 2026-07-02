"""Inferential multiverse: logistic models of caseness on demographic, psychosocial,
and neuroimaging predictors, across the construct x informant x timeframe grid.

Every model is a logistic GLM with cluster-robust standard errors on family (to
account for siblings), adjusting for interview age and sex (except when sex is the
focal predictor) and study site; neuroimaging models additionally adjust for scanner
and mean framewise displacement. The functions here are pure (DataFrame in, results
out); the ``09_inferential_multiverse.py`` script wires them to the cache and CSVs.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from abcd_ksads.multiverse import construct_status
from abcd_ksads.predictors import RACE_LVES, RACE_REF

CONSTRUCTS = [
    ("suicidality", "Suicidality"),
    ("eating", "Eating disorders"),
    ("depression", "Depression"),
    ("anxiety", "Anxiety"),
    ("any-disorder", "Any disorder"),
    ("ADHD", "ADHD"),
]
INFORMANTS = ["parent", "youth", "either", "both"]
STATUSES = ["current", "ever_met"]
THRESH = [False]

NEURAL = [
    ("fc_dmn_within_z", "DMN within-network FC (per SD)"),
    ("fc_sal_within_z", "Salience within-network FC (per SD)"),
    ("fc_fpn_within_z", "FPN within-network FC (per SD)"),
    ("fc_dmn_salience_z", "DMN-salience FC (per SD)"),
    ("fc_dmn_fpn_z", "DMN-FPN FC (per SD)"),
    ("fc_sal_fpn_z", "Salience-FPN FC (per SD)"),
]
NUIS = ["site", "family_id"]

# caseness status -> binary outcome; not-administered is not a usable observation
Y_MAP = {"positive": 1, "administered_negative": 0, "not_administered": np.nan}

# plausibility window: odds ratios outside this range signal separation/instability
OR_MIN, OR_MAX = 0.02, 50


def fit_adj(d, focal, imaging=False):
    """Logistic GLM of d.y on focal term(s), adjusting for interview age, sex (unless sex
    is the focal predictor), and study site as a fixed effect; imaging models additionally
    adjust for scanner manufacturer and mean framewise displacement. Standard errors are
    cluster-robust on family to account for siblings. Returns {term: (OR, p)}; nan on
    failure or |effect| outside 0.02-50."""
    out = {t: (np.nan, np.nan) for t in focal}
    try:
        cols = list(focal) + ([] if "sex_f" in focal else ["sex_f"]) + ["age_z"]
        X = d[cols].copy()
        X = X.join(pd.get_dummies(d.site, prefix="site", drop_first=True).astype(float))
        if imaging:
            X = X.join(
                pd.get_dummies(d.scanner, prefix="scn", drop_first=True).astype(float)
            )
            X["mean_fd_z"] = d["mean_fd_z"].values
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.GLM(d.y, Xc, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.family_id}
        )
        for t in focal:
            orr = float(np.exp(res.params[t]))
            p = float(res.pvalues[t])
            if np.isfinite(orr) and OR_MIN <= orr <= OR_MAX:
                out[t] = (orr, p)
    except Exception:
        pass
    return out


def enough(yv, mask=None):
    """outcome usable: >=100 rows, >=10 positives (in subgroup if given)."""
    if len(yv) < 100 or yv.sum() < 10:
        return False
    if mask is not None and (mask & (yv == 1)).sum() < 10:
        return False
    return True


def eta2(sub):
    """Variance of logOR attributable to each axis (one-way eta^2)."""
    out = {}
    tot = ((sub.logor - sub.logor.mean()) ** 2).sum()
    for ax in ["status", "informant"]:
        ss = sum(
            len(g) * (g.logor.mean() - sub.logor.mean()) ** 2
            for _, g in sub.groupby(ax)
        )
        out[f"eta2_{ax}"] = ss / tot if tot > 0 else np.nan
    return out


def bucket_of(lab):
    if lab.startswith("Female"):
        return "Sex"
    if lab.startswith("Income"):
        return "Income"
    if lab.startswith("Race"):
        return "Race/ethnicity"
    if "FC" in lab:
        return "Neuroimaging"
    return "Culture/environment"


def outcome_frame(P, stat):
    """Join predictors ``P`` with a caseness Series, mapping it to a binary ``y``.

    Rows with a not-administered outcome are dropped. Returns None when the caseness
    Series is empty."""
    if stat is None or len(stat) == 0:
        return None
    y = stat.map(Y_MAP)
    return P.join(y.rename("y"), how="inner").dropna(subset=["y"])


def fit_spec(df):
    """Fit every predictor bucket for one construct x informant x timeframe spec.

    Returns ``{predictor label: (OR, p)}`` for the sex, income, culture/environment,
    neuroimaging, and race buckets, gated by :func:`enough`."""
    out = {}

    # --- Sex bucket: y ~ sex + age (+ site, family-clustered) ---
    d = df[["y", "sex_f", "age_z"] + NUIS].dropna()
    if enough(d.y):
        out["Female (vs male)"] = fit_adj(d, ["sex_f"])["sex_f"]

    # --- Income bucket ---
    d = df[["y", "income_z", "sex_f", "age_z"] + NUIS].dropna()
    if enough(d.y):
        out["Income (per SD)"] = fit_adj(d, ["income_z"])["income_z"]

    # --- Culture/environment bucket ---
    for col, lab in [
        ("screentime_z", "Screen time (per SD)"),
        ("fam_conflict_z", "Family conflict (per SD)"),
    ]:
        d = df[["y", col, "sex_f", "age_z"] + NUIS].dropna()
        if enough(d.y):
            out[lab] = fit_adj(d, [col])[col]

    # --- Neuroimaging bucket: + scanner + mean FD; QC already applied to FC ---
    for col, lab in NEURAL:
        d = df[["y", col, "sex_f", "age_z", "scanner", "mean_fd_z"] + NUIS].dropna()
        if enough(d.y):
            out[lab] = fit_adj(d, [col], imaging=True)[col]

    # --- Race bucket: ONE full-sample model, White reference ---
    d = df[["y", "Race", "sex_f", "age_z"] + NUIS].dropna()
    d = d[d.Race.isin([RACE_REF] + RACE_LVES)]
    if enough(d.y):
        dummies = (
            pd.get_dummies(d.Race)
            .reindex(columns=RACE_LVES, fill_value=0)
            .astype(float)
        )
        dd = pd.concat([d[["y", "sex_f", "age_z"] + NUIS], dummies], axis=1)
        fit = fit_adj(dd, RACE_LVES)
        for lvl in RACE_LVES:
            # blank out separation: <10 positives in that race group
            if ((d.Race == lvl) & (d.y == 1)).sum() < 10:
                fit[lvl] = (np.nan, np.nan)
            out[f"Race: {lvl} vs {RACE_REF}"] = fit[lvl]
    return out


def build_specs(P, cache, constructs=CONSTRUCTS, informants=INFORMANTS,
                statuses=STATUSES, thresholds=THRESH):
    """Fit the full specification grid; returns the per-spec results table."""
    rows = []
    for con, conlab in constructs:
        for inf in informants:
            for status in statuses:
                for subthr in thresholds:
                    stat = construct_status(cache, con, status, inf, subthr, "phobia_in")
                    df = outcome_frame(P, stat)
                    if df is None:
                        continue
                    spec = dict(
                        construct=con,
                        construct_label=conlab,
                        informant=inf,
                        status=status,
                        threshold="full",
                    )
                    for lab, (orr, p) in fit_spec(df).items():
                        rows.append(
                            {
                                **spec,
                                "bucket": bucket_of(lab),
                                "predictor": lab,
                                "OR": orr,
                                "p": p,
                            }
                        )
    res = pd.DataFrame(rows)
    res["sig"] = res.p < 0.05
    res["logor"] = np.log(res.OR)
    return res


def summarize_specs(res):
    """Per predictor x construct: OR range, sign-flip, significance share, and eta^2."""
    summ = []
    for (bucket, con, conlab, pred), sub in res.groupby(
        ["bucket", "construct", "construct_label", "predictor"]
    ):
        sub = sub.dropna(subset=["OR"])
        if sub.empty:
            continue
        flips = (sub.OR.min() < 1) and (sub.OR.max() > 1)
        row = {
            "bucket": bucket,
            "construct": con,
            "construct_label": conlab,
            "predictor": pred,
            "n_specs": len(sub),
            "OR_min": sub.OR.min(),
            "OR_max": sub.OR.max(),
            "pct_sig": 100 * sub.sig.mean(),
            "sign_flip": flips,
            "any_sig": sub.sig.any(),
            "all_sig": sub.sig.all(),
        }
        row.update(eta2(sub))
        summ.append(row)
    return pd.DataFrame(summ)

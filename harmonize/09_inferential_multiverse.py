#!/usr/bin/env python3
"""Inferential multiverse: predictors are read from the consolidated phenotype cache.

The required predictor columns (SOURCES) and their recoding into analysis variables
live in abcd_ksads.predictors; this script reads the cached diagnosis outcomes and
fits the specification grid.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.multiverse import (
    build_primitive_cache,
    construct_status,
    BASE_SES,
)
from abcd_ksads.predictors import (
    load_predictors,
    RACE_REF,
    RACE_LVES,
)

CONSTRUCTS = [
    ("suicidality", "Suicidality"),
    ("eating", "Eating disorders"),
    ("depression", "Depression"),
    ("anxiety", "Anxiety"),
    ("any-disorder", "Any disorder"),
    ("ADHD", "ADHD"),
]
INFORMANTS = ["parent", "either"]  # caregiver / either, as in the literature
STATUSES = ["current", "ever_met"]
THRESH = [False, True]


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
            if np.isfinite(orr) and 0.02 <= orr <= 50:
                out[t] = (orr, p)
    except Exception:
        pass
    return out


def main():
    cw = build_crosswalk()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()
    cache = build_primitive_cache(base, cw)
    P = load_predictors()

    def enough(yv, mask=None):
        """outcome usable: >=100 rows, >=10 positives (in subgroup if given)."""
        if len(yv) < 100 or yv.sum() < 10:
            return False
        if mask is not None and (mask & (yv == 1)).sum() < 10:
            return False
        return True

    rows = []
    for con, conlab in CONSTRUCTS:
        for inf in INFORMANTS:
            for status in STATUSES:
                for subthr in THRESH:
                    stat = construct_status(
                        cache, con, status, inf, subthr, "phobia_in"
                    )
                    if stat is None or len(stat) == 0:
                        continue
                    y = stat.map(
                        {
                            "positive": 1,
                            "administered_negative": 0,
                            "not_administered": np.nan,
                        }
                    )
                    df = P.join(y.rename("y"), how="inner").dropna(subset=["y"])
                    spec = dict(
                        construct=con,
                        construct_label=conlab,
                        informant=inf,
                        status=status,
                        threshold="with_subthreshold" if subthr else "full",
                    )
                    out = {}  # predictor label -> (OR, p)

                    NUIS = ["site", "family_id"]

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
                    for col, lab in [
                        ("fc_dmn_within_z", "DMN within-network FC (per SD)"),
                        ("fc_sal_within_z", "Salience within-network FC (per SD)"),
                        ("fc_fpn_within_z", "FPN within-network FC (per SD)"),
                        ("fc_dmn_salience_z", "DMN-salience FC (per SD)"),
                        ("fc_dmn_fpn_z", "DMN-FPN FC (per SD)"),
                        ("fc_sal_fpn_z", "Salience-FPN FC (per SD)"),
                    ]:
                        d = df[
                            ["y", col, "sex_f", "age_z", "scanner", "mean_fd_z"] + NUIS
                        ].dropna()
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
                        dd = pd.concat(
                            [d[["y", "sex_f", "age_z"] + NUIS], dummies], axis=1
                        )
                        fit = fit_adj(dd, RACE_LVES)
                        for lvl in RACE_LVES:
                            # blank out separation: <10 positives in that race group
                            if ((d.Race == lvl) & (d.y == 1)).sum() < 10:
                                fit[lvl] = (np.nan, np.nan)
                            out[f"Race: {lvl} vs {RACE_REF}"] = fit[lvl]

                    for lab, (orr, p) in out.items():
                        bucket = (
                            "Sex"
                            if lab.startswith("Female")
                            else "Income"
                            if lab.startswith("Income")
                            else "Race/ethnicity"
                            if lab.startswith("Race")
                            else "Neuroimaging"
                            if "FC" in lab
                            else "Culture/environment"
                        )
                        rows.append(
                            {
                                **spec,
                                "bucket": bucket,
                                "predictor": lab,
                                "OR": orr,
                                "p": p,
                            }
                        )

    res = pd.DataFrame(rows)
    res["sig"] = res.p < 0.05
    res["logor"] = np.log(res.OR)
    res.to_csv(config.DERIV / "inferential_specs.csv", index=False)

    # per predictor x construct summary
    def eta2(sub):
        # variance of logOR attributable to each axis (one-way eta^2)
        out = {}
        tot = ((sub.logor - sub.logor.mean()) ** 2).sum()
        for ax in ["status", "informant", "threshold"]:
            ss = sum(
                len(g) * (g.logor.mean() - sub.logor.mean()) ** 2
                for _, g in sub.groupby(ax)
            )
            out[f"eta2_{ax}"] = ss / tot if tot > 0 else np.nan
        return out

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
    S = pd.DataFrame(summ)
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
    for ax in ["status", "informant", "threshold"]:
        print(f"  {ax:11}: {S[f'eta2_{ax}'].mean():.2f}")
    print(
        f"\nWrote inferential_specs.csv ({len(res)} rows) and inferential_summary.csv ({n_pairs} pairs)"
    )


if __name__ == "__main__":
    main()

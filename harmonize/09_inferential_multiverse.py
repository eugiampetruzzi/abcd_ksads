#!/usr/bin/env python3
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

DERIV = config.DERIV
BASE = config.ABCD_70

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
RACE_REF = "White"

BASELINE_EVENT = "baseline_year_1_arm_1"


def _nda_to_sub(s):
    return s.str.replace("NDAR_INV", "sub-", regex=False)


def load_culture_env():
    """3 release-5.1 culture/environment predictors at baseline, IDs mapped to sub-XXXX."""
    ce = BASE / "culture-environment"
    nt = BASE / "novel-technologies"
    # screen time: total of weekday + weekend item hours
    st = pd.read_csv(nt / "nt_y_st.csv")
    st = st[st.eventname == BASELINE_EVENT]
    sc = [
        c
        for c in st.columns
        if c.startswith("screen") and ("_wkdy_y" in c or "_wknd_y" in c)
    ]
    st["screentime"] = (
        st[sc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    )
    fes = pd.read_csv(ce / "ce_y_fes.csv")
    fes = fes[fes.eventname == BASELINE_EVENT][["src_subject_id", "fes_y_ss_fc"]]
    dm = pd.read_csv(ce / "ce_y_dm.csv")
    dm = dm[dm.eventname == BASELINE_EVENT][["src_subject_id", "dim_y_ss_mean"]]
    out = (
        st[["src_subject_id", "screentime"]]
        .merge(fes, on="src_subject_id", how="outer")
        .merge(dm, on="src_subject_id", how="outer")
        .rename(
            columns={"fes_y_ss_fc": "fam_conflict", "dim_y_ss_mean": "discrimination"}
        )
    )
    out["participant_id"] = _nda_to_sub(out.src_subject_id)
    return out.set_index("participant_id")[
        ["screentime", "fam_conflict", "discrimination"]
    ]


def load_imaging():
    """Two commonly-used resting-state network FC measures (release 5.1, baseline)."""
    f = BASE / "imaging" / "mri_y_rsfmr_cor_gp_gp.csv"
    cols = [
        "src_subject_id",
        "eventname",
        "rsfmri_c_ngd_dt_ngd_dt",
        "rsfmri_c_ngd_sa_ngd_sa",
        "rsfmri_c_ngd_fo_ngd_fo",
        "rsfmri_c_ngd_dt_ngd_sa",
        "rsfmri_c_ngd_dt_ngd_fo",
        "rsfmri_c_ngd_sa_ngd_fo",
    ]
    d = pd.read_csv(f, usecols=cols)
    d = d[d.eventname == BASELINE_EVENT].rename(
        columns={
            "rsfmri_c_ngd_dt_ngd_dt": "fc_dmn_within",
            "rsfmri_c_ngd_sa_ngd_sa": "fc_sal_within",
            "rsfmri_c_ngd_fo_ngd_fo": "fc_fpn_within",
            "rsfmri_c_ngd_dt_ngd_sa": "fc_dmn_salience",
            "rsfmri_c_ngd_dt_ngd_fo": "fc_dmn_fpn",
            "rsfmri_c_ngd_sa_ngd_fo": "fc_sal_fpn",
        }
    )
    d["participant_id"] = _nda_to_sub(d.src_subject_id)
    return d.set_index("participant_id")[
        [
            "fc_dmn_within",
            "fc_sal_within",
            "fc_fpn_within",
            "fc_dmn_salience",
            "fc_dmn_fpn",
            "fc_sal_fpn",
        ]
    ]


FC_COLS = [
    "fc_dmn_within",
    "fc_sal_within",
    "fc_fpn_within",
    "fc_dmn_salience",
    "fc_dmn_fpn",
    "fc_sal_fpn",
]


def load_qc_nuisance():
    """Nuisance covariates and rsfMRI QC: site, scanner, family from the covariate file;
    the ABCD rsfMRI inclusion flag from the imaging QC table. fc_qc_pass marks the
    ABCD-standard RSFC inclusion (imgincl==1, mean FD<0.5 mm, >=375 frames retained)."""
    cov = pd.read_excel(BASE / "5_covariates_extended.xlsx")[
        [
            "sub_ID",
            "study_site_baseline",
            "scanner_manufacturer_baseline",
            "family_id",
            "rest_mean_FD_baseline",
            "rest_total_frames_post_scrubbing_baseline",
        ]
    ].rename(
        columns={
            "sub_ID": "participant_id",
            "study_site_baseline": "site",
            "scanner_manufacturer_baseline": "scanner",
            "rest_mean_FD_baseline": "mean_fd",
            "rest_total_frames_post_scrubbing_baseline": "frames_kept",
        }
    )
    inc = pd.read_csv(
        BASE / "imaging" / "mri_y_qc_incl.csv",
        usecols=["src_subject_id", "eventname", "imgincl_rsfmri_include"],
    )
    inc = inc[inc.eventname == BASELINE_EVENT]
    inc["participant_id"] = _nda_to_sub(inc.src_subject_id)
    cov = cov.merge(
        inc[["participant_id", "imgincl_rsfmri_include"]],
        on="participant_id",
        how="left",
    )
    fd = pd.to_numeric(cov.mean_fd, errors="coerce")
    fr = pd.to_numeric(cov.frames_kept, errors="coerce")
    cov["fc_qc_pass"] = (cov.imgincl_rsfmri_include == 1) & (fd < 0.5) & (fr >= 375)
    return cov.set_index("participant_id")


def load_predictors():
    ela = pd.read_excel(BASE / "4_ELA_final.xlsx")[["sub_ID", "interview_age", "sex"]]
    cov = pd.read_excel(BASE / "5_covariates_extended.xlsx")[
        ["sub_ID", "Race", "Income"]
    ]
    d = ela.merge(cov, on="sub_ID", how="outer").rename(
        columns={"sub_ID": "participant_id"}
    )
    d["sex_f"] = d.sex.map({"F": 1, "M": 0})
    d["age_z"] = (d.interview_age - d.interview_age.mean()) / d.interview_age.std()
    d["income_z"] = (d.Income - d.Income.mean()) / d.Income.std()
    d = d.set_index("participant_id")
    d = d.join(load_culture_env(), how="left").join(load_imaging(), how="left")
    d = d.join(
        load_qc_nuisance()[["site", "scanner", "family_id", "mean_fd", "fc_qc_pass"]],
        how="left",
    )
    # apply ABCD-standard RSFC QC: blank FC for participants failing inclusion/motion/frames
    d.loc[not d.fc_qc_pass, FC_COLS] = np.nan
    fdq = pd.to_numeric(d.loc[not d.fc_qc_pass, "mean_fd"], errors="coerce")
    d["mean_fd_z"] = (
        pd.to_numeric(d.mean_fd, errors="coerce") - fdq.mean()
    ) / fdq.std()
    for col in ["screentime", "fam_conflict", "discrimination"] + FC_COLS:
        d[col + "_z"] = (d[col] - d[col].mean()) / d[col].std()
    return d


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

    RACE_LVES = ["Black/AA", "Hispanic", "Asian", "Other/Multiracial"]

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
                        ("discrimination_z", "Discrimination (per SD)"),
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

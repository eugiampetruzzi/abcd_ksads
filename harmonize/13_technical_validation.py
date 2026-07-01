#!/usr/bin/env python3
"""Technical validation: correctness vs raw counts, face validity vs CDC, concordance."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import cohen_kappa_score

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk, build_caseness

# CDC NSCH, US, ages 3-17, current diagnosed, PARENT-REPORTED PROVIDER DIAGNOSIS
# (a different instrument from a structured-interview current diagnosis).
CDC = {
    "Anxiety": ("11%", "current diagnosed anxiety, NSCH 2022-23"),
    "Depression": (
        "3.2%",
        "current diagnosed depression, NSCH 2016 (apples-to-apples current)",
    ),
    "Behavior (ODD/Conduct)": (
        "8%",
        "current diagnosed behavior disorders, NSCH 2022-23",
    ),
}
CDC_ANY_EVER = (
    "21%",
    "ever diagnosed any mental/emotional/behavioral condition, NSCH 2021",
)


def main():
    log = []

    def P(*a):
        s = " ".join(map(str, a))
        print(s)
        log.append(s)

    cw = build_crosswalk()
    res = pd.read_parquet(
        config.DERIV / "ksads_resolved_long.parquet",
        columns=["participant_id", "session_id", "variable", "resolved"],
    )
    for c in ["session_id", "variable", "resolved"]:
        res[c] = res[c].astype(str)
    res = res.merge(
        cw[["variable", "category", "informant"]], on="variable", how="left"
    )

    # ---- Check 1: correctness (resolved positives == raw value-1) ----
    P("=" * 70)
    P("CHECK 1  Correctness: resolved positives == raw ABCD 1-counts")
    P("=" * 70)
    # raw value-1 counts, read from the consolidated cache (values are stored as
    # floats e.g. "1.0", so count numerically rather than string-matching "1").
    source = config.RAW_CACHE / "phenotype.parquet"
    available = set(pq.ParquetFile(source).schema.names)
    dx_vars = [v for v in cw.variable.unique() if v in available]
    raw = pd.read_parquet(source, columns=dx_vars)
    raw_one = {
        v: int((pd.to_numeric(raw[v].astype("object"), errors="coerce") == 1).sum())
        for v in dx_vars
    }
    res_pos = res[res.resolved == "positive"].groupby("variable").size().to_dict()
    cw_cat = cw.set_index("variable")["category"].to_dict()
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
    cdf = pd.DataFrame(corr)
    tot_r, tot_w = int(cdf.n_resolved_positive.sum()), int(cdf.n_raw_value1.sum())
    cdf.to_csv(config.DERIV / "validation_correctness.csv", index=False)
    P(cdf.to_string(index=False))
    P(
        f"\n  TOTAL resolved positives = {tot_r:,}; raw value-1 = {tot_w:,}; "
        f"all categories match: {bool(cdf.match.all())}"
    )

    # ---- Check 2: face validity vs CDC ----
    P("\n" + "=" * 70)
    P("CHECK 2  Face validity vs CDC US current-diagnosed rates")
    P("=" * 70)
    base = res[res.session_id == "ses-00A"].copy()
    cur = build_caseness(
        base, cw, status_set="current", include_subthreshold=False, informant="parent"
    )

    def prev(cats):
        c = cur[cur.category.isin(cats)]
        piv = c.pivot_table(
            index="participant_id", columns="category", values="status", aggfunc="first"
        )
        pos = (piv == "positive").any(axis=1)
        adm = (piv.notna() & (piv != "not_administered")).any(axis=1)
        return 100 * pos.sum() / adm.sum()

    fv = []
    for cat, (cdc, desc) in CDC.items():
        if cat == "Behavior (ODD/Conduct)":
            ours = prev(["ODD", "Conduct"])
        else:
            ours = prev([cat])
        fv.append(
            {
                "construct": cat,
                "default_prevalence_pct": round(ours, 2),
                "cdc_us_current": cdc,
                "cdc_note": desc,
            }
        )
    # any-disorder ever-met vs CDC ever
    eve = build_caseness(
        base, cw, status_set="ever_met", include_subthreshold=False, informant="parent"
    )
    cure = eve  # ever-met at baseline
    anycats = [
        "Depression",
        "Anxiety",
        "ADHD",
        "ODD",
        "Conduct",
        "Bipolar",
        "DMDD",
        "OCD",
        "PTSD",
        "Autism",
        "Tic",
        "Eating",
        "Psychosis",
    ]
    c = cure[cure.category.isin(anycats)]
    piv = c.pivot_table(
        index="participant_id", columns="category", values="status", aggfunc="first"
    )
    anyp = (
        100
        * (piv == "positive").any(axis=1).sum()
        / (piv.notna() & (piv != "not_administered")).any(axis=1).sum()
    )
    fv.append(
        {
            "construct": "Any disorder (ever-met)",
            "default_prevalence_pct": round(anyp, 2),
            "cdc_us_current": CDC_ANY_EVER[0],
            "cdc_note": CDC_ANY_EVER[1],
        }
    )
    fdf = pd.DataFrame(fv)
    fdf.to_csv(config.DERIV / "validation_facevalidity.csv", index=False)
    P(fdf.to_string(index=False))

    # ---- Check 3: parent-youth concordance ----
    P("\n" + "=" * 70)
    P("CHECK 3  Parent-youth concordance (Cohen's kappa), baseline, current")
    P("=" * 70)
    cp = build_caseness(
        base, cw, status_set="current", include_subthreshold=False, informant="parent"
    )
    cy = build_caseness(
        base, cw, status_set="current", include_subthreshold=False, informant="youth"
    )
    con = []
    for cat in ["Depression", "Anxiety"]:
        p = cp[cp.category == cat].set_index("participant_id")["status"]
        y = cy[cy.category == cat].set_index("participant_id")["status"]
        m = pd.DataFrame({"p": p, "y": y}).dropna()
        m = m[(m.p != "not_administered") & (m.y != "not_administered")]
        pb = (m.p == "positive").astype(int)
        yb = (m.y == "positive").astype(int)
        k = cohen_kappa_score(pb, yb) if len(m) > 50 else np.nan
        con.append(
            {
                "category": cat,
                "n_both_assessed": len(m),
                "parent_pos_pct": round(100 * pb.mean(), 2),
                "youth_pos_pct": round(100 * yb.mean(), 2),
                "cohen_kappa": round(k, 3),
                "both_positive": int(((pb == 1) & (yb == 1)).sum()),
            }
        )
    kdf = pd.DataFrame(con)
    kdf.to_csv(config.DERIV / "validation_concordance.csv", index=False)
    P(kdf.to_string(index=False))

    (config.DERIV / "technical_validation_report.txt").write_text("\n".join(log))
    P("\nWrote validation_*.csv and technical_validation_report.txt")


if __name__ == "__main__":
    main()

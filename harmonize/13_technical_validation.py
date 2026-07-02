#!/usr/bin/env python3
"""Technical validation: correctness vs raw counts, face validity vs CDC, concordance.
The check logic lives in abcd_ksads.technical_validation."""

import pandas as pd
import pyarrow.parquet as pq

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_caseness, build_crosswalk
from abcd_ksads.technical_validation import (
    caseness_prevalence,
    concordance_kappa,
    correctness_by_category,
)

# CDC NSCH, US, ages 3-17, current diagnosed, PARENT-REPORTED PROVIDER DIAGNOSIS
# (a different instrument from a structured-interview current diagnosis).
CDC = {
    "Anxiety": ("11%", "current diagnosed anxiety, NSCH 2022-23"),
    "Depression": ("3.2%", "current diagnosed depression, NSCH 2016 (apples-to-apples current)"),
    "Behavior (ODD/Conduct)": ("8%", "current diagnosed behavior disorders, NSCH 2022-23"),
}
CDC_ANY_EVER = ("21%", "ever diagnosed any mental/emotional/behavioral condition, NSCH 2021")
ANY_CATS = [
    "Depression", "Anxiety", "ADHD", "ODD", "Conduct", "Bipolar", "DMDD",
    "OCD", "PTSD", "Autism", "Tic", "Eating", "Psychosis",
]


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
    res = res.merge(cw[["variable", "category", "informant"]], on="variable", how="left")

    # ---- Check 1: correctness (resolved positives == raw value-1) ----
    P("=" * 70)
    P("CHECK 1  Correctness: resolved positives == raw ABCD 1-counts")
    P("=" * 70)
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
    cdf = correctness_by_category(res_pos, raw_one, cw_cat)
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
    cur = build_caseness(base, cw, status_set="current", include_subthreshold=False,
                         informant="parent")
    fv = []
    for cat, (cdc, desc) in CDC.items():
        cats = ["ODD", "Conduct"] if cat == "Behavior (ODD/Conduct)" else [cat]
        fv.append(
            {
                "construct": cat,
                "default_prevalence_pct": round(caseness_prevalence(cur, cats), 2),
                "cdc_us_current": cdc,
                "cdc_note": desc,
            }
        )
    eve = build_caseness(base, cw, status_set="ever_met", include_subthreshold=False,
                         informant="parent")
    fv.append(
        {
            "construct": "Any disorder (ever-met)",
            "default_prevalence_pct": round(caseness_prevalence(eve, ANY_CATS), 2),
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
    cp = build_caseness(base, cw, status_set="current", include_subthreshold=False,
                        informant="parent")
    cy = build_caseness(base, cw, status_set="current", include_subthreshold=False,
                        informant="youth")
    kdf = pd.DataFrame([concordance_kappa(cp, cy, cat) for cat in ["Depression", "Anxiety"]])
    kdf.to_csv(config.DERIV / "validation_concordance.csv", index=False)
    P(kdf.to_string(index=False))

    (config.DERIV / "technical_validation_report.txt").write_text("\n".join(log))
    P("\nWrote validation_*.csv and technical_validation_report.txt")


if __name__ == "__main__":
    main()

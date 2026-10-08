#!/usr/bin/env python3
"""Age-by-correlate interactions in the baseline correlate models."""

import warnings

import numpy as np
import pandas as pd

from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.inferential import (CONSTRUCTS, INFORMANTS, NEURAL, NUIS, RACE_LEVELS, RACE_REF,
                                    STATUSES, enough, fit_adj, outcome_frame)
from abcd_ksads.multiverse import BASE_SES, build_primitive_cache, construct_status
from abcd_ksads.predictors import load_predictors

warnings.filterwarnings("ignore")
SIMPLE = [("sex_f", "Female (vs male)", "Sex"), ("income_z", "Income (per SD)", "Income"),
          ("screentime_z", "Screen time (per SD)", "Culture/environment"),
          ("fam_conflict_z", "Family conflict (per SD)", "Culture/environment")]


def interaction_fits(df):
    out = []
    for col, lab, bucket in SIMPLE + [(c, l, "Neuroimaging") for c, l in NEURAL]:
        img = bucket == "Neuroimaging"
        extra = ["scanner", "mean_fd_z"] if img else []
        d = df[list(dict.fromkeys(["y", col, "sex_f", "age_z"] + extra + NUIS))].dropna().copy()
        if not enough(d.y):
            continue
        d["xage"] = d[col] * d.age_z
        orr, p = fit_adj(d, [col, "xage"], imaging=img)["xage"]
        out.append((bucket, lab, orr, p))
    d = df[["y", "Race", "sex_f", "age_z"] + NUIS].dropna()
    d = d[d.Race.isin([RACE_REF] + RACE_LEVELS)]
    if enough(d.y):
        dum = pd.get_dummies(d.Race).reindex(columns=RACE_LEVELS, fill_value=0).astype(float)
        xs = dum.mul(d.age_z, axis=0).add_suffix("_xage")
        dd = pd.concat([d[["y", "sex_f", "age_z"] + NUIS], dum, xs], axis=1)
        fit = fit_adj(dd, RACE_LEVELS + list(xs.columns))
        for lvl in RACE_LEVELS:
            if ((d.Race == lvl) & (d.y == 1)).sum() >= 10:
                out.append(("Race/ethnicity", f"Race: {lvl} vs {RACE_REF}", *fit[f"{lvl}_xage"]))
    return out


def main():
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    cache = build_primitive_cache(resolved[resolved.session_id == BASE_SES].copy(), build_crosswalk())
    P = load_predictors()
    rows = []
    for con, conlab in CONSTRUCTS:
        for inf in INFORMANTS:
            for status in STATUSES:
                df = outcome_frame(P, construct_status(cache, con, status, inf, False, "phobia_in"))
                if df is None:
                    continue
                for bucket, lab, orr, p in interaction_fits(df):
                    rows.append(dict(construct=con, construct_label=conlab, informant=inf, status=status,
                                     bucket=bucket, predictor=lab, OR=orr, p=p))
    res = pd.DataFrame(rows).dropna(subset=["OR"])
    res["sig"] = res.p < 0.05
    res.to_csv(config.DERIV / "age_moderation_specs.csv", index=False)
    S = res.groupby(["bucket", "predictor"]).agg(specs=("sig", "size"), pct_sig=("sig", "mean")).reset_index()
    S["pct_sig"] = (100 * S.pct_sig).round(1)
    S = S.sort_values("pct_sig", ascending=False)
    S.to_csv(config.DERIV / "age_moderation.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import pandas as pd
import dotenv

# Load environment variables
dotenv.load_dotenv()

ROOT = os.environ.get("ABCD_70", ".")
EXT = "tsv"
SEP = "\t" if EXT == "tsv" else ","


def load(name, cols):
    path = os.path.join(ROOT, "rawdata/phenotype", f"{name}.{EXT}")
    have = pd.read_csv(path, sep=SEP, nrows=0).columns
    use = [c for c in cols if c in have]
    missing = [c for c in cols if c not in have]
    if missing:
        print(f"  warning: {name} is missing {missing}")
    return pd.read_csv(path, sep=SEP, usecols=use)


stc = load(
    "ab_g_stc",
    [
        "participant_id",
        "ab_g_stc__cohort_sex__assigned",
        "ab_g_stc__cohort_ethnrace__leg",
        "ab_g_stc__design_id__fam",
    ],
)

dyn = load(
    "ab_g_dyn",
    ["participant_id", "session_id", "ab_g_dyn__visit_age", "ab_g_dyn__design_site"],
)

demo = load(
    "ab_p_demo", ["participant_id", "session_id", "ab_p_demo__income__hhold_001"]
)

demo_keys = [k for k in ("participant_id", "session_id") if k in demo.columns]
out = dyn.merge(stc, on="participant_id", how="left").merge(
    demo, on=demo_keys, how="left"
)

out = out.rename(
    columns={
        "ab_g_stc__cohort_sex__assigned": "sex_assigned",
        "ab_g_stc__cohort_ethnrace__leg": "race_ethnicity",
        "ab_g_stc__design_id__fam": "family_id",
        "ab_g_dyn__visit_age": "interview_age",
        "ab_g_dyn__design_site": "site",
        "ab_p_demo__income__hhold_001": "income",
    }
)

outfile = os.path.join(ROOT, "subject_demographics.tsv")
out.to_csv(outfile, index=False)
print(
    f"merged {len(out):,} rows ({out.participant_id.nunique():,} participants) -> covariates_merged.csv"
)
print(out.head().to_string(index=False))

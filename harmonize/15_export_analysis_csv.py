#!/usr/bin/env python3
"""Export analysis-ready CSVs from the harmonized BIDS dataset and cached KSADS ages.

Per-session interview age is read from the consolidated cache (phenotype.parquet);
interview date is not available (the dtt columns were removed for deidentification).
"""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.export import AGE_COLS, interview_ages

DSET = config.DATASET
PHENO = DSET / "phenotype"
CSV = DSET / "csv"


def main():
    CSV.mkdir(parents=True, exist_ok=True)

    # ---- per-session interview age (years -> months), youngest wins, from cache ----
    wide = pd.read_parquet(
        config.RAW_CACHE / "phenotype.parquet",
        columns=["participant_id", "session_id"] + list(AGE_COLS.values()),
    )
    sessions = interview_ages(wide)
    sessions.to_csv(CSV / "sessions.csv", index=False)

    age_map = sessions.set_index(["participant_id", "session_id"])["interview_age"]

    # ---- category caseness CSVs (+ interview_age) ----
    for name in ("ksads_categories_current", "ksads_categories_evermet"):
        d = pd.read_csv(PHENO / (name + ".tsv"), sep="\t")
        d["interview_age"] = d.set_index(["participant_id", "session_id"]).index.map(
            age_map
        )
        front = ["participant_id", "session_id", "informant", "interview_age"]
        d = d[front + [c for c in d.columns if c not in front]]
        d.to_csv(CSV / (name + ".csv"), index=False)

    # ---- resolved layer + metadata -> CSV ----
    # convert the already-written resolved tsv.gz to csv.gz (lossless)
    rl = pd.read_csv(PHENO / "ksads_diagnosis_resolved.tsv.gz", sep="\t", dtype=str)
    rl.to_csv(
        CSV / "ksads_diagnosis_resolved.csv.gz", index=False, compression="gzip"
    )
    for t in ("ksads_administration_calendar", "ksads_category_crosswalk"):
        pd.read_csv(PHENO / (t + ".tsv"), sep="\t").to_csv(
            CSV / (t + ".csv"), index=False
        )

    # ---- participants.csv (+ src_subject_id, subjectkey + sex placeholders) ----
    parts = pd.read_csv(DSET / "participants.tsv", sep="\t")
    parts["src_subject_id"] = parts["participant_id"].str.replace(
        "sub-", "", regex=False
    )
    parts["subjectkey"] = ""  # NDA GUID - merge from the ABCD GUID crosswalk
    parts["sex"] = ""  # merge from the full ABCD demographics by participant_id
    parts = parts[
        [
            "participant_id",
            "src_subject_id",
            "subjectkey",
            "sex",
            "n_waves_parent_kSADS",
            "n_waves_youth_kSADS",
        ]
    ]
    parts.to_csv(CSV / "participants.csv", index=False)

    readme = (
        "Analysis-ready CSVs - harmonized ABCD 7.0 KSADS-COMP\n"
        "===================================================\n\n"
        "Files (key on participant_id; even/diagnostic waves for caseness):\n"
        "  ksads_categories_current.csv   caseness, current status, informant column (parent/youth/either)\n"
        "  ksads_categories_evermet.csv   caseness, ever-met status\n"
        "  ksads_diagnosis_resolved.csv.gz  atomic layer: participant x session x diagnosis, resolved state + version\n"
        "  sessions.csv                   participant x session: interview_age (months), ksads_version\n"
        "  ksads_administration_calendar.csv, ksads_category_crosswalk.csv  metadata\n"
        "  participants.csv               one row per participant\n\n"
        "Caseness states are positive / administered_negative / not_administered; compute prevalence over\n"
        "the administered denominator only (never count not_administered as a negative).\n\n"
        "Covariates that are NOT included and must be merged from your own ABCD data by participant_id,\n"
        "because they are not in the KSADS release and are access-restricted:\n"
        "  sex         -> from the full ABCD demographics\n"
        "  subjectkey  -> the NDA GUID, from the ABCD GUID crosswalk (required for NDA submission)\n"
        "interview_age (months) is included from the KSADS interview metadata. interview_date is not\n"
        "available (the date/time fields were removed from the release for deidentification).\n"
    )
    (CSV / "README.txt").write_text(readme)

    print("Analysis-ready CSVs written to:")
    print(f"  {CSV}\n")
    for f in sorted(CSV.iterdir()):
        print(f"  {f.name:42} {f.stat().st_size / 1024:9.1f} KB")
    age_cov = (
        pd.read_csv(CSV / "ksads_categories_current.csv", usecols=["interview_age"])
        .interview_age.notna()
        .mean()
    )
    print(
        f"\n  sessions: {len(sessions):,} | participants: {parts.participant_id.nunique():,} | "
        f"interview_age coverage in caseness: {age_cov * 100:.0f}%"
    )
    print(
        "  sex and subjectkey (GUID) are left blank - merge from your ABCD demographics / GUID crosswalk."
    )


if __name__ == "__main__":
    main()

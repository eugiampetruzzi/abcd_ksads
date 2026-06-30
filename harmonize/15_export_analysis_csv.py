#!/usr/bin/env python3
import os

import pandas as pd

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config
RAW = config.RAW_PHENOTYPE
DSET = config.DATASET
PHENO = os.path.join(DSET, "phenotype")
CSV = os.path.join(DSET, "csv")
os.makedirs(CSV, exist_ok=True)
DEMO = config.DEMOGRAPHICS
V1 = {"ses-00A", "ses-01A", "ses-02A"}


def main():
    # ---- per-session age (years -> months) and interview date, parent then youth ----
    def age_date(pref, mod="dep"):
        f = os.path.join(RAW, f"{pref}_ksads__{mod}.tsv")
        cols = ["participant_id", "session_id", f"{pref}_ksads__{mod}_age", f"{pref}_ksads__{mod}_dtt"]
        d = pd.read_csv(f, sep="\t", usecols=cols, dtype=str)
        d = d.rename(columns={cols[2]: "age_yr", cols[3]: "dtt"})
        return d
    ad = pd.concat([age_date("mh_p"), age_date("mh_y")], ignore_index=True)
    ad["age_yr"] = pd.to_numeric(ad["age_yr"], errors="coerce")
    ad = ad.dropna(subset=["age_yr"]).sort_values("age_yr")
    ad = ad.drop_duplicates(["participant_id", "session_id"], keep="first")
    ad["interview_age"] = (ad["age_yr"] * 12).round().astype("Int64")
    ad["interview_date"] = ad["dtt"].str.slice(0, 10)
    ad["ksads_version"] = ad["session_id"].apply(lambda s: "1.0" if s in V1 else "2.0")
    sessions = ad[["participant_id", "session_id", "interview_age", "interview_date", "ksads_version"]]
    sessions.to_csv(os.path.join(CSV, "sessions.csv"), index=False)

    age_map = ad.set_index(["participant_id", "session_id"])["interview_age"]

    # ---- category caseness CSVs (+ interview_age) ----
    for name in ("ksads_categories_current", "ksads_categories_evermet"):
        d = pd.read_csv(os.path.join(PHENO, name + ".tsv"), sep="\t")
        d["interview_age"] = d.set_index(["participant_id", "session_id"]).index.map(age_map)
        front = ["participant_id", "session_id", "informant", "interview_age"]
        d = d[front + [c for c in d.columns if c not in front]]
        d.to_csv(os.path.join(CSV, name + ".csv"), index=False)

    # ---- resolved layer + metadata -> CSV ----
    res = pd.read_parquet(os.path.join(os.path.dirname(HERE), "harmonize", "derivatives",
                          "ksads_resolved_versioned.parquet")) if False else None
    # convert the already-written resolved tsv.gz to csv.gz (lossless)
    rl = pd.read_csv(os.path.join(PHENO, "ksads_diagnosis_resolved.tsv.gz"), sep="\t", dtype=str)
    rl.to_csv(os.path.join(CSV, "ksads_diagnosis_resolved.csv.gz"), index=False, compression="gzip")
    for t in ("ksads_administration_calendar", "ksads_category_crosswalk"):
        pd.read_csv(os.path.join(PHENO, t + ".tsv"), sep="\t").to_csv(
            os.path.join(CSV, t + ".csv"), index=False)

    # ---- participants.csv (+ src_subject_id, subjectkey + sex placeholders) ----
    parts = pd.read_csv(os.path.join(DSET, "participants.tsv"), sep="\t")
    parts["src_subject_id"] = parts["participant_id"].str.replace("sub-", "", regex=False)
    parts["subjectkey"] = ""   # NDA GUID - merge from the ABCD GUID crosswalk
    parts["sex"] = ""          # merge from the full ABCD demographics by participant_id
    parts = parts[["participant_id", "src_subject_id", "subjectkey", "sex",
                   "n_waves_parent_kSADS", "n_waves_youth_kSADS"]]
    parts.to_csv(os.path.join(CSV, "participants.csv"), index=False)

    readme = (
        "Analysis-ready CSVs - harmonized ABCD 7.0 KSADS-COMP\n"
        "===================================================\n\n"
        "Files (key on participant_id; even/diagnostic waves for caseness):\n"
        "  ksads_categories_current.csv   caseness, current status, informant column (parent/youth/either)\n"
        "  ksads_categories_evermet.csv   caseness, ever-met status\n"
        "  ksads_diagnosis_resolved.csv.gz  atomic layer: participant x session x diagnosis, resolved state + version\n"
        "  sessions.csv                   participant x session: interview_age (months), interview_date, ksads_version\n"
        "  ksads_administration_calendar.csv, ksads_category_crosswalk.csv  metadata\n"
        "  participants.csv               one row per participant\n\n"
        "Caseness states are positive / administered_negative / not_administered; compute prevalence over\n"
        "the administered denominator only (never count not_administered as a negative).\n\n"
        "Two covariates are NOT included and must be merged from your own ABCD data by participant_id,\n"
        "because they are not in the KSADS release and are access-restricted:\n"
        "  sex         -> from the full ABCD demographics\n"
        "  subjectkey  -> the NDA GUID, from the ABCD GUID crosswalk (required for NDA submission)\n"
        "interview_age and interview_date are included (from the KSADS interview metadata).\n")
    open(os.path.join(CSV, "README.txt"), "w").write(readme)

    print("Analysis-ready CSVs written to:")
    print(f"  {CSV}\n")
    for f in sorted(os.listdir(CSV)):
        print(f"  {f:42} {os.path.getsize(os.path.join(CSV, f))/1024:9.1f} KB")
    age_cov = pd.read_csv(os.path.join(CSV, "ksads_categories_current.csv"),
                          usecols=["interview_age"]).interview_age.notna().mean()
    print(f"\n  sessions: {len(sessions):,} | participants: {parts.participant_id.nunique():,} | "
          f"interview_age coverage in caseness: {age_cov*100:.0f}%")
    print("  sex and subjectkey (GUID) are left blank - merge from your ABCD demographics / GUID crosswalk.")


if __name__ == "__main__":
    main()
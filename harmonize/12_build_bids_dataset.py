#!/usr/bin/env python3
import importlib.util
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
ROOT = os.path.join(os.path.dirname(HERE), "abcd_ksads_harmonized")
PHENO = os.path.join(ROOT, "phenotype")
os.makedirs(PHENO, exist_ok=True)


def _load(f):
    spec = importlib.util.spec_from_file_location(
        f[:-3].replace(".", "_"), os.path.join(HERE, f)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L3 = _load("03_category_crosswalk.py")
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]
ALL_SES = [
    "ses-00A",
    "ses-01A",
    "ses-02A",
    "ses-03A",
    "ses-04A",
    "ses-05A",
    "ses-06A",
    "ses-07A",
]
DISORDER_CATS = [
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


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    cw = L3.build_crosswalk()
    ver = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_versioned.parquet"))
    for c in [
        "session_id",
        "variable",
        "resolved",
        "informant",
        "module",
        "status_layer",
    ]:
        ver[c] = ver[c].astype(str)
    ver = ver.merge(
        cw[["variable", "category", "is_subthreshold"]], on="variable", how="left"
    )

    # ---- 1. resolved diagnosis layer (long; drop no_record = absent) ----
    res = ver[ver.resolved != "no_record"][
        [
            "participant_id",
            "session_id",
            "informant",
            "module",
            "category",
            "status_layer",
            "variable",
            "is_subthreshold",
            "resolved",
            "ksads_version",
            "version_valid",
        ]
    ].copy()
    res = res.sort_values(["participant_id", "session_id", "informant", "variable"])
    res.to_csv(
        os.path.join(PHENO, "ksads_diagnosis_resolved.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip",
    )

    # ---- 2. category caseness tables (wide; recommended default = parent) ----
    base = ver[ver.session_id.isin(EVEN)].copy()
    RANK = {"positive": 3, "administered_negative": 2, "not_administered": 1}
    INV = {3: "positive", 2: "administered_negative", 1: "not_administered"}

    def caseness_informant(status_set, informant):
        if informant == "either":
            cp = L3.build_caseness(
                base,
                cw,
                status_set=status_set,
                include_subthreshold=False,
                informant="parent",
            )
            cy = L3.build_caseness(
                base,
                cw,
                status_set=status_set,
                include_subthreshold=False,
                informant="youth",
            )
            c = pd.concat([cp, cy])
            c["rk"] = c.status.map(RANK)
            c = (
                c.groupby(["participant_id", "session_id", "category"])["rk"]
                .max()
                .reset_index()
            )
            c["status"] = c.rk.map(INV)
            return c[["participant_id", "session_id", "category", "status"]]
        return L3.build_caseness(
            base,
            cw,
            status_set=status_set,
            include_subthreshold=False,
            informant=informant,
        )

    def caseness_wide(status_set):
        frames = []
        for inf in ["parent", "youth", "either"]:
            c = caseness_informant(status_set, inf)
            c = c[c.category.isin(DISORDER_CATS)]
            wide = c.pivot_table(
                index=["participant_id", "session_id"],
                columns="category",
                values="status",
                aggfunc="first",
            ).reset_index()
            for cat in DISORDER_CATS:
                if cat not in wide:
                    wide[cat] = "not_administered"
                wide[cat] = wide[cat].fillna("not_administered")
            wide.insert(2, "informant", inf)
            frames.append(
                wide[["participant_id", "session_id", "informant"] + DISORDER_CATS]
            )
        return pd.concat(frames, ignore_index=True)

    cur = caseness_wide("current")
    eve = caseness_wide("ever_met")
    cur.to_csv(
        os.path.join(PHENO, "ksads_categories_current.tsv"), sep="\t", index=False
    )
    eve.to_csv(
        os.path.join(PHENO, "ksads_categories_evermet.tsv"), sep="\t", index=False
    )

    # ---- 3. metadata tables ----
    cal = pd.read_csv(os.path.join(DERIV, "ksads_administration_calendar.csv"))
    cal.to_csv(
        os.path.join(PHENO, "ksads_administration_calendar.tsv"), sep="\t", index=False
    )
    cw.to_csv(
        os.path.join(PHENO, "ksads_category_crosswalk.tsv"), sep="\t", index=False
    )

    # ---- 4. participants.tsv ----
    adm = res[res.resolved.isin(["positive", "administered_negative"])]
    nwav = (
        adm.groupby(["participant_id", "informant"])["session_id"]
        .nunique()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("parent", "youth"):
        if col not in nwav:
            nwav[col] = 0
    parts = nwav.rename(
        columns={"parent": "n_waves_parent_kSADS", "youth": "n_waves_youth_kSADS"}
    )
    parts = parts[["participant_id", "n_waves_parent_kSADS", "n_waves_youth_kSADS"]]
    parts.to_csv(os.path.join(ROOT, "participants.tsv"), sep="\t", index=False)

    # ---- 5. JSON data dictionaries ----
    RESOLVED_LEVELS = {
        "positive": "Met criteria (value 1)",
        "administered_negative": "Assessed and criteria not met (value 0)",
        "not_administered": "Module/wave not administered to this participant (value 555)",
    }
    write_json(
        os.path.join(PHENO, "ksads_diagnosis_resolved.json"),
        {
            "participant_id": {"Description": "ABCD participant identifier"},
            "session_id": {
                "Description": "ABCD session (wave); ses-00A baseline ... ses-07A 7-year"
            },
            "informant": {
                "Description": "Report source",
                "Levels": {"parent": "parent/caregiver", "youth": "youth self-report"},
            },
            "module": {"Description": "KSADS-COMP diagnostic module"},
            "category": {
                "Description": "Harmonized DSM category (see ksads_category_crosswalk)"
            },
            "status_layer": {
                "Description": "Episode layer",
                "Levels": {
                    "present": "current",
                    "past": "past episode",
                    "partial_remission": "partial remission",
                    "remission": "full remission",
                    "unknown": "unspecified",
                },
            },
            "variable": {"Description": "Original ABCD 7.0 diagnosis variable name"},
            "is_subthreshold": {
                "Description": "1 = 'other specified'/'unspecified' subthreshold diagnosis"
            },
            "resolved": {
                "Description": "Resolved administration state (the core correction; "
                "never collapse not_administered to a negative)",
                "Levels": RESOLVED_LEVELS,
            },
            "ksads_version": {
                "Description": "KSADS-COMP version, wave-determined",
                "Levels": {"1.0": "ses-00A/01A/02A", "2.0": "ses-03A onward"},
            },
            "version_valid": {
                "Description": "False if a 2.0-only diagnosis recorded under 1.0"
            },
        },
    )
    cat_dict = {
        "participant_id": {"Description": "ABCD participant identifier"},
        "session_id": {"Description": "ABCD session (even/diagnostic waves only)"},
        "informant": {
            "Description": "Report source / combination rule",
            "Levels": {
                "parent": "parent/caregiver report",
                "youth": "youth self-report (fewer modules; e.g., no ADHD/ODD)",
                "either": "positive if parent or youth is positive (recommended default)",
            },
        },
    }
    for cat in DISORDER_CATS:
        cat_dict[cat] = {
            "Description": f"{cat} caseness over the administered denominator",
            "Levels": RESOLVED_LEVELS,
        }
    write_json(
        os.path.join(PHENO, "ksads_categories_current.json"),
        {
            **cat_dict,
            "_config": {
                "status": "current (present only)",
                "informant": "parent, youth, and either (see informant column)",
                "threshold": "full (subthreshold excluded)",
                "anxiety": "includes specific phobia",
            },
        },
    )
    write_json(
        os.path.join(PHENO, "ksads_categories_evermet.json"),
        {
            **cat_dict,
            "_config": {
                "status": "ever-met (present|past|partial_remission)",
                "informant": "parent, youth, and either (see informant column)",
                "threshold": "full (subthreshold excluded)",
                "anxiety": "includes specific phobia",
            },
        },
    )
    write_json(
        os.path.join(PHENO, "ksads_administration_calendar.json"),
        {
            "informant": {"Description": "parent or youth"},
            "module": {"Description": "KSADS-COMP module"},
            "session_id": {"Description": "ABCD session"},
            "status": {
                "Description": "Administration state of the module at this wave",
                "Levels": {
                    "administered": "given to the cohort",
                    "not_administered": "present but all 555",
                    "absent": "module/variable not in the release at this wave",
                },
            },
            "n_administered": {
                "Description": "count of administered cells (positive + negative)"
            },
            "flag": {"Description": "added@/dropped_after@/intermittent annotations"},
        },
    )
    write_json(
        os.path.join(PHENO, "ksads_category_crosswalk.json"),
        {
            "variable": {"Description": "ABCD 7.0 diagnosis variable"},
            "informant": {"Description": "parent or youth"},
            "module": {"Description": "KSADS-COMP module"},
            "status_layer": {
                "Description": "present/past/partial_remission/remission/unknown"
            },
            "category": {"Description": "harmonized DSM category"},
            "broadband": {"Description": "broadband dimension(s), pipe-delimited"},
            "is_subthreshold": {
                "Description": "1 = other-specified/unspecified subthreshold"
            },
        },
    )
    write_json(
        os.path.join(ROOT, "participants.json"),
        {
            "participant_id": {"Description": "ABCD participant identifier"},
            "n_waves_parent_kSADS": {
                "Description": "number of waves with an administered parent KSADS diagnosis"
            },
            "n_waves_youth_kSADS": {
                "Description": "number of waves with an administered youth KSADS diagnosis"
            },
        },
    )

    # ---- 6. BIDS descriptors ----
    write_json(
        os.path.join(ROOT, "dataset_description.json"),
        {
            "Name": "Harmonized ABCD KSADS-COMP diagnostic dataset",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "abcd_ksads harmonization pipeline",
                    "Version": "1.0",
                    "Description": "4-layer resolver / calendar / crosswalk / version pipeline",
                }
            ],
            "SourceDatasets": [
                {
                    "Description": "ABCD Study 7.0 data release, KSADS-COMP "
                    "(mh_p_ksads / mh_y_ksads), via the NIMH Data Archive"
                }
            ],
            "HowToAcknowledge": "Access is restricted to credentialed ABCD/NDA users. "
            "Cite the harmonization Data Descriptor and the ABCD Study.",
        },
    )
    readme = (
        "Harmonized ABCD KSADS-COMP diagnostic dataset\n"
        "=============================================\n\n"
        "Analysis-ready diagnostic data derived from the ABCD Study 7.0 KSADS-COMP release.\n"
        "Every diagnosis cell carries an explicit administration state, so not-administered\n"
        "(555) is never silently treated as a negative. Prevalence must be computed over the\n"
        "administered denominator (resolved in {positive, administered_negative}).\n\n"
        "phenotype/\n"
        "  ksads_diagnosis_resolved.tsv.gz  atomic layer: participant x session x diagnosis,\n"
        "                                   resolved state + KSADS version (no_record rows omitted).\n"
        "  ksads_categories_current.tsv     ready-to-use caseness, current, parent, full threshold.\n"
        "  ksads_categories_evermet.tsv     ready-to-use caseness, ever-met, parent, full threshold.\n"
        "  ksads_administration_calendar.tsv  module x wave x informant administration map.\n"
        "  ksads_category_crosswalk.tsv     diagnosis-variable to DSM-category mapping.\n\n"
        "Other operationalizations (youth/either/both informant, +subthreshold, phobia-out) are\n"
        "reproducible from the resolved layer with the published harmonization engine.\n\n"
        "Data are access-restricted (NDA). This folder contains derived values for credentialed\n"
        "ABCD users only.\n"
    )
    open(os.path.join(ROOT, "README"), "w").write(readme)
    open(os.path.join(ROOT, "CHANGES"), "w").write(
        "1.0\n  - Initial harmonized release (ABCD 7.0).\n"
    )

    # ---- report ----
    print("Built harmonized BIDS dataset at:")
    print(f"  {ROOT}\n")
    for dp, _, fs in os.walk(ROOT):
        for f in sorted(fs):
            p = os.path.join(dp, f)
            kb = os.path.getsize(p) / 1024
            print(f"  {os.path.relpath(p, ROOT):48} {kb:8.1f} KB")
    print(
        f"\n  resolved rows: {len(res):,}  | participants: {parts.participant_id.nunique():,}"
    )
    print(f"  category tables: current {len(cur):,} rows, ever-met {len(eve):,} rows")


if __name__ == "__main__":
    main()

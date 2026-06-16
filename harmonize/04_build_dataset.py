import importlib.util
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
RAW = ("/Users/eu/Library/CloudStorage/OneDrive-Stanford/Research Projects/1 - Data/ABCD/"
       "ABCD 7.0/KSADS/rawdata/phenotype")
DSET = os.path.join(os.path.dirname(HERE), "dataset")
os.makedirs(DSET, exist_ok=True)

spec = importlib.util.spec_from_file_location("cw3", os.path.join(HERE, "03_category_crosswalk.py"))
L3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L3)

EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]
V1 = {"ses-00A", "ses-01A", "ses-02A"}
CATS = ["Depression", "Anxiety", "ADHD", "ODD", "Conduct", "Bipolar", "DMDD",
        "OCD", "PTSD", "Autism", "Tic", "Eating", "Psychosis"]
RANK = {"positive": 3, "administered_negative": 2, "not_administered": 1}
INV = {3: "positive", 2: "administered_negative", 1: "not_administered"}
NDA_FRONT = ["subjectkey", "src_subject_id", "participant_id", "session_id",
             "interview_age", "interview_date", "sex"]

cw = L3.build_crosswalk()
res = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
for c in ["session_id", "variable", "resolved", "informant", "module", "status_layer"]:
    res[c] = res[c].astype(str)
res = res.merge(cw[["variable", "category", "is_subthreshold"]], on="variable", how="left")
res["ksads_version"] = res.session_id.map(lambda s: "1.0" if s in V1 else "2.0")


def age_date(pref):
    cols = ["participant_id", "session_id", f"{pref}_ksads__dep_age", f"{pref}_ksads__dep_dtt"]
    d = pd.read_csv(os.path.join(RAW, f"{pref}_ksads__dep.tsv"), sep="\t", usecols=cols, dtype=str)
    return d.rename(columns={cols[2]: "age_yr", cols[3]: "dtt"})


ad = pd.concat([age_date("mh_p"), age_date("mh_y")], ignore_index=True)
ad["age_yr"] = pd.to_numeric(ad.age_yr, errors="coerce")
ad = ad.dropna(subset=["age_yr"]).sort_values("age_yr").drop_duplicates(["participant_id", "session_id"])
ad["interview_age"] = (ad.age_yr * 12).round().astype("Int64")
ad["interview_date"] = ad.dtt.str.slice(0, 10)
SESS = ad.set_index(["participant_id", "session_id"])[["interview_age", "interview_date"]]


def nda(df):
    d = df.join(SESS, on=["participant_id", "session_id"])
    d["subjectkey"] = ""
    d["src_subject_id"] = d.participant_id.str.replace("sub-", "", regex=False)
    d["sex"] = ""
    return d[NDA_FRONT + [c for c in d.columns if c not in NDA_FRONT]]


keep = res[res.resolved != "no_record"]
nda(keep).to_csv(os.path.join(DSET, "ksads_diagnosis_resolved.csv.gz"),
                 index=False, compression="gzip")

base = res[res.session_id.isin(EVEN)]


def caseness(cwx, status_set, informant):
    if informant == "either":
        x = pd.concat([L3.build_caseness(base, cwx, status_set, False, "parent"),
                       L3.build_caseness(base, cwx, status_set, False, "youth")])
        x["rk"] = x.status.map(RANK)
        x = x.groupby(["participant_id", "session_id", "category"])["rk"].max().reset_index()
        x["status"] = x.rk.map(INV)
        return x[["participant_id", "session_id", "category", "status"]]
    return L3.build_caseness(base, cwx, status_set, False, informant)


def wide(cwx, cols, status_set):
    frames = []
    for inf in ["parent", "youth", "either"]:
        c = caseness(cwx, status_set, inf)
        c = c[c.category.isin(cols)]
        w = c.pivot_table(index=["participant_id", "session_id"], columns="category",
                          values="status", aggfunc="first").reset_index()
        for col in cols:
            if col not in w:
                w[col] = "not_administered"
            w[col] = w[col].fillna("not_administered")
        w["informant"] = inf
        frames.append(w[["participant_id", "session_id", "informant"] + cols])
    return pd.concat(frames, ignore_index=True)


DISORDER = {"dep": "Depression", "gad": "GAD", "sepanx": "Separation anxiety",
            "socanx": "Social anxiety", "panic": "Panic", "agor": "Agoraphobia",
            "phobia": "Specific phobia", "ocd": "OCD", "ptsd": "PTSD", "adhd": "ADHD",
            "odd": "ODD", "cond": "Conduct", "bpd": "Bipolar", "dmdd": "DMDD",
            "asd": "Autism", "tic": "Tic", "ed": "Eating", "psych": "Psychosis"}
DCOLS = list(DISORDER.values())
cw_d = cw.copy()
cw_d["category"] = cw_d.module.map(DISORDER)
cw_d = cw_d[cw_d.category.notna()]

for status_set, cat_name, dis_name in [
        ("current", "ksads_categories_current", "ksads_disorders_current"),
        ("ever_met", "ksads_categories_evermet", "ksads_disorders_evermet")]:
    nda(wide(cw, CATS, status_set)).to_csv(os.path.join(DSET, f"{cat_name}.csv"), index=False)
    nda(wide(cw_d, DCOLS, status_set)).to_csv(os.path.join(DSET, f"{dis_name}.csv"), index=False)

sess = ad[["participant_id", "session_id", "interview_age", "interview_date"]].copy()
sess["subjectkey"] = ""
sess["src_subject_id"] = sess.participant_id.str.replace("sub-", "", regex=False)
sess["sex"] = ""
sess["ksads_version"] = sess.session_id.map(lambda s: "1.0" if s in V1 else "2.0")
sess[NDA_FRONT + ["ksads_version"]].to_csv(os.path.join(DSET, "sessions.csv"), index=False)


def n_waves(inf):
    a = res[(res.informant == inf) & res.resolved.isin(["positive", "administered_negative"])]
    return a.groupby("participant_id").session_id.nunique()


parts = pd.DataFrame({"participant_id": sorted(res.participant_id.unique())})
parts["subjectkey"] = ""
parts["src_subject_id"] = parts.participant_id.str.replace("sub-", "", regex=False)
parts["sex"] = ""
parts["n_waves_parent_ksads"] = parts.participant_id.map(n_waves("parent")).fillna(0).astype(int)
parts["n_waves_youth_ksads"] = parts.participant_id.map(n_waves("youth")).fillna(0).astype(int)
parts[["subjectkey", "src_subject_id", "participant_id", "sex",
       "n_waves_parent_ksads", "n_waves_youth_ksads"]].to_csv(
    os.path.join(DSET, "participants.csv"), index=False)

cw.to_csv(os.path.join(DSET, "ksads_category_crosswalk.csv"), index=False)
(pd.read_csv(os.path.join(DERIV, "ksads_administration_calendar.csv"))
 .to_csv(os.path.join(DSET, "ksads_administration_calendar.csv"), index=False))

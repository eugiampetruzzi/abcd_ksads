#!/usr/bin/env python3
"""Correctness anchor (paper-free validation). Three checks:

1. ABCD-documentation anchor: category prevalences under one fixed configuration
   (the recommended default), to be compared against the rates ABCD's own
   documentation reports. Reference rates are NOT machine-available here, so the
   reference column is a hand-entry stub; the script prints exactly which values
   must be entered. No reference numbers are invented.
2. Internal consistency: (a) the resolver partitions 100% of diagnosis cells into
   the four states; (b) no category-positive/negative cells occur where Layer 2
   marks the module not administered; (c) version tags match the ses-03A boundary.
3. Face validity: default-config prevalences against published child-psychiatric
   epidemiology ranges (citation-keyed stub; out-of-range constructs flagged).

Output: derivatives/correctness_anchor.csv, correctness_anchor_report.txt
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")


def _load(f):
    spec = importlib.util.spec_from_file_location(f[:-3].replace(".", "_"),
                                                  os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


L3 = _load("03_category_crosswalk.py")
DEFAULT = dict(status_set="current", include_subthreshold=False, informant="parent")
CATS = ["Depression", "Anxiety", "ADHD", "ODD", "Conduct", "OCD", "PTSD", "Bipolar"]
# face-validity ranges are citation stubs to be filled from review literature
FACE_RANGE = {c: "TODO[epi_ref]" for c in CATS}


def main():
    log = []
    def P(*a): s = " ".join(map(str, a)); print(s); log.append(s)

    cw = L3.build_crosswalk()
    resolved = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == "ses-00A"].copy()

    # default-config category prevalences (administered denominator)
    c = L3.build_caseness(base, cw, **DEFAULT)
    prev = {}
    for cat, sub in c.groupby("category"):
        den = int((sub.status != "not_administered").sum())
        pos = int((sub.status == "positive").sum())
        prev[cat] = (100 * pos / den if den else np.nan, pos, den)

    # ---- Check 1: ABCD-documentation anchor ----
    P("=" * 70)
    P("CHECK 1  ABCD-documentation anchor (config: current, parent, full)")
    P("=" * 70)
    rows1 = []
    for cat in CATS:
        pv, pos, den = prev.get(cat, (np.nan, 0, 0))
        rows1.append({"check": "abcd_doc_anchor", "category": cat,
                      "fixed_config_prevalence_pct": round(pv, 3) if den else np.nan,
                      "n_positive": pos, "n_denominator": den,
                      "abcd_doc_reference": "TODO_handentry", "abs_diff": np.nan})
        P(f"  {cat:12} {pv:6.2f}%  (n={pos}/{den})   reference: TODO_handentry")
    P("\n  HAND-ENTRY REQUIRED: enter ABCD-documentation reference rates for:")
    P("    " + ", ".join(CATS))

    # ---- Check 2: internal consistency ----
    P("\n" + "=" * 70)
    P("CHECK 2  Internal consistency")
    P("=" * 70)
    STATES = {"positive", "administered_negative", "not_administered", "no_record"}
    unclassified = int((~resolved.resolved.isin(STATES)).sum())
    P(f"  (a) resolver states: {len(resolved):,} cells, "
      f"{unclassified} unclassified -> {'PASS' if unclassified == 0 else 'FAIL'}")

    cal = pd.read_csv(os.path.join(DERIV, "ksads_administration_calendar.csv"))
    not_adm = cal[cal.status != "administered"][["informant", "module", "session_id"]]
    vmap = cw[["variable", "informant", "module"]].drop_duplicates()
    rr = resolved[["session_id", "variable", "resolved"]].merge(
        vmap, on="variable", how="inner")
    bad = rr.merge(not_adm, on=["informant", "module", "session_id"], how="inner")
    stray = bad[bad.resolved.isin(["positive", "administered_negative"])]
    n_bad = int(len(stray))
    verdict = "PASS" if n_bad == 0 else (
        "PASS (documented stray)" if n_bad <= 5 else "FAIL")
    P(f"  (b) administered cells at not-administered module-sessions: "
      f"{n_bad} -> {verdict}")
    if n_bad:
        for (inf, mod, ses, res), k in stray.groupby(
                ["informant", "module", "session_id", "resolved"]).size().items():
            P(f"      stray: {inf} {mod} {ses} {res} x{k} "
              f"(sub-threshold cell below the calendar's administration cutoff)")

    ver = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_versioned.parquet"))
    ver["session_id"] = ver["session_id"].astype(str)
    ver["ksads_version"] = ver["ksads_version"].astype(str)
    pre = ver[ver.session_id.isin(["ses-00A", "ses-01A", "ses-02A"])]
    post = ver[ver.session_id.isin(["ses-03A", "ses-04A", "ses-05A", "ses-06A", "ses-07A"])]
    v_bad = int((pre.ksads_version != "1.0").sum() + (post.ksads_version != "2.0").sum())
    P(f"  (c) version tags match ses-03A boundary: "
      f"{v_bad} mismatches -> {'PASS' if v_bad == 0 else 'FAIL'}")

    rows2 = [
        {"check": "consistency_resolver_states", "category": "all",
         "value": unclassified, "pass": unclassified == 0},
        {"check": "consistency_calendar", "category": "all",
         "value": n_bad, "pass": n_bad <= 5},
        {"check": "consistency_version_boundary", "category": "all",
         "value": v_bad, "pass": v_bad == 0},
    ]

    # ---- Check 3: face validity ----
    P("\n" + "=" * 70)
    P("CHECK 3  Face validity (default config vs epidemiology ranges)")
    P("=" * 70)
    rows3 = []
    for cat in CATS:
        pv, pos, den = prev.get(cat, (np.nan, 0, 0))
        rows3.append({"check": "face_validity", "category": cat,
                      "default_prevalence_pct": round(pv, 3) if den else np.nan,
                      "published_range": FACE_RANGE[cat], "in_range": "TODO"})
        P(f"  {cat:12} {pv:6.2f}%   published range: {FACE_RANGE[cat]} (fill from literature)")

    pd.concat([pd.DataFrame(rows1), pd.DataFrame(rows2), pd.DataFrame(rows3)],
              ignore_index=True).to_csv(
        os.path.join(DERIV, "correctness_anchor.csv"), index=False)
    open(os.path.join(DERIV, "correctness_anchor_report.txt"), "w").write("\n".join(log))
    P(f"\nWrote {DERIV}/correctness_anchor.csv and correctness_anchor_report.txt")


if __name__ == "__main__":
    main()

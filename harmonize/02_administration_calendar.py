#!/usr/bin/env python3
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")

SESSIONS = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
            "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
V2_SWITCH = "ses-03A"   # KSADS-COMP 1.0 -> 2.0 at the 3-year follow-up
ADMIN_THRESHOLD = 0.5   # fraction of (administered / non-no_record) to call "administered"
GRID = {"administered": "X", "not_administered": ".", "absent": ""}


def classify(pos, neg, not_admin, no_rec):
    administered = pos + neg
    in_release = administered + not_admin          # cells that exist in the file
    total = in_release + no_rec
    if total == 0 or in_release == 0:
        return "absent"
    if administered > 0 and administered / in_release >= ADMIN_THRESHOLD:
        return "administered"
    if administered == 0:
        return "not_administered"
    return "administered"   # partial but non-trivial coverage


def main():
    s = pd.read_csv(os.path.join(DERIV, "ksads_resolution_summary.csv"))
    pres = s[s.status_layer == "present"].copy()
    # modules whose only diagnosis layer is not "present" (e.g. some have past-only)
    have_present = set(zip(pres.informant, pres.module))
    extra = s[~s.set_index(["informant", "module"]).index.isin(have_present)]
    pres = pd.concat([pres, extra], ignore_index=True)

    g = (pres.groupby(["informant", "module", "session_id"], as_index=False)
              [["n_positive", "n_administered_negative",
                "n_not_administered", "n_no_record"]].sum())
    g["status"] = g.apply(lambda r: classify(
        r.n_positive, r.n_administered_negative,
        r.n_not_administered, r.n_no_record), axis=1)
    g["n_administered"] = g.n_positive + g.n_administered_negative

    # flags: "administered" is the only "given" state; absent and not_administered
    # both mean "not given to anyone at this wave". The diagnostic cadence is
    # biennial (even waves), so drop/add are benchmarked against the even waves;
    # odd interim waves (01A/03A/05A/07A) are reported but not used for drop logic.
    EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]
    flags = {}
    for (inf, mod), sub in g.groupby(["informant", "module"]):
        st = sub.set_index("session_id")["status"].reindex(SESSIONS).fillna("absent")
        admin_any = [w for w in SESSIONS if st[w] == "administered"]
        admin_even = [w for w in EVEN if st[w] == "administered"]
        notes = []
        if not admin_any:
            notes.append("never_administered")
        else:
            if admin_any[0] != "ses-00A":                       # absent at baseline
                notes.append("added@" + admin_any[0][-3:])
            if admin_even and admin_even[-1] != "ses-06A":      # stops before the biennial endpoint
                notes.append("dropped_after@" + admin_even[-1][-3:])
            if admin_even:
                hole = [w for w in EVEN if admin_even[0] < w < admin_even[-1]
                        and st[w] != "administered"]
                if hole:
                    notes.append("intermittent")
        flags[(inf, mod)] = ";".join(notes)
    g["flag"] = g.apply(lambda r: flags.get((r.informant, r.module), ""), axis=1)

    long_cols = ["informant", "module", "session_id", "status",
                 "n_administered", "n_positive", "n_not_administered",
                 "n_no_record", "flag"]
    g[long_cols].sort_values(["informant", "module", "session_id"]).to_csv(
        os.path.join(DERIV, "ksads_administration_calendar.csv"), index=False)

    # readable grid
    grid = (g.assign(cell=g.status.map(GRID))
              .pivot_table(index=["informant", "module", "flag"],
                           columns="session_id", values="cell",
                           aggfunc="first", fill_value=""))
    grid = grid.reindex(columns=SESSIONS).reset_index()
    grid.to_csv(os.path.join(DERIV, "ksads_administration_grid.csv"), index=False)

    # report
    print("Layer 2 administration calendar")
    print("  X = administered, . = not administered (all 555), blank = absent")
    print(f"  V2 switch at {V2_SWITCH}\n")
    cur = None
    for _, r in grid.iterrows():
        if r["informant"] != cur:
            cur = r["informant"]
            print(f"\n[{cur.upper()}]  " + " ".join(w[-3:] for w in SESSIONS))
        cells = " ".join(f"{str(r[w]):>3}" for w in SESSIONS)
        fl = f"  <-- {r['flag']}" if r["flag"] else ""
        print(f"  {r['module']:8} {cells}{fl}")
    nadd = grid.flag.str.contains("added@").sum()
    ndrop = grid.flag.str.contains("dropped_after@").sum()
    print(f"\nFlagged: {nadd} added mid-study, {ndrop} dropped before the final wave.")
    print(f"\nWrote {DERIV}/ksads_administration_calendar.csv")
    print(f"Wrote {DERIV}/ksads_administration_grid.csv")


if __name__ == "__main__":
    main()
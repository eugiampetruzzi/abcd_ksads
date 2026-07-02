"""Module x wave x informant administration calendar.

Classifies whether each KSADS module was administered at each wave (from resolved
state counts) and annotates the biennial administration cadence (added / dropped /
intermittent). The ``02_administration_calendar.py`` script wires this to the CSVs.
"""

import pandas as pd

SESSIONS = [
    "ses-00A", "ses-01A", "ses-02A", "ses-03A",
    "ses-04A", "ses-05A", "ses-06A", "ses-07A",
]
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]  # biennial diagnostic waves
V2_SWITCH = "ses-03A"  # KSADS-COMP 1.0 -> 2.0 at the 3-year follow-up
ADMIN_THRESHOLD = 0.5  # fraction of (administered / non-no_record) to call "administered"
GRID = {"administered": "X", "not_administered": ".", "absent": ""}

_COUNT_COLS = [
    "n_positive", "n_administered_negative", "n_not_administered", "n_no_record",
]
_LONG_COLS = [
    "informant", "module", "session_id", "status", "n_administered",
    "n_positive", "n_not_administered", "n_no_record", "flag",
]


def classify(pos, neg, not_admin, no_rec):
    """Administration state of a module-wave cell from its resolved-state counts."""
    administered = pos + neg
    in_release = administered + not_admin  # cells that exist in the file
    total = in_release + no_rec
    if total == 0 or in_release == 0:
        return "absent"
    if administered > 0 and administered / in_release >= ADMIN_THRESHOLD:
        return "administered"
    if administered == 0:
        return "not_administered"
    return "administered"  # partial but non-trivial coverage


def administration_flags(status_by_session, sessions=SESSIONS, even=EVEN):
    """Cadence annotation for one module's per-session status Series.

    Benchmarks against the even (biennial) waves: flags a module added after
    baseline, dropped before the final biennial wave, or administered intermittently.
    """
    st = status_by_session.reindex(sessions).fillna("absent")
    admin_any = [w for w in sessions if st[w] == "administered"]
    admin_even = [w for w in even if st[w] == "administered"]
    notes = []
    if not admin_any:
        notes.append("never_administered")
    else:
        if admin_any[0] != "ses-00A":  # absent at baseline
            notes.append("added@" + admin_any[0][-3:])
        if admin_even and admin_even[-1] != "ses-06A":  # stops before the endpoint
            notes.append("dropped_after@" + admin_even[-1][-3:])
        if admin_even:
            hole = [
                w for w in even
                if admin_even[0] < w < admin_even[-1] and st[w] != "administered"
            ]
            if hole:
                notes.append("intermittent")
    return ";".join(notes)


def build_calendar(summary):
    """From the resolution summary, build the (long calendar, wide grid) tables."""
    pres = summary[summary.status_layer == "present"].copy()
    # modules whose only diagnosis layer is not "present" (e.g. some are past-only)
    have_present = set(zip(pres.informant, pres.module))
    extra = summary[~summary.set_index(["informant", "module"]).index.isin(have_present)]
    pres = pd.concat([pres, extra], ignore_index=True)

    g = pres.groupby(["informant", "module", "session_id"], as_index=False)[
        _COUNT_COLS
    ].sum()
    g["status"] = g.apply(
        lambda r: classify(
            r.n_positive, r.n_administered_negative, r.n_not_administered, r.n_no_record
        ),
        axis=1,
    )
    g["n_administered"] = g.n_positive + g.n_administered_negative

    flags = {}
    for (inf, mod), sub in g.groupby(["informant", "module"]):
        flags[(inf, mod)] = administration_flags(sub.set_index("session_id")["status"])
    g["flag"] = g.apply(lambda r: flags.get((r.informant, r.module), ""), axis=1)

    long = g[_LONG_COLS].sort_values(["informant", "module", "session_id"])
    grid = (
        g.assign(cell=g.status.map(GRID))
        .pivot_table(
            index=["informant", "module", "flag"],
            columns="session_id",
            values="cell",
            aggfunc="first",
            fill_value="",
        )
        .reindex(columns=SESSIONS)
        .reset_index()
    )
    return long, grid

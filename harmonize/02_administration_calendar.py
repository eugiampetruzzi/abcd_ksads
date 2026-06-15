import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
SESSIONS = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
            "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
EVEN = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]
THRESH = 0.5

r = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"),
                    columns=["informant", "module", "session_id", "status_layer", "resolved"])
for c in r.columns:
    r[c] = r[c].astype(str)
p = r[r.status_layer == "present"]
g = (p.groupby(["informant", "module", "session_id"]).resolved
       .value_counts().unstack(fill_value=0).reset_index())
for s in ("positive", "administered_negative", "not_administered"):
    if s not in g:
        g[s] = 0
g["n_administered"] = g.positive + g.administered_negative
g["in_release"] = g.n_administered + g.not_administered


def state(row):
    if row.in_release == 0:
        return "absent"
    if row.n_administered > 0 and row.n_administered / row.in_release >= THRESH:
        return "administered"
    return "not_administered" if row.n_administered == 0 else "administered"


g["status"] = g.apply(state, axis=1)

flags = {}
for (inf, mod), sub in g.groupby(["informant", "module"]):
    st = sub.set_index("session_id")["status"].reindex(SESSIONS).fillna("absent")
    admin = [s for s in SESSIONS if st[s] == "administered"]
    admin_even = [s for s in EVEN if st[s] == "administered"]
    notes = []
    if not admin:
        notes.append("never_administered")
    else:
        if admin[0] != "ses-00A":
            notes.append("added@" + admin[0][-3:])
        if admin_even and admin_even[-1] != "ses-06A":
            notes.append("dropped_after@" + admin_even[-1][-3:])
        if admin_even and [s for s in EVEN
                           if admin_even[0] < s < admin_even[-1] and st[s] != "administered"]:
            notes.append("intermittent")
    flags[(inf, mod)] = ";".join(notes)
g["flag"] = g.apply(lambda r: flags[(r.informant, r.module)], axis=1)

g = g.rename(columns={"positive": "n_positive", "not_administered": "n_not_administered"})
(g[["informant", "module", "session_id", "status", "n_administered",
    "n_positive", "n_not_administered", "flag"]]
 .sort_values(["informant", "module", "session_id"])
 .to_csv(os.path.join(DERIV, "ksads_administration_calendar.csv"), index=False))

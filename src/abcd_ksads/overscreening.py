"""Module over-screening: baseline parent present-diagnosis prevalence by module,
compared to approximate epidemiological context. The ``14_module_overscreening.py``
script wires this to the resolved cache and CSVs.
"""

import pandas as pd

# approximate childhood (~9-10 yr) prevalence context, for flagging only
EPI = {
    "phobia": "~5", "ocd": "1-2", "bpd": "<1", "psych": "<0.5", "adhd": "~7-9",
    "odd": "~3-5", "cond": "~1-2", "dep": "<1 (current)", "gad": "~1",
    "socanx": "~1-2", "panic": "<1", "ptsd": "~1",
}
LABEL = {
    "phobia": "Specific phobia", "ocd": "OCD", "bpd": "Bipolar",
    "psych": "Psychotic disorders", "adhd": "ADHD", "odd": "ODD",
    "cond": "Conduct", "dep": "Depression (MDD/PDD)", "gad": "GAD",
    "socanx": "Social anxiety", "panic": "Panic", "ptsd": "PTSD",
}


def prevalence_over_assessed(d):
    """(#positive participants, #assessed participants, pct) over the assessed set.

    Assessed = participants with a positive or administered_negative cell; prevalence
    is computed over that denominator only (never counting not_administered)."""
    pos = set(d[d.resolved == "positive"].participant_id)
    adm = set(d[d.resolved.isin(["positive", "administered_negative"])].participant_id)
    return len(pos), len(adm), (100 * len(pos) / len(adm) if adm else 0.0)


def module_overscreening(base, labels=LABEL, epi=EPI):
    """Per-module baseline present-diagnosis prevalence, ordered high to low."""
    rows = []
    for mod in labels:
        npos, nadm, pct = prevalence_over_assessed(
            base[(base.module == mod) & (base.status_layer == "present")]
        )
        rows.append(
            {
                "disorder": labels[mod],
                "module": mod,
                "present_core_pct": round(pct, 2),
                "n_positive": npos,
                "n_administered": nadm,
                "approx_childhood_pct": epi.get(mod, ""),
            }
        )
    return pd.DataFrame(rows).sort_values("present_core_pct", ascending=False)


def depression_breakdown(base):
    """Depression present vs past prevalence (core MDD/PDD), with the past:present ratio."""
    dep = base[base.module == "dep"]
    pn, _, pp = prevalence_over_assessed(dep[dep.status_layer == "present"])
    an, _, ap = prevalence_over_assessed(dep[dep.status_layer == "past"])
    return pd.DataFrame(
        [
            {
                "construct": "Depression (core MDD/PDD)",
                "present_pct": round(pp, 2),
                "n_present": pn,
                "past_pct": round(ap, 2),
                "n_past": an,
                "past_to_present_ratio": round(an / pn, 1) if pn else None,
            }
        ]
    )

"""Decompose the anxiety construct into its constituent sub-disorders and quantify
the effect of including specific phobia. The ``10b_aux_anxiety_decomposition.py``
script wires this to the resolved cache, the CSV, and the figure.
"""

import pandas as pd

SUBS = [
    ("gad", "GAD"),
    ("sepanx", "Separation"),
    ("socanx", "Social"),
    ("panic", "Panic"),
    ("agor", "Agoraphobia"),
    ("phobia", "Specific phobia"),
]


def _module_pos(base, mod):
    """Positive-participant and assessed-participant sets for one sub-disorder module."""
    m = base[base.module == mod]
    pos = set(m[m.resolved == "positive"].participant_id)
    assessed = set(m[m.resolved.isin(["positive", "administered_negative"])].participant_id)
    return pos, assessed


def decompose_anxiety(base, subs=SUBS):
    """Per-sub-disorder prevalence plus cumulative 'any anxiety' with/without phobia.

    ``base`` is resolved rows at baseline, parent informant, present layer. Returns
    ``(per_sub_df, cumulative_pcts, any_with_phobia, any_without_phobia, n_assessed_all)``
    where the cumulative list adds sub-disorders in ``subs`` order over the shared
    assessed set (union of the sub-disorder assessed sets, size ``n_assessed_all``).
    """
    rows, pos_sets, assessed_all = [], {}, set()
    for mod, lab in subs:
        pos, assessed = _module_pos(base, mod)
        pos_sets[mod] = pos
        assessed_all |= assessed
        rows.append(
            {
                "sub": lab,
                "n_pos": len(pos),
                "n_assessed": len(assessed),
                "prevalence_pct": round(100 * len(pos) / len(assessed), 2),
            }
        )
    dec = pd.DataFrame(rows)

    cum, ids = [], set()
    for mod, lab in subs:
        ids |= pos_sets[mod]
        cum.append(100 * len(ids) / len(assessed_all))
    any_with = cum[-1]
    any_without_phobia = (
        100
        * len(set().union(*[pos_sets[m] for m, _ in subs if m != "phobia"]))
        / len(assessed_all)
    )
    return dec, cum, any_with, any_without_phobia, len(assessed_all)

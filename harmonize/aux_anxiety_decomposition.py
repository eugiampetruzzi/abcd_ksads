#!/usr/bin/env python3
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

SUBS = [
    ("gad", "GAD"),
    ("sepanx", "Separation"),
    ("socanx", "Social"),
    ("panic", "Panic"),
    ("agor", "Agoraphobia"),
    ("phobia", "Specific phobia"),
]


def main():
    r = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "module", "informant", "status_layer", "resolved"]:
        r[c] = r[c].astype(str)
    base = r[
        (r.session_id == "ses-00A")
        & (r.informant == "parent")
        & (r.status_layer == "present")
    ]

    # per sub-disorder: positive at any present-dx variable in that module
    def module_pos(mod):
        m = base[base.module == mod]
        pos = set(m[m.resolved == "positive"].participant_id)
        assessed = set(
            m[m.resolved.isin(["positive", "administered_negative"])].participant_id
        )
        return pos, assessed

    rows, pos_sets, assessed_all = [], {}, set()
    for mod, lab in SUBS:
        pos, assessed = module_pos(mod)
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

    # cumulative "any anxiety" as sub-disorders are added in order
    cum, ids = [], set()
    for mod, lab in SUBS:
        ids |= pos_sets[mod]
        cum.append(100 * len(ids) / len(assessed_all))
    any_with = cum[-1]
    any_without_phobia = (
        100
        * len(set().union(*[pos_sets[m] for m, _ in SUBS if m != "phobia"]))
        / len(assessed_all)
    )
    dec_out = pd.concat(
        [
            dec,
            pd.DataFrame(
                [
                    {
                        "sub": "ANY (without phobia)",
                        "n_pos": np.nan,
                        "n_assessed": len(assessed_all),
                        "prevalence_pct": round(any_without_phobia, 2),
                    },
                    {
                        "sub": "ANY (with phobia)",
                        "n_pos": np.nan,
                        "n_assessed": len(assessed_all),
                        "prevalence_pct": round(any_with, 2),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    dec_out.to_csv(os.path.join(DERIV, "anxiety_decomposition.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    labs = [l for _, l in SUBS]
    ax1.barh(labs[::-1], dec.prevalence_pct[::-1], color="#2166AC")
    ax1.set_xlabel("Baseline present-diagnosis prevalence (%)")
    for i, v in enumerate(dec.prevalence_pct[::-1]):
        ax1.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8)

    ax2.plot(range(1, len(cum) + 1), cum, "-o", color="#B2182B")
    ax2.set_xticks(range(1, len(cum) + 1))
    ax2.set_xticklabels([l for _, l in SUBS], rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel('"Any anxiety" prevalence (%)')
    ax2.set_xlabel("sub-disorders included (cumulative)")
    ax2.axhline(any_without_phobia, ls="--", lw=0.8, color="#999999")
    ax2.text(
        1,
        any_without_phobia + 0.3,
        f"without phobia = {any_without_phobia:.1f}%",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(DERIV, "fig_anxiety_decomposition.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(dec.to_string(index=False))
    print(f"\n'Any anxiety' with all sub-disorders:  {any_with:.2f}%")
    print(f"'Any anxiety' excluding specific phobia: {any_without_phobia:.2f}%")
    print(
        f"-> including specific phobia changes the anxiety construct "
        f"{any_with / any_without_phobia:.1f}x."
    )
    print(f"\nWrote {DERIV}/fig_anxiety_decomposition.png")


if __name__ == "__main__":
    main()

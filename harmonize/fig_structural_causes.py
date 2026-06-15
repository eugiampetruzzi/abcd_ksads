"""Figure 3: structural features of the release that make diagnoses ambiguous.

(A) Every diagnosis cell resolves to one of four administration states; the
    not-administered code (555) is 37.7% of all ~18 million cells, so reading it
    as a screened-negative fabricates millions of unassessed "healthy" cells.
(B) The diagnostic battery is not constant: modules are administered on differing
    schedules and several are added or dropped mid-study (administration calendar).

Sources: derivatives/ksads_resolution_summary.csv, ksads_administration_grid.csv
Output: derivatives/fig_structural_causes.png / .pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.linewidth": 0.8})

SES = ["ses-00A", "ses-01A", "ses-02A", "ses-03A",
       "ses-04A", "ses-05A", "ses-06A", "ses-07A"]
STATE_ORDER = ["positive", "administered_negative", "not_administered", "no_record"]
STATE_LAB = {"positive": "positive (met criteria)",
             "administered_negative": "administered negative",
             "not_administered": "not administered (555)",
             "no_record": "no record"}
STATE_COL = {"positive": "#2166AC", "administered_negative": "#C6DBEF",
             "not_administered": "#D55E00", "no_record": "#EEEEEE"}


def panel_a(ax):
    s = pd.read_csv(os.path.join(DERIV, "ksads_resolution_summary.csv"))
    tot = {st: int(s[f"n_{st}"].sum()) for st in STATE_ORDER}
    n = sum(tot.values())
    left = 0
    for st in STATE_ORDER:
        w = 100 * tot[st] / n
        ax.barh(0, w, left=left, color=STATE_COL[st], edgecolor="white", height=0.6)
        if w > 4:
            ax.text(left + w / 2, 0, f"{w:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if st in ("positive", "not_administered") else "#333")
        left += w
    pos_pct = 100 * tot["positive"] / n
    ax.annotate(f"positive {pos_pct:.1f}%", xy=(pos_pct / 2, 0.32),
                xytext=(6, 0.62), fontsize=7.5, color="#2166AC", ha="left",
                arrowprops=dict(arrowstyle="-", color="#2166AC", lw=0.7))
    ax.set_xlim(0, 100); ax.set_ylim(-0.5, 0.9)
    ax.set_yticks([]); ax.set_xlabel("share of all ~18 million diagnosis cells (%)")
    ax.set_title("A   four administration states; 555 (not administered) is 37.7%",
                 loc="left", fontsize=9, fontweight="bold")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Patch(fc=STATE_COL[st], label=STATE_LAB[st]) for st in STATE_ORDER],
              loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False,
              fontsize=7.8, handlelength=1.2, columnspacing=1.4)


def panel_b(ax):
    g = pd.read_csv(os.path.join(DERIV, "ksads_administration_grid.csv"))
    g = g.sort_values(["informant", "module"]).reset_index(drop=True)
    code = {"X": 2, ".": 1, "": 0}
    M = np.array([[code.get(str(g.loc[i, s]).strip(), 0) for s in SES]
                  for i in range(len(g))])
    cmap = ListedColormap(["#FFFFFF", "#FCE3D0", "#D55E00"])  # absent, not-admin, administered
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(SES)))
    ax.set_xticklabels([s[-3:] for s in SES], fontsize=7.5)
    ax.set_yticks(range(len(g)))
    labels = [f"{r.informant[0]}:{r.module}" for _, r in g.iterrows()]
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_xlabel("session")
    ax.set_title("B   module x wave administration (p: parent, y: youth)",
                 loc="left", fontsize=9, fontweight="bold")
    for x in range(len(SES) + 1):
        ax.axvline(x - 0.5, color="white", lw=0.5)
    for y in range(len(g) + 1):
        ax.axhline(y - 0.5, color="white", lw=0.5)
    # mark added/dropped rows
    for i, (_, r) in enumerate(g.iterrows()):
        if isinstance(r.flag, str) and r.flag.strip():
            ax.text(len(SES) - 0.4, i, r.flag.split(";")[0].replace("dropped_after", "drop")
                    .replace("added", "add"), va="center", ha="left", fontsize=5.6, color="#444")
    ax.legend(handles=[Patch(fc="#D55E00", label="administered"),
                       Patch(fc="#FCE3D0", label="not administered"),
                       Patch(fc="#FFFFFF", ec="#ccc", label="absent")],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False,
              fontsize=7.5, handlelength=1.2)


def main():
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 6.2], hspace=0.42)
    panel_a(fig.add_subplot(gs[0]))
    panel_b(fig.add_subplot(gs[1]))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DERIV, f"fig_structural_causes.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("Wrote fig_structural_causes.png / .pdf")


if __name__ == "__main__":
    main()

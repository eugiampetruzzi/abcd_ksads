#!/usr/bin/env python3
"""Figure 1: how a single construct's prevalence depends on its definition.

Each point is one defensible operationalization of the construct, computed on the
same release-7.0 data. Point color encodes the diagnostic-status choice (current
vs ever-met) and point shape encodes the informant rule -- the two dominant
levers -- so the reader sees what drives the spread without decoding a matrix.
The dashed line is the median; the range and fold are annotated.

Main: any-disorder. Supplement: 2x2 for depression, anxiety, externalizing, ADHD.
Output: derivatives/fig_specification_curve_any.png/.pdf, fig_specification_curve_panel.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})

STATUS_COLOR = {"current": "#4393C3", "ever_met": "#D6604D"}
INFORMANT_MARKER = {"parent": "o", "youth": "s", "either": "^", "both": "D"}
GREY = "#888888"


def spec_panel(grid, construct, ax, *, legend=False):
    g = grid[grid.construct == construct].sort_values(
        "prevalence_pct").reset_index(drop=True)
    x = np.arange(len(g))
    med = g.prevalence_pct.median()
    lo, hi = g.prevalence_pct.min(), g.prevalence_pct.max()
    fold = hi / lo if lo > 0 else np.inf

    ax.axhline(med, color=GREY, lw=0.9, ls="--", zorder=1)
    for i, r in g.iterrows():
        ax.scatter(i, r.prevalence_pct, s=46,
                   color=STATUS_COLOR.get(r.status, "#999999"),
                   marker=INFORMANT_MARKER.get(r.informant, "o"),
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(-0.7, len(g) - 0.3)
    ax.set_xticks([])
    ax.set_ylabel("prevalence (%)")
    ax.set_xlabel("each point = one definition of the construct")
    foldtxt = f"{fold:.0f}-fold" if np.isfinite(fold) and fold < 100 else f"{hi - lo:.0f}-point"
    ax.set_title(f"{construct}:  {lo:.1f}% to {hi:.1f}%   ({foldtxt} range across "
                 f"{len(g)} definitions)", loc="left", fontsize=9.5)
    ax.annotate(f"{med:.0f}% median", xy=(len(g) - 1, med), xytext=(len(g) - 1, med),
                ha="right", va="bottom", color=GREY, fontsize=7.5)

    if legend:
        h1 = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="white",
                     label="current" if k == "current" else "ever-met")
              for k, c in STATUS_COLOR.items()]
        h2 = [Line2D([0], [0], marker=m, ls="", mfc="#666", mec="white", label=k)
              for k, m in INFORMANT_MARKER.items()]
        leg1 = ax.legend(handles=h1, loc="upper left", frameon=False, fontsize=8,
                         title="status", title_fontsize=8, handletextpad=0.2)
        ax.add_artist(leg1)
        ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, 0.78),
                  frameon=False, fontsize=8, title="informant", title_fontsize=8,
                  ncol=2, handletextpad=0.2, columnspacing=1.0)


def main():
    grid = pd.read_csv(os.path.join(DERIV, "multiverse_grid.csv"))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    spec_panel(grid, "any-disorder", ax, legend=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DERIV, f"fig_specification_curve_any.{ext}"),
                    dpi=300, bbox_inches="tight")

    cons = ["depression", "anxiety", "externalizing", "ADHD"]
    figp, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for k, con in enumerate(cons):
        spec_panel(grid, con, axes[k // 2, k % 2], legend=(k == 0))
    figp.tight_layout()
    figp.savefig(os.path.join(DERIV, "fig_specification_curve_panel.png"),
                 dpi=300, bbox_inches="tight")
    print("Wrote fig_specification_curve_any.png and fig_specification_curve_panel.png")


if __name__ == "__main__":
    main()

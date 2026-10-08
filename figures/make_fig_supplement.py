import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from abcd_ksads import config

config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
for fam in ("Arial", "Helvetica"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ROWS = [("Female (vs male)", "Female"), ("Income (per SD)", "Income"),
        ("Race: Black/AA vs White", "Black vs White"), ("Race: Hispanic vs White", "Hispanic vs White"),
        ("Race: Asian vs White", "Asian vs White"), ("Race: Other/Multiracial vs White", "Other vs White"),
        ("Screen time (per SD)", "Screen time"), ("Family conflict (per SD)", "Family conflict"),
        ("DMN within-network FC (per SD)", "DMN within"), ("Salience within-network FC (per SD)", "Salience within"),
        ("FPN within-network FC (per SD)", "FPN within"), ("DMN-salience FC (per SD)", "DMN–salience"),
        ("DMN-FPN FC (per SD)", "DMN–FPN"), ("Salience-FPN FC (per SD)", "Salience–FPN")]
COLS = ["Any disorder", "Depression", "Anxiety", "Suicidality", "ADHD", "Eating disorders"]
GROUPS = [("Demographic", 0, 5), ("Psychosocial", 6, 7), ("Neuroimaging", 8, 13)]


def heatmap(cell, cbar_label, fname):
    """cell: dict (predictor, construct_label) -> (pct, flip)."""
    M = np.full((len(ROWS), len(COLS)), np.nan)
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    for i, (pred, _) in enumerate(ROWS):
        for j, con in enumerate(COLS):
            if (pred, con) in cell:
                M[i, j] = cell[(pred, con)][0]
    im = ax.imshow(np.ma.masked_invalid(M), cmap="cividis", vmin=0, vmax=100, aspect="auto")
    for i, (pred, _) in enumerate(ROWS):
        for j, con in enumerate(COLS):
            if (pred, con) in cell:
                v, flip = cell[(pred, con)]
                ax.text(j, i, f"{v:.0f}" + ("†" if flip else ""), ha="center", va="center",
                        fontsize=9, color="black" if v > 60 else "white")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888888")
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, lab in ROWS])
    ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for name, a, b in GROUPS:
        ax.annotate("", xy=(-0.33, a - 0.4), xytext=(-0.33, b + 0.4), xycoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", lw=1.2, color="#444444"))
        ax.text(-0.38, (a + b) / 2, name, transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(config.FIGURES_OUT / fname, dpi=300, bbox_inches="tight")
    print("wrote", fname)


def fig_s1():
    S = pd.read_csv(config.DERIV / "inferential_summary.csv")
    cell = {(r.predictor, r.construct_label): (r.pct_sig, bool(r.sign_flip)) for r in S.itertuples()}
    heatmap(cell, "% of specifications significant (p < .05)", "FigureS1_robustness.png")


def fig_s3():
    R = pd.read_csv(config.DERIV / "age_moderation_specs.csv")
    cell = {}
    for (pred, con), d in R.groupby(["predictor", "construct_label"]):
        cell[(pred, con)] = (100 * d.sig.mean(), bool(d.OR.min() < 1 < d.OR.max()))
    heatmap(cell, "% of specifications with significant age interaction (p < .05)", "FigureS3_age_moderation.png")


def fig_s2():
    g = pd.read_csv(config.DERIV / "wave_variation_grid.csv")
    g = g[g.construct == "any-disorder"]
    color = {"current": "#0072B2", "ever_met": "#E69F00"}
    mark = {"parent": "o", "youth": "s", "either": "^", "both": "D"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.6), sharey=True)
    for ax, (ses, lab) in zip(axes.flat, [("ses-00A", "Baseline"), ("ses-02A", "Year 2"),
                                           ("ses-04A", "Year 4"), ("ses-06A", "Year 6")]):
        d = g[g.wave == ses].sort_values("prevalence_pct").reset_index(drop=True)
        for i, r in d.iterrows():
            ax.scatter(i, r.prevalence_pct, s=46, color=color[r.status], marker=mark[r.informant],
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(lab, loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim(0, 62)
    for ax in axes[:, 0]:
        ax.set_ylabel("Any-disorder prevalence (%)")
    for ax in axes[1]:
        ax.set_xlabel("Specifications, ordered by prevalence")
    h1 = [Line2D([0], [0], marker="o", ls="", mfc=c, mec="white", label=l)
          for c, l in [(color["current"], "Current"), (color["ever_met"], "Lifetime")]]
    h2 = [Line2D([0], [0], marker=m, ls="", mfc="#777777", mec="white", label=l)
          for m, l in [("o", "Caregiver"), ("s", "Youth"), ("^", "Either"), ("D", "Both")]]
    l1 = axes[0, 0].legend(handles=h1, title="Timeframe", loc="upper left", frameon=False, fontsize=9)
    axes[0, 0].add_artist(l1)
    axes[0, 0].legend(handles=h2, title="Informant", loc="upper left", bbox_to_anchor=(0, 0.72),
                      frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(config.FIGURES_OUT / "FigureS2_prevalence_by_assessment.png", dpi=300, bbox_inches="tight")
    print("wrote FigureS2_prevalence_by_assessment.png")


if __name__ == "__main__":
    fig_s1()
    fig_s2()
    fig_s3()

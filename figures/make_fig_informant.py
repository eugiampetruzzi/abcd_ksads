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
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
BLUE, RED, GREY = "#0072B2", "#E69F00", "#888888"  # Okabe-Ito colorblind-safe
PREV = pd.read_csv(config.DERIV / "informant_prevalence.csv")
CONC = pd.read_csv(config.DERIV / "informant_concordance.csv")
SES = ["ses-00A", "ses-02A", "ses-04A", "ses-06A"]
XV = {s: i for i, s in enumerate(SES)}
XLAB = ["0", "2", "4", "6"]
ORDER = [
    "Depression",
    "Suicidality",
    "Anxiety",
    "Bipolar",
    "Conduct",
    "Eating",
    "DMDD",
    "OCD",
    "PTSD",
]


# multi-panel trajectories (parent vs youth prevalence over waves)
def fig_trajectories():
    cats = [c for c in ORDER if c in PREV.category.unique()]
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.2 * nrow), sharex=True)
    for ax, cat in zip(axes.flat, cats):
        for inf, c in (("parent", BLUE), ("youth", RED)):
            d = PREV[(PREV.category == cat) & (PREV.informant == inf)].dropna(
                subset=["prevalence_pct"]
            )
            d = d[d.n_denominator > 0]
            x = [XV[s] for s in d.session]
            y = d.prevalence_pct.values
            ax.plot(x, y, "-o", color=c, lw=1.8, ms=5, mec="white", mew=0.6)
        k = CONC[CONC.category == cat]
        ksum = "; ".join(f"y{s[4]}:κ={kk:.2f}" for s, kk in zip(k.session, k.kappa))
        ax.set_title(cat, fontsize=10, fontweight="bold", loc="left")
        ax.set_xticks(range(len(SES)))
        ax.set_xticklabels(XLAB)
        ax.set_xlim(-0.3, len(SES) - 0.7)
    for ax in axes.flat[len(cats) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Years from baseline")
    for ax in axes[:, 0]:
        ax.set_ylabel("Prevalence (%)")
    hl = [
        Line2D([0], [0], color=BLUE, marker="o", mec="white", label="Caregiver"),
        Line2D([0], [0], color=RED, marker="o", mec="white", label="Youth"),
    ]
    fig.legend(
        handles=hl,
        loc="upper right",
        frameon=False,
        fontsize=10,
        ncol=2,
        bbox_to_anchor=(0.99, 1.01),
    )
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(
            config.FIGURES_OUT / f"Figure_informant_trajectories.{e}",
            dpi=300,
            bbox_inches="tight",
        )
    print("wrote Figure_informant_trajectories.png/pdf")


# dumbbell - caregiver vs youth prevalence by category (latest co-assessed wave)
def fig_dumbbell():
    # use ses-06A (all categories co-assessed); fall back per category to its last available wave
    rows = []
    for cat in ORDER:
        for ses in reversed(SES):
            d = PREV[(PREV.category == cat) & (PREV.session == ses)]
            dp = d[d.informant == "parent"]
            dy = d[d.informant == "youth"]
            if (
                len(dp)
                and len(dy)
                and dp.n_denominator.iloc[0] > 0
                and dy.n_denominator.iloc[0] > 0
            ):
                k = CONC[(CONC.category == cat) & (CONC.session == ses)]
                rows.append(
                    {
                        "category": cat,
                        "session": ses,
                        "parent": dp.prevalence_pct.iloc[0],
                        "youth": dy.prevalence_pct.iloc[0],
                        "kappa": k.kappa.iloc[0] if len(k) else np.nan,
                    }
                )
                break
    D = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, r in D.iterrows():
        ax.plot([r.parent, r.youth], [i, i], color=GREY, lw=1.6, zorder=1)
        ax.scatter(
            r.parent, i, s=85, color=BLUE, edgecolor="white", linewidth=0.7, zorder=3
        )
        ax.scatter(
            r.youth, i, s=85, color=RED, edgecolor="white", linewidth=0.7, zorder=3
        )
        xr = max(r.parent, r.youth)
        ax.annotate(
            f"κ={r.kappa:.2f}",
            (xr, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color="#444",
        )
    ax.set_yticks(range(len(D)))
    ax.set_yticklabels([f"{r.category}" for _, r in D.iterrows()])
    ax.set_xlabel("Prevalence (%)")
    ax.set_ylim(-0.6, len(D) - 0.4)
    hl = [
        Line2D([0], [0], marker="o", ls="", mfc=BLUE, mec="white", label="Caregiver"),
        Line2D([0], [0], marker="o", ls="", mfc=RED, mec="white", label="Youth"),
    ]
    ax.legend(
        handles=hl,
        loc="lower right",
        frameon=False,
        fontsize=9,
        title="Informant",
        title_fontsize=9,
    )
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(
            config.FIGURES_OUT / f"Figure_informant_dumbbell.{e}",
            dpi=300,
            bbox_inches="tight",
        )
    print("wrote Figure_informant_dumbbell.png/pdf")


# diverging concordance (who drives each diagnosis) at ses-06A
def fig_diverging():
    d = CONC[CONC.session == "ses-06A"].copy()
    d = (
        d.set_index("category")
        .reindex([c for c in ORDER if c in d.category.values])
        .dropna(how="all")
        .reset_index()
    )
    d["p_share"] = 100 * d.parent_only / d.union_pos
    d["y_share"] = 100 * d.youth_only / d.union_pos
    d["b_share"] = 100 * d.both_pos / d.union_pos
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = range(len(d))
    ax.barh(y, -d.p_share, color=BLUE, label="Caregiver only")
    ax.barh(y, d.y_share, color=RED, label="Youth only")
    ax.barh(y, d.b_share, left=-d.b_share / 2, color="#999999", label="Both agree")
    ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(d.category)
    # diverging layout draws caregiver bars at negative x; label both sides as positive shares
    M = np.ceil(max(d.p_share.max(), d.y_share.max()) / 5) * 5
    ax.set_xlim(-M, M)
    xt = np.arange(-75, 76, 25)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{abs(int(t))}" for t in xt])
    ax.set_xlabel("Share of positive cases (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(
            config.FIGURES_OUT / f"Figure_informant_diverging.{e}",
            dpi=300,
            bbox_inches="tight",
        )
    print("wrote Figure_informant_diverging.png/pdf")


if __name__ == "__main__":
    fig_trajectories()
    fig_dumbbell()
    fig_diverging()

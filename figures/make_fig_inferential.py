import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager, gridspec
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
R = pd.read_csv(config.DERIV / "inferential_specs.csv").dropna(subset=["OR"])
S = pd.read_csv(config.DERIV / "inferential_summary.csv")

fig = plt.figure(figsize=(12.5, 8.6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.35], wspace=0.32)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])

# --- panel a: hero specification curve, Female -> Depression ---
h = R[(R.predictor == "Female (vs male)") & (R.construct == "depression")].copy()
h = h.sort_values("OR").reset_index(drop=True)
for i, row in h.iterrows():
    c = BLUE if row.OR < 1 else RED
    axa.scatter(
        i,
        row.OR,
        s=70,
        color=(c if row.sig else "white"),
        edgecolor=c,
        linewidth=1.4,
        zorder=3,
    )
axa.axhline(1.0, color=GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
axa.set_xticks([])
axa.set_xlabel("Operationalizations, ordered by odds ratio")
axa.set_ylabel("Odds ratio: female vs male, depression")
axa.set_xlim(-0.7, len(h) - 0.3)
lg = [
    Line2D(
        [0], [0], marker="o", ls="", mfc=BLUE, mec=BLUE, label="Significant (p<.05)"
    ),
    Line2D([0], [0], marker="o", ls="", mfc="white", mec=GREY, label="Non-significant"),
]
axa.legend(handles=lg, loc="upper left", frameon=False, fontsize=8.5, handletextpad=0.3)
axa.text(
    0.02,
    0.02,
    "OR>1: girls higher\nOR<1: girls lower",
    transform=axa.transAxes,
    fontsize=8,
    color="#666",
    va="bottom",
)
axa.set_title("a", loc="left", fontweight="bold", fontsize=12)


# --- panel b: vibration of effects, grouped into per-variable buckets ---
def lab(r):
    p = (
        r.predictor.replace("Female (vs male)", "Female")
        .replace(" (per SD)", "")
        .replace("Race: ", "")
    )
    return f"{p} - {r.construct_label}"


S = S.copy()
S["label"] = S.apply(lab, axis=1)
med = R.groupby(["predictor", "construct"]).OR.median().rename("med")
S = S.merge(med, on=["predictor", "construct"], how="left")
yt, ylab, y = [], [], 0
seps = []
for bk in ["Sex", "Income", "Race/ethnicity", "Culture/environment", "Neuroimaging"]:
    g = S[S.bucket == bk].sort_values("med")
    if g.empty:
        continue
    for _, row in g.iterrows():
        c = RED if row.sign_flip else BLUE
        axb.plot(
            [row.OR_min, row.OR_max],
            [y, y],
            color=c,
            lw=2.4,
            alpha=0.85,
            solid_capstyle="round",
            zorder=2,
        )
        axb.scatter(row.med, y, s=16, color=c, zorder=3)
        yt.append(y)
        ylab.append(row.label)
        y += 1
    axb.text(
        0.205,
        y - len(g) / 2 - 0.5,
        bk,
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#333",
    )
    seps.append(y - 0.5)
    y += 0.8
axb.axvline(1.0, color="#333", lw=1.0, zorder=1)
for s in seps[:-1]:
    axb.axhline(s, color="#dddddd", lw=0.8, zorder=0)
axb.set_yticks(yt)
axb.set_yticklabels(ylab, fontsize=6.3)
axb.set_xscale("log")
axb.set_xlabel("Odds ratio across operationalizations (log scale)")
axb.set_ylim(-0.8, y - 0.8)
axb.set_xticks([0.25, 0.5, 1, 2, 4])
axb.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
axb.set_xlim(0.2, 5.5)
lg2 = [
    Line2D([0], [0], color=RED, lw=2.4, label="Sign flips (crosses OR=1)"),
    Line2D([0], [0], color=BLUE, lw=2.4, label="Robust direction"),
]
axb.legend(handles=lg2, loc="lower right", frameon=False, fontsize=8.5)
axb.set_title("b", loc="left", fontweight="bold", fontsize=12)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(
        config.FIGURES_OUT / f"Figure_inferential.{e}", dpi=300, bbox_inches="tight"
    )
print("wrote Figure_inferential.png/pdf")

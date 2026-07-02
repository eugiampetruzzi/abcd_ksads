import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager, gridspec
from abcd_ksads import config

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
BLUE, RED, GREEN, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#888888",
)  # Okabe-Ito colorblind-safe
STATUS_COLOR = {"current": BLUE, "ever_met": RED}
STATUS_LAB = {"current": "Current episode", "ever_met": "Ever-met (lifetime)"}
INF_MARK = {"parent": "o", "youth": "s", "either": "^", "both": "D"}
INF_LAB = {"parent": "Parent", "youth": "Youth", "either": "Either", "both": "Both"}

config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(config.DERIV / "multiverse_grid.csv")

fig = plt.figure(figsize=(12.5, 5.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.30)
axa = fig.add_subplot(gs[0])
axb = fig.add_subplot(gs[1])

# --- panel a: any-disorder, varying only timeframe x informant (full criteria; threshold not varied) ---
a = (
    g[(g.construct == "any-disorder") & (g.threshold == "full")]
    .sort_values("prevalence_pct")
    .reset_index(drop=True)
)
med = a.prevalence_pct.median()
axa.axhline(med, color=GREY, lw=1.0, ls=(0, (5, 4)), zorder=1)
axa.text(
    len(a) - 0.4, med + 1.0, "Median", ha="right", va="bottom", color=GREY, fontsize=8.5
)
for i, row in a.iterrows():
    axa.scatter(
        i,
        row.prevalence_pct,
        s=48,
        color=STATUS_COLOR.get(row.status, GREY),
        marker=INF_MARK.get(row.informant, "o"),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
axa.set_xticks([])
axa.set_xlabel("Any-disorder operationalizations, ordered by prevalence")
axa.set_ylabel("Baseline prevalence (%)")
axa.set_xlim(-0.7, len(a) - 0.3)
axa.set_ylim(0, a.prevalence_pct.max() * 1.10)
hs = [
    Line2D([0], [0], marker="o", ls="", mfc=c, mec="white", label=STATUS_LAB[k])
    for k, c in STATUS_COLOR.items()
]
hi = [
    Line2D([0], [0], marker=m, ls="", mfc=GREY, mec="white", label=INF_LAB[k])
    for k, m in INF_MARK.items()
]
l1 = axa.legend(
    handles=hs,
    loc="upper left",
    frameon=False,
    fontsize=9,
    title="Timeframe",
    title_fontsize=9,
    handletextpad=0.2,
)
l1._legend_box.align = "left"
axa.add_artist(l1)
l2 = axa.legend(
    handles=hi,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.74),
    frameon=False,
    fontsize=9,
    title="Informant",
    title_fontsize=9,
    ncol=2,
    handletextpad=0.2,
    columnspacing=1.0,
)
l2._legend_box.align = "left"
axa.set_title("a", loc="left", fontweight="bold", fontsize=12)

# --- panel b: top-5 literature constructs, operationalizations used in practice (caregiver / either, full criteria) ---
INF_B = {"parent": "o", "either": "^"}
INF_BLAB = {"parent": "Caregiver", "either": "Either"}
ROWS = [
    ("suicidality", "Suicidality"),
    ("eating", "Eating disorders"),
    ("depression", "Depression"),
    ("any-disorder", "Any disorder"),
    ("ADHD", "ADHD"),
]
b = g[g.informant.isin(INF_B) & (g.threshold == "full")]
yh = 0.30
for i, (con, lab) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    sub = b[b.construct == con]
    if sub.empty:
        continue
    xs = sub.prevalence_pct.values
    axb.plot(
        [xs.min(), xs.max()],
        [y, y],
        color=GREY,
        lw=1.6,
        zorder=1,
        solid_capstyle="round",
    )
    order = sub.sort_values("prevalence_pct").reset_index(drop=True)
    dy = np.linspace(-yh, yh, len(order)) if len(order) > 1 else [0]
    for k, row in order.iterrows():
        axb.scatter(
            row.prevalence_pct,
            y + dy[k],
            s=44,
            color=STATUS_COLOR.get(row.status, GREY),
            marker=INF_B.get(row.informant, "o"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
axb.set_yticks(range(len(ROWS)))
axb.set_yticklabels([l for _, l in ROWS][::-1])
axb.set_xlabel("Baseline prevalence (%)")
axb.set_xlim(-1, 55)
axb.set_ylim(-0.6, len(ROWS) - 0.4)
hbi = [
    Line2D([0], [0], marker=m, ls="", mfc=GREY, mec="white", label=INF_BLAB[k])
    for k, m in INF_B.items()
]
lb = axb.legend(
    handles=hs,
    loc="upper right",
    bbox_to_anchor=(1.0, 1.02),
    frameon=False,
    fontsize=9,
    title="Timeframe",
    title_fontsize=9,
    handletextpad=0.2,
)
axb.add_artist(lb)
axb.legend(
    handles=hbi,
    loc="upper right",
    bbox_to_anchor=(1.0, 0.72),
    frameon=False,
    fontsize=9,
    title="Informant",
    title_fontsize=9,
    handletextpad=0.2,
)
axb.set_title("b", loc="left", fontweight="bold", fontsize=12)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(
        config.FIGURES_OUT / f"Figure1_2_combined.{e}", dpi=300, bbox_inches="tight"
    )
print("wrote Figure1_2_combined.png/pdf")

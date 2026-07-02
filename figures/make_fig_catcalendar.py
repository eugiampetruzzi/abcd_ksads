import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager
from abcd_ksads import config

config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
for fam in ("Arial", "Helvetica"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        break
plt.rcParams.update({"font.size": 9.5})
BLUE, RED, GREY = "#0072B2", "#E69F00", "#CCCCCC"  # Okabe-Ito colorblind-safe
SES = [
    "ses-00A",
    "ses-01A",
    "ses-02A",
    "ses-03A",
    "ses-04A",
    "ses-05A",
    "ses-06A",
    "ses-07A",
]

# DSM category -> (N, diagnosis list, parent modules, youth modules)
CATS = [
    (
        "Anxiety",
        6,
        "GAD; separation anxiety; social anxiety; panic; agoraphobia; specific phobia",
        ["gad", "sepanx", "socanx", "panic", "agor", "phobia"],
        ["gad", "socanx", "panic"],
    ),
    (
        "Eating",
        3,
        "Anorexia nervosa; bulimia nervosa; binge-eating disorder",
        ["ed"],
        ["ed"],
    ),
    ("Psychosis", 3, "Schizophrenia; schizoaffective; schizophreniform", ["psych"], []),
    ("Tic", 3, "Tourette's; persistent tic; provisional tic", ["tic"], []),
    (
        "Depression",
        2,
        "Major depressive disorder; persistent depressive disorder",
        ["dep"],
        ["dep"],
    ),
    ("Bipolar", 2, "Bipolar I; bipolar II", ["bpd"], ["bpd"]),
    ("ADHD", 1, "Attention-deficit/hyperactivity disorder", ["adhd"], []),
    ("ODD", 1, "Oppositional defiant disorder", ["odd"], []),
    ("Conduct", 1, "Conduct disorder", ["cond"], ["cond"]),
    ("DMDD", 1, "Disruptive mood dysregulation disorder", ["dmdd"], ["dmdd"]),
    ("OCD", 1, "Obsessive-compulsive disorder", ["ocd"], ["ocd"]),
    ("PTSD", 1, "Post-traumatic stress disorder", ["ptsd"], ["ptsd"]),
    ("Autism", 1, "Autism spectrum disorder", ["asd"], []),
]
cal = pd.read_csv(config.DERIV / "ksads_administration_calendar.csv")


def color(mods, inf, ses):
    if not mods:
        return GREY  # module not part of this informant's interview
    sub = cal[
        (cal.informant == inf) & (cal.module.isin(mods)) & (cal.session_id == ses)
    ]
    return BLUE if (sub.status == "administered").any() else RED


n = len(CATS)
fig, ax = plt.subplots(figsize=(11.6, 7.0))
x0 = 7.1
cw = 0.62
gap = 0.06
hh = 0.34
cN, cDx = 1.95, 2.35  # x positions for N and Diagnoses columns
for i, (cat, N, dx, pmods, ymods) in enumerate(CATS):
    y = n - 1 - i
    ax.text(-0.05, y, cat, ha="left", va="center", fontweight="bold", fontsize=10)
    ax.text(cN, y, str(N), ha="center", va="center", fontsize=9.5)
    ax.text(cDx, y, dx, ha="left", va="center", fontsize=7.5, color="#333")
    for j, ses in enumerate(SES):
        xc = x0 + j * (cw + gap)
        ax.add_patch(
            Rectangle(
                (xc, y),
                cw,
                hh,
                facecolor=color(pmods, "parent", ses),
                edgecolor="white",
                lw=0.8,
            )
        )  # parent top
        ax.add_patch(
            Rectangle(
                (xc, y - hh),
                cw,
                hh,
                facecolor=color(ymods, "youth", ses),
                edgecolor="white",
                lw=0.8,
            )
        )  # youth bottom
for j, ses in enumerate(SES):
    ax.text(
        x0 + j * (cw + gap) + cw / 2,
        n - 0.32,
        ses.replace("ses-", ""),
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
# explicit parent/youth labels to the right of the first (Anxiety) row
xr = x0 + 8 * (cw + gap) + 0.12
ax.annotate(
    "Parent",
    (xr - 0.06, n - 1 + hh / 2),
    xytext=(xr + 0.18, n - 1 + hh / 2),
    va="center",
    ha="left",
    fontsize=8.5,
    color="#444",
    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
)
ax.annotate(
    "Youth",
    (xr - 0.06, n - 1 - hh / 2),
    xytext=(xr + 0.18, n - 1 - hh / 2),
    va="center",
    ha="left",
    fontsize=8.5,
    color="#444",
    arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
)
ax.text(
    -0.05,
    n - 0.05,
    "DSM category",
    ha="left",
    va="bottom",
    fontweight="bold",
    fontsize=9.5,
)
ax.text(cN, n - 0.05, "N", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
ax.text(
    cDx, n - 0.05, "Diagnoses", ha="left", va="bottom", fontweight="bold", fontsize=9.5
)
ax.text(
    x0 + 3 * (cw + gap) + cw / 2,
    n + 0.78,
    "Session",
    ha="center",
    va="bottom",
    fontweight="bold",
    fontsize=9.5,
)
# version boundary between ses-02A and ses-03A
xb = x0 + 3 * (cw + gap) - gap / 2
ax.plot([xb, xb], [-0.6, n + 0.28], color="#444", lw=1.2, ls=(0, (4, 3)))
ax.text(
    x0 + 1 * (cw + gap),
    n + 0.32,
    "KSADS-COMP 1.0",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#444",
)
ax.text(
    x0 + 5.4 * (cw + gap),
    n + 0.32,
    "KSADS-COMP 2.0",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#444",
)
# legend (color states)
leg = [
    Line2D(
        [0], [0], marker="s", ls="", mfc=BLUE, mec="white", ms=11, label="Administered"
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        ls="",
        mfc=RED,
        mec="white",
        ms=11,
        label="Not administered",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        ls="",
        mfc=GREY,
        mec="white",
        ms=11,
        label="Not administered to youth",
    ),
]
ax.legend(
    handles=leg,
    loc="lower left",
    bbox_to_anchor=(0.0, -0.12),
    frameon=False,
    ncol=3,
    fontsize=9,
    handletextpad=0.3,
    columnspacing=1.4,
)
ax.set_xlim(-0.4, x0 + 8 * (cw + gap) + 1.4)
ax.set_ylim(-0.95, n + 0.7)
ax.axis("off")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(
        config.FIGURES_OUT / f"Figure_category_calendar.{e}",
        dpi=300,
        bbox_inches="tight",
    )
print("wrote Figure_category_calendar.png/pdf")

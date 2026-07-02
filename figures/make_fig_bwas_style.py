import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager, gridspec
from abcd_ksads import config

config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
for fam in ("Arial", "Helvetica"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})

# bucket -> color (Okabe-Ito 5-class, colorblind-safe & mutually distinct)
CMAP = {"Sex": "#E69F00", "Income": "#009E73", "Race/ethnicity": "#56B4E9",
        "Culture/environment": "#CC79A7", "Neuroimaging": "#D55E00"}
ORDER = ["Sex", "Income", "Race/ethnicity", "Culture/environment", "Neuroimaging"]

R = pd.read_csv(config.DERIV / "inferential_specs.csv").dropna(subset=["OR"]).copy()
R["abslog"] = np.abs(np.log(R.OR))      # association strength, like |r_BWAS|

# per pair (predictor x construct): all spec strengths, summary stats
grp = R.groupby(["bucket", "predictor", "construct"])
pairs = grp.abslog.agg(med="median", mn="min", mx="max").reset_index()

rng = np.random.RandomState(0)

fig = plt.figure(figsize=(9.0, 7.2))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.28)
axC = fig.add_subplot(gs[0]); axD = fig.add_subplot(gs[1])

# ---- Shared ordering: both panels ranked by consolidated (median) strength, so each
#      cloud in (a) sits directly above its range bar in (b).
cord = pairs.sort_values("med").reset_index(drop=True)
xpos = {(r.bucket, r.predictor, r.construct): i for i, r in cord.iterrows()}
for _, row in R.iterrows():
    key = (row.bucket, row.predictor, row.construct)
    if key not in xpos:
        continue
    x = xpos[key] + rng.uniform(-0.28, 0.28)
    axC.scatter(x, row.abslog, s=9, color=CMAP[row.bucket], alpha=0.45,
                edgecolor="none", zorder=2)
axC.set_xlim(-1, len(cord)); axC.set_ylim(0, R.abslog.max() * 1.05)
axC.set_xticks([])
axC.set_xlabel("Predictor × disorder pairs")
axC.set_ylabel("Association strength  |ln(OR)|")

# ---- Panel D: consolidated effect (median across specs) per pair, same ordering as (a)
dord = cord
for i, row in dord.iterrows():
    axD.plot([i, i], [row.mn, row.mx], color="#cccccc", lw=1.0, zorder=1)
    axD.scatter(i, row.med, s=26, color=CMAP[row.bucket], edgecolor="white",
                linewidth=0.4, zorder=3)
axD.set_xlim(-1, len(dord)); axD.set_ylim(0, R.abslog.max() * 1.05)
axD.set_xticks([])
axD.set_xlabel("Predictor × disorder pairs")
axD.set_ylabel("Consolidated strength  |ln(OR)|")

handles = [Line2D([0], [0], marker="o", ls="", mfc=CMAP[b], mec="white",
                  ms=7, label=b) for b in ORDER]
axC.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.2,
           handletextpad=0.2, labelspacing=0.3)

for e in ("png", "pdf"):
    fig.savefig(config.FIGURES_OUT / f"Figure_bwas_style.{e}", dpi=300, bbox_inches="tight")
print("wrote Figure_bwas_style.png/pdf  |  pairs:", len(pairs))
"""
Question 3 — figures and analysis snippets from the prepared SD subset.

Reads sd_his_2021.h5 / sd_rn_adj.npy / sd_meta.csv (paths relative to the
LARGEST_RAW environment variable, default /home/hussain/DL4TSG/data/largest)
and writes
  q3/figures/fig_week_sensors.pdf      (a) three sensors over one week
  q3/figures/fig_adjacency_heatmap.pdf (b) adjacency heatmap
  q3/figures/fig_degree_distribution.pdf (b) neighbor-count distribution
and q3/results/q3_analysis.json with the numbers the report quotes.
The figures are also copied to figures/ for the report build.
"""
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW = Path(os.environ.get("LARGEST_RAW", "/home/hussain/DL4TSG/data/largest")) / "sd"
HERE = Path(__file__).resolve()
Q3 = HERE.parents[1]
FIGDIR = Q3 / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESULTS = Q3 / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
REPO_FIGURES = HERE.parents[2] / "figures"
REPO_FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

meta = pd.read_csv(RAW / "sd_meta.csv")
adj = np.load(RAW / "sd_rn_adj.npy")
n = adj.shape[0]
off = adj.copy()
np.fill_diagonal(off, 0.0)
deg = (off != 0).sum(axis=1)

# ---------------------------------------------------------------- figure (a)
WEEK_START, WEEK_END = "2021-06-07", "2021-06-13"     # one full week, Mon-Sun
his = pd.read_hdf(RAW / "sd_his_2021.h5", key="t")
week = his[WEEK_START:WEEK_END]
sel = {"I-5": "1114091", "I-8": "1111514", "I-15": "1119689"}
fig, ax = plt.subplots(figsize=(7.0, 3.0))
for label, sid in sel.items():
    ax.plot(week.index, week[sid].astype(np.float32), linewidth=0.7, label=f"{label} ({sid})")
ax.set_ylabel("traffic flow (vehicles / 15 min)")
ax.set_xlabel("time")
ax.set_title(f"Three SD sensors over one week ({WEEK_START} to {WEEK_END})")
ax.legend(loc="upper right", fontsize=7)
fig.tight_layout()
figa = FIGDIR / "fig_week_sensors.pdf"
fig.savefig(figa, bbox_inches="tight")
plt.close(fig)
print("wrote", figa, "week samples:", len(week))

# ---------------------------------------------------------------- figure (b)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(adj, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
ax.set_xlabel("sensor index (SD order)")
ax.set_ylabel("sensor index (SD order)")
ax.set_title("Road-distance adjacency (716 x 716)")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("adjacency weight")
figb = FIGDIR / "fig_adjacency_heatmap.pdf"
fig.savefig(figb, bbox_inches="tight")
plt.close(fig)
print("wrote", figb)

fig, ax = plt.subplots(figsize=(7.0, 2.8))
ax.hist(deg, bins=np.arange(0, deg.max() + 2), color="steelblue", edgecolor="white", linewidth=0.4)
ax.axvline(deg.mean(), color="crimson", linestyle="--", linewidth=1.2, label=f"mean = {deg.mean():.1f}")
ax.set_xlabel("number of neighbors (A$_{ij} \\neq$ 0, self excluded)")
ax.set_ylabel("sensors")
ax.set_title("Degree distribution of the SD road-distance graph")
ax.legend()
fig.tight_layout()
figc = FIGDIR / "fig_degree_distribution.pdf"
fig.savefig(figc, bbox_inches="tight")
plt.close(fig)
print("wrote", figc)

# ------------------------------------------------------- one/two-hop example
# pick the sensor whose degree is closest to the median degree
med = float(np.median(deg))
i_star = int(np.argmin(np.abs(deg - med)))
one_hop = np.flatnonzero(off[i_star] != 0)
# two-hop-only set: nodes reachable in <=2 steps, excluding the node itself and its one-hop set
reach2 = set(one_hop.tolist())
cur = set(one_hop.tolist())
frontier = cur
for _ in range(1):  # one more step
    nxt = set()
    for u in frontier:
        nxt |= set(np.flatnonzero(off[u] != 0).tolist())
    reach2 |= nxt
    frontier = nxt
two_hop = sorted(reach2 - {i_star} - set(one_hop.tolist()))
meta_row = meta.iloc[i_star]
analysis = {
    "selected_sensor": {
        "index": i_star, "ID": int(meta_row.ID), "highway": meta_row.Fwy,
        "direction": meta_row.Direction, "county": meta_row.County,
        "district": int(meta_row.District), "ID2": int(meta_row.ID2),
        "degree": int(deg[i_star]),
        "one_hop_neighbors": [int(meta.iloc[j].ID) for j in one_hop],
        "one_hop_count": int(len(one_hop)),
        "two_hop_count": int(len(two_hop)),
        "two_hop_neighbors_total_excl_self": int(len(reach2 - {i_star})),
    },
    "degree": {"mean": float(deg.mean()), "min": int(deg.min()), "max": int(deg.max()),
               "median": med, "zero_neighbor_sensors": int((deg == 0).sum())},
    "week": {"start": WEEK_START, "end": WEEK_END, "samples": int(len(week)),
             "sensors": sel},
}
with open(RESULTS / "q3_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)
print("Wrote", RESULTS / "q3_analysis.json")
print(json.dumps({k: v for k, v in analysis["selected_sensor"].items() if k != "one_hop_neighbors"}, indent=2))

for name in ("fig_week_sensors.pdf", "fig_adjacency_heatmap.pdf", "fig_degree_distribution.pdf"):
    shutil.copy(FIGDIR / name, REPO_FIGURES / name)
print("copied figures to figures/")

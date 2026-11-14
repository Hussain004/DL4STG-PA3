"""
Question 5(b) — visualize the learned adjacency matrix.

Reads /home/hussain/DL4TSG/data/largest/sd/{sd_rn_adj.npy, q5_adaptive_adj.npy,
q5_hybrid_adj.npy} and writes
  q5/figures/fig_adjacency_learnt.pdf
and prints comparison statistics quoted in the report.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RAW = Path(os.environ.get("LARGEST_RAW", "/home/hussain/DL4TSG/data/largest")) / "sd"
HERE = Path(__file__).resolve()
Q5 = HERE.parents[1]
FIGDIR = Q5 / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
REPO_FIG = HERE.parents[2] / "figures"
REPO_FIG.mkdir(parents=True, exist_ok=True)
RESULTS = Q5 / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

A = np.load(RAW / "sd_rn_adj.npy")
L = np.load(RAW / "q5_adaptive_adj.npy")
print("physical A:", A.shape, "learned L:", L.shape)

offA = A.copy(); np.fill_diagonal(offA, 0.0)
offL = L.copy(); np.fill_diagonal(offL, 0.0)
threshold = 1e-3
nz_A = offA != 0
nz_L = offL > threshold
top10_mass = np.sort(offL, axis=1)[:, -10:].sum(1)
print(f"off-diagonal learned entries above {threshold}: {int(nz_L.sum())} of {offL.size} ({100 * nz_L.sum() / offL.size:.1f}%)")
print(f"physical off-diagonal entries (nonzero): {int(nz_A.sum())}")
print(f"median row mass on 10 largest learned neighbours: {np.median(top10_mass):.3f}")

K = 10
top_in = 0
for i in range(L.shape[0]):
    top = np.argsort(-offL[i])[:K]
    top_in += int(nz_A[i, top].sum())
print(f"top-{K} learned neighbours in physical graph: {top_in}/{L.shape[0] * K} "
      f"({100 * top_in / (L.shape[0] * K):.1f}%)")
# overlap of physical adjacency with the learned graph's above-threshold support
print(f"physical edges inside learned >{threshold}: {int((nz_A & nz_L).sum())} of {nz_A.sum()}")

fig = plt.figure(figsize=(11.6, 4.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.4)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])
for ax, M, title in zip(axes, (A, L), ("(a) physical road-distance adjacency", "(b) learned adjacency")):
    im = ax.imshow(M, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sensor index (SD order)")
    ax.set_ylabel("sensor index (SD order)")
fig.colorbar(im, cax=cax, label="adjacency weight")
figpath = FIGDIR / "fig_adjacency_learnt.pdf"
fig.savefig(figpath, bbox_inches="tight")
plt.close(fig)
print("wrote", figpath)

stats = {
    "off_diag_physical": int(nz_A.sum()),
    "learned_above_1e-3": int(nz_L.sum()),
    "learned_above_1e-3_frac": float(nz_L.sum() / offL.size),
    "median_top10_row_mass": float(np.median(top10_mass)),
    "physical_in_learned_support": int((nz_A & nz_L).sum()),
    "top10_learned_in_physical": top_in, "top10_in_physical_frac": top_in / (L.shape[0] * K),
}
with open(RESULTS / "adjacency_compare.json", "w") as f:
    json.dump(stats, f, indent=2)
print("wrote", RESULTS / "adjacency_compare.json")

import shutil
shutil.copy(figpath, REPO_FIG / "fig_adjacency_learnt.pdf")
print("copied figure to figures/")

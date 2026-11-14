"""
Question 2 — "Which way is the slowdown moving?"
Ring road of 40 sensors; slowdown shifted one sensor against traffic per step:
    x_i(t+1) = x_{i+1}(t)        (indices mod 40)

Computes every number quoted in the report and draws the two heatmaps of
part (a)(ii).  Run:

    python3 q2/code/ring_motion.py

Writes q2/results/ring_motion.json and q2/figures/fig_ring_bump.pdf
(the latter is copied to figures/ for the report build).
"""
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N = 40
RESULTS = Path(__file__).resolve().parents[1] / "results" / "ring_motion.json"
FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Part (a)(i): fit x_hat_i = a x_i + b (x_{i-1} + x_{i+1}) to the isolated bump
# ---------------------------------------------------------------------------
x_spike = np.zeros(N); x_spike[20] = 1.0
y_true_spike = np.roll(x_spike, -1)            # true next snapshot: 1 at sensor 19
A = np.column_stack([x_spike, np.roll(x_spike, 1) + np.roll(x_spike, -1)])
(a_fit, b_fit), _, _, _ = np.linalg.lstsq(A, y_true_spike, rcond=None)
sse_fit = np.sum((y_true_spike - A @ np.array([a_fit, b_fit]))**2)
print(f"Part (a)(i): least-squares fit a = {a_fit:.6f}, b = {b_fit:.6f}, SSE over all 40 sensors = {sse_fit:.6f}")
assert np.allclose([a_fit, b_fit], [0.0, 0.5])
assert np.isclose(sse_fit, 0.5)
# hand-derivation cross-check: predictions at sensors 19, 20, 21 are b, a, b
print(f"            sensor 19: pred b = {b_fit:.3f} (true 1);  sensor 20: pred a = {a_fit:.3f} (true 0);  sensor 21: pred b = {b_fit:.3f} (true 0)")

# ---------------------------------------------------------------------------
# Part (a)(ii): the smooth bump under the fitted layer vs the true motion
# ---------------------------------------------------------------------------
idx = np.arange(N)
x0 = np.exp(-(idx - 20) ** 2 / 8)
pred_seq, true_seq = [x0], [x0]
p, t = x0.copy(), x0.copy()
for step in range(20):
    p = 0.5 * np.roll(p, 1) + 0.5 * np.roll(p, -1)
    t = np.roll(t, -1)
    pred_seq.append(p); true_seq.append(t)
pred_seq, true_seq = np.array(pred_seq), np.array(true_seq)
print(f"Part (a)(ii): peak at t = 20:  model {pred_seq[-1].max():.4f}  vs  true {true_seq[-1].max():.4f}")
print(f"              argmax at t = 20: model {np.argmax(pred_seq[-1])}  vs  true {np.argmax(true_seq[-1])} (20 - 20 mod 40 = 0)")
print(f"              mass at t = 20:   model {pred_seq[-1].sum():.6f}, true {true_seq[-1].sum():.6f}, t = 0: {x0.sum():.6f}")
assert np.isclose(pred_seq[-1].sum(), x0.sum()) and np.isclose(true_seq[-1].sum(), x0.sum())

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharex=True, sharey=True)
vmax = 1.0
for ax, seq, title in zip(axes, (pred_seq, true_seq),
                          ("(a) fitted symmetric layer", "(b) true motion")):
    im = ax.imshow(seq, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax,
                   extent=[-0.5, N - 0.5, 20.5, -0.5], interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("sensor $i$")
    ax.set_xticks([0, 10, 20, 30, 39])
axes[0].set_ylabel("time $t$")
axes[0].set_ylim(20.5, -0.5)
fig.colorbar(im, ax=axes, label="sensor value", shrink=0.85)
figpath = FIGDIR / "fig_ring_bump.pdf"
fig.savefig(figpath, bbox_inches="tight")
plt.close(fig)
print(f"            wrote {figpath}")
# keep the report's \graphicspath{{figures/}} copy in sync
report_fig = Path(__file__).resolve().parents[2] / "figures" / "fig_ring_bump.pdf"
report_fig.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(figpath, report_fig)
print(f"            copied to {report_fig}")

# ---------------------------------------------------------------------------
# Part (b): x_hat = a x_i + w_- x_{i-1} + w_+ x_{i+1}
# Exact for every input iff a = 0, w_- = 0, w_+ = 1; kernel [0, 0, 1]
# ---------------------------------------------------------------------------
max_err_b = 0.0
for _ in range(100):
    x = rng.random(N)
    pred = 0.0 * np.roll(x, 1) + 0.0 * x + 1.0 * np.roll(x, -1)   # the kernel [0, 0, 1]
    max_err_b = max(max_err_b, np.max(np.abs(pred - np.roll(x, -1))))
print(f"Part (b):   a = 0, w_- = 0, w_+ = 1  ->  x_hat = x_(i+1) exactly; kernel [0,0,1] on 100 snapshots: max abs err = {max_err_b}")
assert max_err_b == 0.0

# ---------------------------------------------------------------------------
# Part (c)(i): backward motion, a x_i + b (x_{i-1} + x_{i+1}) + c x_i(t-1)
# x_i(t-1) = x_{i-1}(t)  =>  need a = 0, b = 1, c = -1
# ---------------------------------------------------------------------------
max_err_c1, max_err_c2 = 0.0, 0.0
for _ in range(100):
    base = rng.random(N)
    cur = np.roll(base, -1)               # backward-motion step
    pred = np.roll(cur, 1) + np.roll(cur, -1) - base
    max_err_c1 = max(max_err_c1, np.max(np.abs(pred - np.roll(cur, -1))))
    # part (c)(ii): forward motion, same coefficients
    fwd = np.roll(base, 1)                # forward-motion step
    pred_fwd = np.roll(fwd, 1) + np.roll(fwd, -1) - base
    max_err_c2 = max(max_err_c2, np.max(np.abs(pred_fwd - np.roll(fwd, 1))))
print(f"Part (c)(i): a = 0, b = 1, c = -1  ->  exact on 100 random snapshot pairs: max abs err = {max_err_c1}")
print(f"Part (c)(ii): same coefficients under forward motion, 100 pairs:       max abs err = {max_err_c2}")
assert max_err_c1 < 1e-12 and max_err_c2 < 1e-12

out = {
    "N_sensors": N,
    "part_ai": {"a": float(a_fit), "b": float(b_fit), "sse": float(sse_fit)},
    "part_aii": {
        "x0_mass": float(x0.sum()),
        "model_peak_t20": float(pred_seq[-1].max()),
        "true_peak_t20": float(true_seq[-1].max()),
        "model_argmax_t20": int(np.argmax(pred_seq[-1])),
        "true_argmax_t20": int(np.argmax(true_seq[-1])),
        "model_mass_t20": float(pred_seq[-1].sum()),
        "true_mass_t20": float(true_seq[-1].sum()),
    },
    "part_b": {"a": 0, "w_minus": 0, "w_plus": 1, "kernel_[i-1,i,i+1]": [0, 0, 1]},
    "part_ci": {"a": 0, "b": 1, "c": -1, "random_pairs_exact": 100},
    "part_cii": {"a": 0, "b": 1, "c": -1, "forward_motion_exact": True},
}
RESULTS.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS, "w") as f:
    json.dump(out, f, indent=2)
print("Wrote q2/results/ring_motion.json")

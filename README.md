# AI651 Assignment 3 (Fall 2026)

Muhammad Hussain Habib (27100016)

Deep Learning for Space, Time and Graphs. The assignment has seven questions;
this repository covers **Question 1 (What does a sum remember?)**, **Question 2
(Which way is the slowdown moving?)**, **Question 3 (Dataset Preparation and
Graph Construction)**, **Question 4 (Spatio-Temporal Traffic Forecasting)**,
**Question 5 (Fixed versus Learnable Adjacency Matrices)**, and **Question 6
(Experimental Analysis and Interpretation)**; Question 7 is a checklist of
the submission's required contents and is addressed in the report's Question 7
section.

| path | what it is |
|---|---|
| `report.pdf` | the report: start here |
| `report.tex` | the LaTeX source of the report |
| `figures/` | the figures the report includes |
| `q1/` | the Question 1 material: `q1/code/` (the NumPy verification) and `q1/results/` (its output, quoted in the report) |
| `q2/` | the Question 2 material: `q2/code/`, `q2/results/`, `q2/figures/` |
| `q3/` | the Question 3 material: `q3/code/` (LargeST-SD preparation and figures), `q3/results/`, `q3/figures/` |
| `q4/` | the Question 4 material: `q4/code/` (the STGNN/temporal models), `q4/results/` (per-model JSONs), `q4/figures/` |
| `q5/` | the Question 5 material: `q5/code/` (learned/hybrid adjacency figure), `q5/results/` (adjacency comparison JSON + figures) |
| `q6/` | the Question 6 material: `q6/code/` (adjacency analysis), `q6/results/` (measurements), `q6/figures/` |

## Question 1

Question 1 is about what a graph layer can infer from an aggregated neighbourhood
message. Part (a) shows an affine message remembers only the neighbourhood size
and the total trend, so matching those two numbers forces identical outputs for
every choice of `w`, `c` and `rho`. Part (b) builds a scalar nonlinear message that
stores the three trend counts in separate base-`(K+1)` digits, and verifies in
NumPy that all 164 non-empty count triples for `K = 8` decode exactly. Part (c)
compares sum, mean and max aggregation on one-hot messages for three tasks
(count steady neighbours; majority slowing; any slowing), shows sum solves all
three, mean solves the two proportion-based ones, and max solves only existence,
and gives the explicit counterexample neighbourhoods for every "No".

## Question 2

Question 2 asks whether a graph layer can predict the direction a slowdown is
moving on a 40-sensor ring road, where one true step is `x_i(t+1) = x_{i+1}(t)`.
Part (a) fits a layer whose two neighbour weights are shared, and shows it splits
the isolated slowdown equally and diffuses the smooth bump instead of moving it
(Figure `figures/fig_ring_bump.pdf`). Part (b) gives the two directions separate
weights and gets the exact rule `x_hat = x_{i+1}` with the convolution kernel
`[0, 0, 1]` under circular padding. Part (c) keeps the neighbour weights equal and
instead feeds one step of history; `a = 0, b = 1, c = -1` is exact under the stated
motion rule (and equally under forward motion, by the dual substitution), which
shows that history alone is enough to break the left-right symmetry.

## Question 3

Question 3 works with the LargeST benchmark, San Diego subset (716 sensors,
District 11, 2020 and 2021, 5-minute raw flow resampled to 15 minutes by summation
per the manual's interval-counts rule; the LargeST reference pipeline instead uses
`resample('15T').mean()` and is deliberately not used here). The zero-filled-window
masks (`sd_mask_2020/2021.npy`) are saved alongside the tensors for masked loss and
masked metrics later. It covers data extraction and
missing-value handling, the 716x716 road-distance adjacency submatrix (its
nodes, nonzero edges, non-symmetry, degree distribution and sparsity), and a
one/two-hop neighbourhood walk on one selected sensor, with three one-week
sensor traces and the adjacency/degree figures the report includes. The heavy
artifacts (`sd_his_2020/2021.h5`, `sd_mask_2020/2021.npy`, `sd_rn_adj.npy`) are regenerable on the Lab
PC with `q3/code/prepare_largest.py`; the repo carries the code, the JSON
summaries and the figures.

## Question 4

Question 4 trains a lightweight spatio-temporal model on the SD subset: a per-sensor
temporal convolutional encoder feeding one directed graph convolution (row-normalised
road-distance adjacency) and a linear head, predicting the next four 15-minute traffic
measurements from the previous eight. Normalisation uses the training split only, from the
valid entries; loss and metrics are masked to the valid (non-filled) entries from
Question 3. The physical-adjacency model is the worst of the three graph models (test MAE
147.7 vs 75.2 adaptive-learned, 83.5 hybrid, 75.7 temporal-only), and the report analyses
why the fixed adjacency hurts, with two ablations that test the explanation (a 50% self-
weight split and a dead-neighbour removal).

## Question 5

Question 5 asks whether learning the graph beats asserting it. The same architecture and
protocol as Question 4 are run with three adjacencies: fixed (the physical road-distance
graph, row-normalized), adaptive (`softmax(relu(E1 E2^T))` row-wise, two 716x32 learnable
embeddings), and hybrid (0.5 fixed + 0.5 adaptive, both row-stochastic). The fixed graph
fails as in Question 4; the adaptive model edges every other variant on test (75.19 vs
75.7 temporal-only, 83.5 hybrid — and 147.7 for fixed); the hybrid is worse than
adaptive but far better than fixed; the learned adjacency is concentrated, not dense
(55,200 of 511,940 off-diagonal entries are above 1e-3), and looks nothing like the
physical graph.

## Question 6

Question 6 analyses the four models end to end: the required comparison table, predictions
vs actual for three sensors, the learned graph's structure (top-k overlap with the physical
graph, geographic distances, flow correlations, same-corridor share), and the critical
discussion of why lower error does not mean a graph is a road map, why a hybrid can
dominate but does not here, and the limits of a static graph.

## Reproducing the checks

```bash
python3 q1/code/verify_encodings.py
python3 q2/code/ring_motion.py
# Q3 (on the Lab PC; raw CA flow files and ca_meta.csv / ca_rn_adj.npy in LARGEST_RAW):
export LARGEST_RAW=/path/to/largest
/home/hussain/DL4TSG/venv_pa3/bin/python q3/code/prepare_largest.py
/home/hussain/DL4TSG/venv_pa3/bin/python q3/code/make_q3_figures.py
# Q4 (on the Lab PC; reads the artifacts above):
/home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model stgnn    --seed 0
/home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model temporal --seed 0
# Q5 (on the Lab PC; fixed / adaptive / hybrid, same protocol as Q4):
/home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model stgnn --adjacency fixed    --seed 0 --out q5/results
/home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model stgnn --adjacency adaptive --seed 0 --out q5/results
/home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model stgnn --adjacency hybrid   --seed 0 --out q5/results
# Q5 figure (on the Lab PC):
/home/hussain/DL4TSG/venv_pa3/bin/python q5/code/make_q5_figures.py
# Q6 (on the Lab PC):
/home/hussain/DL4TSG/venv_pa3/bin/python q6/code/q6_analysis.py
```

The Q1/Q2 scripts print the numbers quoted in the report and rewrite their
JSONs under `q1/results/` and `q2/results/`. The Q3--Q6 scripts are run on the
Lab PC against the LargeST files; they expect `LARGEST_RAW` to hold
`ca_his_raw_2020.h5`, `ca_his_raw_2021.h5`, `ca_meta.csv` and `ca_rn_adj.npy`,
and they rewrite the JSONs under the corresponding `q*/results/`. The figures
the report includes are in `figures/`. The data itself comes from
`https://www.kaggle.com/datasets/liuxu77/largest` (see the report, Question 3).
The environment needs NumPy, Matplotlib and PyTables (plus PyTorch for Q4-Q6);
on the Lab PC the venv at `/home/hussain/DL4TSG/venv_pa3` has them.

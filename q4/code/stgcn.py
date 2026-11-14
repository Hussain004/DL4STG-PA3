"""
Question 4 — Spatio-Temporal Traffic Forecasting

Trains a lightweight spatio-temporal forecasting model on the SD subset:
  8 observed 15-minute steps of traffic flow at 716 sensors
  -> temporal convolutional encoder
  -> one directed graph convolution  H = ReLU(A_hat Z Wg)
  -> small MLP head -> next 4 steps at every sensor

Requires (on the Lab PC):
  /home/hussain/DL4TSG/data/largest/sd/{sd_his_2020.h5, sd_his_2021.h5,
                                         sd_mask_2020.npy, sd_mask_2021.npy,
                                         sd_rn_adj.npy}

Usage:
  /home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model stgnn    --seed 0
  /home/hussain/DL4TSG/venv_pa3/bin/python q4/code/stgcn.py --model temporal --seed 0

Writes q4/results/stgcn_<model>_s<seed>.json and q4/figures/fig_q4_predictions.pdf.
The mask convention is sd_mask_*.npy == True means the window was not measured
(all three readings missing, filled with 0); loss and metrics are computed only
on valid entries.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW = Path("/home/hussain/DL4TSG/data/largest/sd")
Q4 = Path(__file__).resolve().parents[1]
RESULTS = Q4 / "results"
FIGDIR = Q4 / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

HISTORY = 8          # t-7 ... t
HORIZON = 4          # t+1 ... t+4
N_SENSORS = 716
DIM = 32
BATCH = 128
EPOCHS = 20
LR = 1e-3


def load_data():
    x2020 = pd.read_hdf(RAW / "sd_his_2020.h5", key="t").to_numpy().astype(np.float32)
    x2021 = pd.read_hdf(RAW / "sd_his_2021.h5", key="t").to_numpy().astype(np.float32)
    m2020 = np.load(RAW / "sd_mask_2020.npy")
    m2021 = np.load(RAW / "sd_mask_2021.npy")
    adj = np.load(RAW / "sd_rn_adj.npy").astype(np.float32)
    splits = {
        "train": (x2020, m2020),
        "val": (x2021[:8640], m2021[:8640]),          # Jan-Mar 2021
        "test": (x2021[8640:], m2021[8640:]),          # Apr-Dec 2021
    }
    return splits, adj


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x, mask):
        self.x = torch.from_numpy(x)
        self.mask = torch.from_numpy(mask)
        self.starts = torch.arange(len(x) - HISTORY - HORIZON + 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i].item()
        return (self.x[s:s + HISTORY],      # 8 x N
                self.x[s + HISTORY:s + HISTORY + HORIZON],  # 4 x N
                self.mask[s + HISTORY:s + HISTORY + HORIZON])  # True = invalid


class STGNN(nn.Module):
    def __init__(self, dim, adj, graph=True, mode="fixed", alpha=0.5, emb=32, dead=None):
        super().__init__()
        self.graph = graph
        self.mode = mode
        self.t1 = nn.Conv1d(1, dim, kernel_size=3, padding=1)
        self.t2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if graph:
            a = np.array(adj, dtype=np.float64)
            if mode == "fixeddrop":
                # drop the sensors that are absent all of 2021 entirely from the graph
                if dead is None:
                    raise ValueError("fixeddrop needs the dead-sensor mask")
                a = a.copy()
                a[:, dead] = 0.0
                np.fill_diagonal(a, 1.0)
            if mode == "fixed50":
                a50 = a / a.sum(1, keepdims=True)
                a = 0.5 * np.eye(a.shape[0]) + 0.5 * a50
            self.register_buffer("adj_fixed", torch.from_numpy(a / a.sum(1, keepdims=True)).float())
            if mode == "hybrid":
                self.alpha = alpha
            if mode in ("adaptive", "hybrid"):
                self.E1 = nn.Parameter(torch.empty(N_SENSORS, emb))
                self.E2 = nn.Parameter(torch.empty(N_SENSORS, emb))
                nn.init.normal_(self.E1, std=0.05)
                nn.init.normal_(self.E2, std=0.05)
            self.wg = nn.Linear(dim, dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, HORIZON))

    def adjacency(self):
        if self.mode in ("fixed", "fixed50", "fixeddrop"):
            return self.adj_fixed
        a = F.softmax(F.relu(self.E1 @ self.E2.T), dim=1)
        if self.mode == "adaptive":
            return a
        return self.alpha * self.adj_fixed + (1.0 - self.alpha) * a

    def forward(self, x):
        # x: (B, 8, N)
        B, T, N = x.shape
        z = x.permute(0, 2, 1).reshape(B * N, 1, T)
        z = self.relu(self.t1(z))
        z = self.relu(self.t2(z))[:, :, -1]          # (B*N, dim)
        z = z.reshape(B, N, -1)                       # (B, N, dim)
        if self.graph:
            z = torch.einsum("ij,bjd->bid", self.adjacency(), z)   # Â Z
            z = self.relu(self.wg(z))                              # H = ReLU(Â Z Wg)
        return self.head(z).permute(0, 2, 1)          # (B, 4, N)


def masked_mae(pred, target, invalid):
    valid = ~invalid
    err = (pred - target).abs() * valid
    return err.sum() / valid.sum()


def evaluate(model, loader, mean, std):
    model.eval()
    tot_abs = tot_sq = tot_n = 0.0
    with torch.no_grad():
        for xb, yb, mb in loader:
            pred = model(xb.cuda()).cpu()
            err = (pred - yb) * (~mb)
            tot_abs += err.abs().sum().item()
            tot_sq += (err ** 2).sum().item()
            tot_n += (~mb).sum().item()
    mae_norm = tot_abs / tot_n
    rmse_norm = (tot_sq / tot_n) ** 0.5
    return mae_norm * std, rmse_norm * std     # original units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["stgnn", "temporal"], required=True)
    ap.add_argument("--adjacency", choices=["fixed", "adaptive", "hybrid", "fixed50", "fixeddrop"], default="fixed")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", type=Path, default=None, help="results dir to write the JSON into")
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args()

    label = {"fixed": "fixed road-distance", "adaptive": "adaptive learned", "hybrid": "hybrid (fixed + learned)",
             "fixed50": "fixed (50% self-weight)", "fixeddrop": "fixed (dead sensors dropped)"}.get(args.adjacency, args.adjacency) if args.model == "stgnn" else "temporal-convolutional"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits, adj = load_data()
    x_train, m_train = splits["train"]
    # training-only normalisation statistics, valid entries only
    valid = ~m_train
    mean, std = float(x_train[valid].mean()), float(x_train[valid].std())
    print(f"train-valid normalisation: mean {mean:.2f}, std {std:.2f}")

    data, loaders = {}, {}
    for name, (x, m) in splits.items():
        x_n = ((x - mean) / std).astype(np.float32)
        ds = WindowDataset(x_n, m)
        data[name] = ds
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=BATCH, shuffle=(name == "train"),
            num_workers=2, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed))

    graph = args.model == "stgnn"
    adj_mode = args.adjacency if graph else "fixed"
    his21 = pd.read_hdf(RAW / "sd_his_2021.h5", key="t")
    dead = (his21.mean(axis=0).to_numpy() == 0.0)
    model = STGNN(DIM, adj, graph=graph, mode=adj_mode, alpha=args.alpha, dead=dead).cuda()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model {args.model}: {n_params} params")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for xb, yb, mb in loaders["train"]:
            xb, yb, mb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True), mb.cuda(non_blocking=True)
            opt.zero_grad()
            pred = model(xb)
            loss = masked_mae(pred, yb, mb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * (~mb).sum().item()
            ep_n += (~mb).sum().item()
        print(f"epoch {epoch:02d} train MAE(norm) {ep_loss / ep_n:.4f}", flush=True)
    train_time = time.time() - t0

    val_mae, val_rmse = evaluate(model, loaders["val"], mean, std)
    test_mae, test_rmse = evaluate(model, loaders["test"], mean, std)
    print(f"val  MAE {val_mae:.4f}  RMSE {val_rmse:.4f}")
    print(f"test MAE {test_mae:.4f}  RMSE {test_rmse:.4f}")

    # per-sensor test MAE for the three report sensors (diagnostic)
    meta = pd.read_csv(RAW / "sd_meta.csv")
    id2col = {int(r.ID): i for i, r in meta.iterrows()}
    model.eval()
    per_err = np.zeros(N_SENSORS)
    per_n = np.zeros(N_SENSORS)
    with torch.no_grad():
        for xb, yb, mb in loaders["test"]:
            pred = model(xb.cuda()).cpu()
            err = (pred - yb).abs() * (~mb)
            per_err += err.sum(dim=(0, 1)).numpy()
            per_n += (~mb).sum(dim=(0, 1)).numpy()
    for slabel, sid in [("I-5", 1114091), ("I-8", 1111514), ("I-15", 1119689)]:
        c = id2col[sid]
        print(f"test MAE sensor {sid} ({slabel}): {per_err[c] / per_n[c] * std:.2f} (valid cells {int(per_n[c])})")

    meta = pd.read_csv(RAW / "sd_meta.csv")
    id2col = {int(r.ID): i for i, r in meta.iterrows()}
    per_err = np.zeros(N_SENSORS)
    per_n = np.zeros(N_SENSORS)
    model.eval()
    with torch.no_grad():
        for xb, yb, mb in loaders["test"]:
            pred = model(xb.cuda()).cpu()
            err = (pred - yb).abs() * (~mb)
            per_err += err.sum(dim=(0, 1)).numpy()
            per_n += (~mb).sum(dim=(0, 1)).numpy()
    per_mae = np.where(per_n > 0, per_err / per_n * std, np.nan)
    per_sensor_mae = {str(int(meta.ID[i])): float(per_mae[i]) for i in range(N_SENSORS)}

    out = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "dim": DIM, "batch": BATCH, "lr": LR,
        "adjacency": args.adjacency if graph else "none", "alpha": args.alpha if args.adjacency == "hybrid" else None,
        "params": n_params, "train_seconds": train_time,
        "seconds_per_epoch": train_time / args.epochs,
        "train_mean": mean, "train_std": std,
        "val_mae": val_mae, "val_rmse": val_rmse,
        "test_mae": test_mae, "test_rmse": test_rmse,
        "windows": {k: len(data[k]) for k in data},
        "per_sensor_test_mae": per_sensor_mae,
        "hardware": "NVIDIA GeForce RTX 5060 Ti (16 GB)",
    }
    out_dir = args.out if args.out is not None else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.adjacency if graph else "temporal"
    path = out_dir / f"stgcn_{tag}_s{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")

    # dump the learned/hybrid adjacency (not part of the repo; regenerable)
    if graph and args.adjacency in ("adaptive", "hybrid"):
        adj_path = RAW / f"q5_{args.adjacency}_adj.npy"
        np.save(adj_path, model.adjacency().detach().cpu().numpy())
        print(f"wrote {adj_path}")

    if args.skip_figure:
        return
    # one test week figure, horizon-1 predictions
    model.eval()
    test_x = splits["test"][0]
    x_n = ((test_x - mean) / std).astype(np.float32)
    meta = pd.read_csv(RAW / "sd_meta.csv")
    id2col = {int(r.ID): i for i, r in meta.iterrows()}
    t_start = (pd.Timestamp("2021-06-07") - pd.Timestamp("2021-04-01")).days * 96
    sel = {"I-5 (1114091)": 1114091, "I-8 (1111514)": 1111514, "I-15 (1119689)": 1119689}
    make_figure(model, x_n, mean, std, id2col, t_start, sel, label)


def make_figure(model, x_n, mean, std, id2col, t_start, sel, label):
    """One week in the test period, actual vs 1-step-ahead prediction."""
    y_true_all, y_pred_all = [], []
    T = len(x_n)
    with torch.no_grad():
        for s in range(t_start, min(t_start + 7 * 96, T - HORIZON)):
            xb = torch.from_numpy(x_n[s:s + HISTORY]).unsqueeze(0).cuda()  # (1, 8, N)
            pred = model(xb).cpu().numpy()[0]  # (4, N)
            y_true_all.append(x_n[s + HISTORY].copy())
            y_pred_all.append(pred[0].copy())  # horizon 1
    yt = np.array(y_true_all) * std + mean
    yp = np.array(y_pred_all) * std + mean
    times = pd.date_range("2021-06-07", periods=len(yt), freq="15min")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))
    for ax, (panel_label, sid) in zip(axes, sel.items()):
        col = id2col[sid]
        ax.plot(times, yt[:, col], linewidth=0.7, label="actual", color="tab:blue")
        ax.plot(times, yp[:, col], linewidth=0.7, label="predicted", color="tab:orange", alpha=0.8)
        ax.set_title(panel_label, fontsize=8)
        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=6)
    axes[0].set_ylabel("traffic flow (vehicles / 15 min)")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"STGNN ({label} adjacency), one test week, horizon 1")
    fig.tight_layout()
    figpath = FIGDIR / "fig_q4_predictions.pdf"
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figpath}")


if __name__ == "__main__":
    main()

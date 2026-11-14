import json
import numpy as np, pandas as pd

A = np.load('/home/hussain/DL4TSG/data/largest/sd/sd_rn_adj.npy')
L = np.load('/home/hussain/DL4TSG/data/largest/sd/q5_adaptive_adj.npy')
meta = pd.read_csv('/home/hussain/DL4TSG/data/largest/sd/sd_meta.csv')
his = pd.read_hdf('/home/hussain/DL4TSG/data/largest/sd/sd_his_2021.h5', key='t')
X = his.to_numpy().astype(np.float64)
std = X.std(axis=0); keep = std > 0
Xc = X[:, keep] - X[:, keep].mean(0)
cn = np.corrcoef(Xc.T); np.fill_diagonal(cn, 0.0)

offA = A.copy(); np.fill_diagonal(offA, 0.0); nzA = offA != 0
offL = L.copy(); np.fill_diagonal(offL, 0.0); nzL = offL != 0
out = {"top_k_learned_in_physical": {}, "phys_neighbors_in_learned_topk": {}}
for K in (1, 5, 10, 20):
    tl = np.argsort(-offL, axis=1)[:, :K]
    inphys = np.take_along_axis(nzA, tl, axis=1).sum()
    out["top_k_learned_in_physical"][f"top{K}"] = float(inphys) / (716 * K)
for K in (10, 20):
    tot_phys, retained = int(nzA.sum()), 0
    tl = np.argsort(-offL, axis=1)[:, :K]
    for i in range(716):
        p = np.flatnonzero(nzA[i]); retained += int(np.isin(p, tl[i]).sum())
    out["phys_neighbors_in_learned_topk"][f"top{K}"] = float(retained) / tot_phys
lat = np.radians(meta.Lat.to_numpy()); lng = np.radians(meta.Lng.to_numpy())
all_d = np.zeros((716, 716))
for i in range(716):
    dlat = lat - lat[i]; dlng = lng - lng[i]
    a = np.sin(dlat/2)**2 + np.cos(lat[i])*np.cos(lat)*np.sin(dlng/2)**2
    all_d[i] = 2 * 6371.0 * np.arcsin(np.sqrt(a))
np.fill_diagonal(all_d, 0.0)
tA = np.argsort(-offA, axis=1)[:, :10]; tL = np.argsort(-offL, axis=1)[:, :10]
out["mean_geodist_physical_top10"] = float(np.array([all_d[i, tA[i]].mean() for i in range(716)]).mean())
out["mean_geodist_learned_top10"] = float(np.array([all_d[i, tL[i]].mean() for i in range(716)]).mean())
out["mean_geodist_random_pair"] = float(all_d[np.triu_indices(716, 1)].mean())
idx_map = np.full(716, -1); idx_map[np.flatnonzero(keep)] = np.arange(keep.sum())
def corr10(t):
    tot, n = 0.0, 0
    for i in range(716):
        if idx_map[i] < 0: continue
        for j in t[i]:
            if idx_map[j] < 0: continue
            tot += abs(cn[idx_map[i], idx_map[j]]); n += 1
    return float(tot / n)
out["mean_absflowcorr_learned_top10"] = corr10(tL)
out["mean_absflowcorr_physical_top10"] = corr10(tA)
fwy = meta.Fwy.str.split('-').str[0].to_numpy()
out["same_corridor_learned_top10"] = float(np.array([np.mean(fwy[tL[i]] == fwy[i]) for i in range(716)]).mean())
out["same_corridor_physical_top10"] = float(np.array([np.mean(fwy[tA[i]] == fwy[i]) for i in range(716)]).mean())

with open("/home/hussain/DL4TSG/PA3/q6/results/q6_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

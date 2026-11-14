"""
Question 3(a)/(b) — LargeST San Diego (SD) data preparation.

Reads the raw CA flow files (5-minute, 2017-2021 PeMS), resamples to
15-minute intervals, extracts the 716 SD sensors (District == 11), builds the
716x716 road-distance adjacency submatrix, and writes small JSON stats the
report quotes.  Heavy artifacts (sd_his_*.h5, sd_rn_adj.npy, sd_mask_*.npy)
go to the same directory as the raw data (default
/home/hussain/DL4TSG/data/largest, override with the LARGEST_RAW environment
variable); only JSONs and figures ship in the repo.

Resampling choice: the raw LargeST flow values are interval vehicle counts
(integer-valued, non-negative; per-lane peaks of ~160-275 per 5-minute step,
i.e. ~2,000 veh/h/lane — plausible as counts, impossible as an hourly rate;
only ~8.3% of positive values are multiples of 12, so they are not hourly
rates built from 5-minute counts).  The manual's rule for counts is
summation, so a 15-minute value is the SUM of the three 5-minute readings.
We use .sum(min_count=3): a window is kept only when all three readings are
present, so no zero is invented for a missing window; missing windows are
then filled with 0 later (see the report for why that remains defensible).
Values are therefore vehicles per 15 minutes, in the same units as vehicles,
i.e. the unit-preserving aggregate.  The mean-based result of the reference
LargeST pipeline would be 1/3 of ours and is deliberately NOT used.

Run on the Lab PC:
    /home/hussain/DL4TSG/venv_pa3/bin/python q3/code/prepare_largest.py
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(os.environ.get("LARGEST_RAW", "/home/hussain/DL4TSG/data/largest"))
HERE = Path(__file__).resolve()
Q3 = HERE.parents[1]                     # .../q3
OUT_ARTIFACTS = RAW / "sd"               # big artifacts live beside the raw data
OUT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS = Q3 / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SD_DISTRICT = 11
YEARS = ("2020", "2021")
CEILING = 999

meta = pd.read_csv(RAW / "ca_meta.csv")
sd_meta = meta[meta.District == SD_DISTRICT].reset_index(drop=True)
sd_ids = sd_meta.ID.astype(str).tolist()          # h5 column names
sd_id2 = sd_meta.ID2.to_numpy()                   # row/col indices into the CA adjacency
assert len(sd_ids) == len(sd_id2) == 716
print(f"SD sensors: {len(sd_ids)} (District {SD_DISTRICT})")
sd_meta.to_csv(OUT_ARTIFACTS / "sd_meta.csv", index=False)

stats = {"sd_sensors": int(len(sd_ids)), "years": {}}
for year in YEARS:
    raw_path = RAW / f"ca_his_raw_{year}.h5"
    print(f"[{year}] reading {raw_path} ...", flush=True)
    df = pd.read_hdf(raw_path, key="t")
    raw_shape = df.shape
    raw_nan = int(df.isna().sum().sum())
    raw_cols_with_nan = int((df.isna().sum() > 0).sum())
    raw_all_nan_cols = int((df.isna().sum() == len(df)).sum())
    raw_min = float(df.min(skipna=True).min())
    sample = df.sample(n=min(2000, len(df)), random_state=0).to_numpy()
    keep = ~np.isnan(sample)
    raw_intlike = bool(np.allclose(sample[keep], np.round(sample[keep]), atol=1e-8))
    del sample
    ceiling_cells = int((df[sd_ids].to_numpy(na_value=np.nan) == CEILING).sum())
    # genuine zero rate among valid (non-NaN) 5-minute readings, and daytime share
    idx = df.index
    daymask = (idx.hour >= 6) & (idx.hour < 19)
    sd_vals = df[sd_ids].to_numpy(na_value=np.nan)
    valid_vals = sd_vals.ravel()
    valid_vals = valid_vals[~np.isnan(valid_vals)]
    genuine_zeros = int((valid_vals == 0).sum())
    zeros_day = int((sd_vals[daymask] == 0).sum())
    day_cells = int((~np.isnan(sd_vals[daymask])).sum())
    del valid_vals, sd_vals
    # gap structure on the SD columns: maximal NaN runs
    sd_nan = df[sd_ids].isna().to_numpy()
    event_lens = []
    for col in range(sd_nan.shape[1]):
        colnan = sd_nan[:, col]; i = 0; n = len(colnan)
        while i < n:
            if colnan[i]:
                j = i
                while j < n and colnan[j]: j += 1
                event_lens.append(j - i); i = j
            else:
                i += 1
    event_lens = np.array(event_lens)
    gap_events = int(len(event_lens))
    frac_events_le_1h = float((event_lens <= 12).sum() / gap_events)
    frac_cells_outage_ge_1d = float(event_lens[event_lens >= 288].sum() / sd_nan.sum())
    per_lane = (df[sd_ids].max(skipna=True) / sd_meta.Lanes.to_numpy()).astype(float)
    print(f"[{year}] raw {raw_shape}, NaN cells {raw_nan} in {raw_cols_with_nan} cols "
          f"(all-missing {raw_all_nan_cols}), min {raw_min}, integer-valued {raw_intlike}, "
          f"max {CEILING} occurs {ceiling_cells} times, per-lane peak median {per_lane.median():.1f}")
    valid_cells = int((~sd_nan).sum())
    print(f"[{year}] genuine zeros {genuine_zeros} ({100*genuine_zeros/valid_cells:.3f}% of valid cells), "
          f"daytime zeros {zeros_day} ({100*zeros_day/day_cells:.3f}% of day cells), "
          f"gap events {gap_events}, events<=1h {100*frac_events_le_1h:.1f}%, "
          f"cells in outages>=1d {100*frac_cells_outage_ge_1d:.1f}%")
    raw_cnt = df[sd_ids].resample("15min").count().to_numpy()
    partial_windows = int(((raw_cnt >= 1) & (raw_cnt <= 2)).sum())
    df15 = df.resample("15min").sum(min_count=3)
    res_shape = df15.shape
    sd_res = df15[sd_ids]
    sd_res_nan = int(sd_res.isna().sum().sum())
    mask = sd_res.isna().to_numpy()
    sd_his = sd_res.fillna(0).astype(np.float32)
    np.save(OUT_ARTIFACTS / f"sd_mask_{year}.npy", mask)
    sd_path = OUT_ARTIFACTS / f"sd_his_{year}.h5"
    sd_his.to_hdf(sd_path, key="t", mode="w")
    sd_zeros = int((sd_his == 0).sum().sum())
    sd_all_zero_cols = int(((sd_his == 0).sum() == len(sd_his)).sum())
    x = sd_his.to_numpy()
    pooled_mean_zeros, pooled_std_zeros = float(x.mean()), float(x.std())
    vm = x[~mask]
    pooled_mean_valid, pooled_std_valid = float(vm.mean()), float(vm.std())
    del vm
    print(f"[{year}] resampled {res_shape} (SD NaN windows {sd_res_nan} = {100*sd_res_nan/mask.size:.2f}% of SD cells, "
          f"partial windows {partial_windows}), SD {sd_his.shape} -> {sd_path}; "
          f"zero cells after fill {sd_zeros} ({100*sd_zeros/sd_his.size:.2f}%), all-zero sensors {sd_all_zero_cols}")
    print(f"[{year}] pooled mean/std with zeros {pooled_mean_zeros:.2f}/{pooled_std_zeros:.2f}; "
          f"excluding filled windows {pooled_mean_valid:.2f}/{pooled_std_valid:.2f}")
    stats["years"][year] = {
        "raw_shape": list(raw_shape), "raw_missing_cells": raw_nan,
        "raw_sd_missing_cells": int(sd_nan.sum()), "raw_sd_missing_frac": float(sd_nan.sum() / sd_nan.size),
        "raw_cols_with_missing": raw_cols_with_nan, "raw_all_missing_cols": raw_all_nan_cols,
        "raw_min_value": raw_min, "raw_integer_valued": raw_intlike,
        "ceiling_999_cells": ceiling_cells,
        "per_lane_peak_median": float(per_lane.median()), "per_lane_peak_p90": float(per_lane.quantile(0.9)),
        "genuine_zero_cells": genuine_zeros, "genuine_zero_frac": float(genuine_zeros / valid_cells),
        "daytime_zero_cells": zeros_day, "daytime_zero_frac_of_day_cells": float(zeros_day / day_cells),
        "gap_events": gap_events, "frac_events_le_1h": frac_events_le_1h,
        "frac_cells_outage_ge_1d": frac_cells_outage_ge_1d,
        "resampled_shape": list(res_shape),
        "resampled_missing_cells": sd_res_nan,
        "resampled_missing_frac": float(sd_res_nan / mask.size),
        "partial_windows": partial_windows, "partial_windows_frac": float(partial_windows / raw_cnt.size),
        "sd_shape": list(sd_his.shape),
        "sd_zero_cells": sd_zeros, "sd_zero_frac": sd_zeros / sd_his.size,
        "sd_all_zero_sensors": sd_all_zero_cols,
        "pooled_mean_with_zeros": pooled_mean_zeros, "pooled_std_with_zeros": pooled_std_zeros,
        "pooled_mean_excluding_filled": pooled_mean_valid, "pooled_std_excluding_filled": pooled_std_valid,
    }
    del df, df15, sd_res, sd_his, x, sd_nan
    gc.collect()

# ------------------------------------------------------------------ adjacency
print("building the SD adjacency submatrix ...")
ca_adj = np.load(RAW / "ca_rn_adj.npy", mmap_mode="r")
sd_adj = np.ascontiguousarray(ca_adj[sd_id2][:, sd_id2]).astype(np.float64)
del ca_adj
np.save(OUT_ARTIFACTS / "sd_rn_adj.npy", sd_adj)
off = sd_adj.copy()
np.fill_diagonal(off, 0.0)
deg = (off != 0).sum(axis=1)
symmetrizable = bool(np.array_equal(sd_adj, sd_adj.T))
max_abs_diff = float(np.abs(sd_adj - sd_adj.T).max())
n_asym = int((sd_adj != sd_adj.T).sum())
stats["adjacency"] = {
    "shape": list(sd_adj.shape),
    "nonzero_entries_incl_diagonal": int(np.count_nonzero(sd_adj)),
    "nonzero_entries_excl_diagonal": int(np.count_nonzero(off)),
    "diagonal_value": float(np.unique(sd_adj.diagonal())[0]),
    "symmetric": symmetrizable,
    "max_abs_diff_vs_transpose": max_abs_diff,
    "asymmetric_entries": n_asym,
    "degree_mean": float(deg.mean()), "degree_min": int(deg.min()),
    "degree_max": int(deg.max()), "degree_median": float(np.median(deg)),
    "sensors_with_zero_neighbors": int((deg == 0).sum()),
    "sparsity_frac_nonzero": float(np.count_nonzero(sd_adj) / sd_adj.size),
}
print(json.dumps(stats["adjacency"], indent=2))
with open(RESULTS / "prepare_largest.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Wrote {RESULTS / 'prepare_largest.json'}")

"""Convert METR-LA / PeMS-BAY sensor data to grid tensors.

Outputs in data/:
  <dataset>_grid_speed.npy        (T, 16, 16) float32, mph
  <dataset>_grid_observed.npy     (T, 16, 16) uint8
  <dataset>_grid_strict_mask.npy  (16, 16) uint8
  <dataset>_grid_dilated_mask.npy (16, 16) uint8
  <dataset>_grid_meta.json

Usage:
  python scripts/prepare_data.py --dataset metrla
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
H = W = 16
K_NEIGHBORS = 4

DATASETS = {
    "metrla": {
        "h5": "metr-la.h5", "key": "df",
        "loc": "graph_sensor_locations.csv", "header": True,
        "prefix": "metrla_grid",
    },
    "pemsbay": {
        "h5": "pems-bay.h5", "key": "speed",
        "loc": "graph_sensor_locations_bay.csv", "header": False,
        "prefix": "pemsbay_grid",
    },
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(dataset: str):
    cfg = DATASETS[dataset]
    with h5py.File(DATA / cfg["h5"], "r") as f:
        g = f[cfg["key"]]
        values = g["block0_values"][:].astype(np.float32)
        cols = [x.decode() if isinstance(x, bytes) else str(x) for x in g["axis0"][:]]
    if cfg["header"]:
        loc = pd.read_csv(DATA / cfg["loc"])
    else:
        loc = pd.read_csv(DATA / cfg["loc"], header=None,
                          names=["sensor_id", "latitude", "longitude"])
    loc["sensor_id"] = loc["sensor_id"].astype(float).astype(int).astype(str)
    return values, cols, loc, cfg


def build_mapping(cols, loc):
    lat = loc["latitude"].to_numpy(float)
    lon = loc["longitude"].to_numpy(float)
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    lat_span = (lat_max - lat_min) * 1.1
    lon_span = (lon_max - lon_min) * 1.1
    lat_origin = lat_max + (lat_max - lat_min) * 0.05
    lon_origin = lon_min - (lon_max - lon_min) * 0.05

    sid_to_cell = {}
    for sid, la, lo in zip(loc["sensor_id"], lat, lon):
        r = min(int((lat_origin - la) / lat_span * H), H - 1)
        c = min(int((lo - lon_origin) / lon_span * W), W - 1)
        sid_to_cell[sid] = (max(0, r), max(0, c))

    cell_to_cols = {}
    for j, sid in enumerate(cols):
        cell = sid_to_cell.get(sid)
        if cell is not None:
            cell_to_cols.setdefault(cell, []).append(j)

    strict = np.zeros((H, W), dtype=np.uint8)
    for r, c in cell_to_cols:
        strict[r, c] = 1
    dilated = ndimage.binary_dilation(strict.astype(bool), iterations=1).astype(np.uint8)
    return sid_to_cell, cell_to_cols, strict, dilated


def aggregate(values, cell_to_cols):
    """每个格点取非零传感器均值；全零=缺失。返回 speed(T,H,W), observed(T,H,W)。"""
    T = values.shape[0]
    speed = np.zeros((T, H, W), dtype=np.float32)
    observed = np.zeros((T, H, W), dtype=np.uint8)
    for (r, c), cols in cell_to_cols.items():
        v = values[:, cols]
        pos = v > 0
        cnt = pos.sum(axis=1)
        total = (v * pos).sum(axis=1)
        ok = cnt > 0
        speed[ok, r, c] = total[ok] / cnt[ok]
        observed[ok, r, c] = 1
    return speed, observed


def forward_fill_time(speed, strict):
    """严格格点上用上一时刻值做因果填充；前导缺失仍为 0。"""
    mask = strict.astype(bool)
    for t in range(1, speed.shape[0]):
        miss = (speed[t] == 0) & mask
        speed[t][miss] = speed[t - 1][miss]
    return speed


def build_spatial_weights(strict, dilated):
    """膨胀格点到严格格点的反距离权重；严格格点不给自己权重。"""
    target = np.argwhere(dilated > 0)
    source = np.argwhere(strict > 0)
    weights = np.zeros((len(target), len(source)), dtype=np.float32)
    src_idx = {tuple(p): i for i, p in enumerate(source)}
    for i, p in enumerate(target):
        d2 = ((source - p) ** 2).sum(axis=1).astype(np.float32)
        self_i = src_idx.get(tuple(p))
        if self_i is not None:
            d2[self_i] = np.inf
        k = min(K_NEIGHBORS, len(source))
        nn = np.argpartition(d2, k - 1)[:k]
        w = 1.0 / (d2[nn] + 1e-6)
        w = w / w.sum()
        weights[i, nn] = w
    return target, source, weights


def spatial_fill(speed, strict, dilated, target, source, weights):
    """同一时刻用有效严格格点插值到缺失的膨胀格点；不跨时间。"""
    sr, sc = source[:, 0], source[:, 1]
    T = speed.shape[0]
    vals = speed[:, sr, sc]                      # (T, S)
    valid = vals > 0
    num = (vals * valid) @ weights.T             # (T, K)
    den = valid.astype(np.float32) @ weights.T   # (T, K)
    spatial = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    empty = valid.sum(axis=1) == 0
    if empty.any():
        for t in np.where(empty)[0]:
            if t > 0:
                speed[t] = speed[t - 1]

    for i, (r, c) in enumerate(target):
        fill = (speed[:, r, c] == 0) & (den[:, i] > 0)
        if fill.any():
            speed[fill, r, c] = spatial[fill, i]
    speed *= dilated
    return speed


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), default="metrla")
    args = p.parse_args()

    raw, cols, loc, cfg = load_raw(args.dataset)
    prefix = cfg["prefix"]
    sid_to_cell, cell_to_cols, strict, dilated = build_mapping(cols, loc)
    speed, observed = aggregate(raw, cell_to_cols)

    n_missing = int(((observed == 0) & (strict > 0)[None]).sum())
    n_observed = int(observed.sum())
    raw_all_zero = int((raw == 0).all(axis=1).sum())

    speed = forward_fill_time(speed, strict)
    target, source, weights = build_spatial_weights(strict, dilated)
    speed = spatial_fill(speed, strict, dilated, target, source, weights)

    np.save(DATA / f"{prefix}_speed.npy", speed)
    np.save(DATA / f"{prefix}_observed.npy", observed)
    np.save(DATA / f"{prefix}_strict_mask.npy", strict)
    np.save(DATA / f"{prefix}_dilated_mask.npy", dilated)

    raw_path = DATA / cfg["h5"]
    meta = {
        "dataset": args.dataset,
        "source": str(raw_path.relative_to(BASE)),
        "source_md5": md5(raw_path),
        "shape": list(speed.shape),
        "sensor_count": int(len(loc)),
        "strict_cells": int(strict.sum()),
        "dilated_cells": int(dilated.sum()),
        "dilation_only_cells": int(dilated.sum() - strict.sum()),
        "observed_values": n_observed,
        "missing_strict_values": n_missing,
        "raw_all_zero_frames": raw_all_zero,
        "aggregation": "mean of non-zero sensors per cell; all-zero cell = missing",
        "time_fill": "causal forward fill on strict cells",
        "spatial_fill": f"inverse-distance {K_NEIGHBORS}-nearest strict cells per frame",
        "expected": {
            "speed_min": float(speed.min()),
            "speed_max": float(speed.max()),
            "speed_mean": float(speed.mean()),
            "zero_inside_dilated": float((speed[:, dilated > 0] == 0).mean()),
            "observed_ratio_strict": float(n_observed / (strict.sum() * speed.shape[0])),
        },
    }
    (DATA / f"{prefix}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

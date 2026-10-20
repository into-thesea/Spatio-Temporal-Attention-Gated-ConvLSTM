"""Draw prediction maps on a basemap for METR-LA and PeMS-BAY.

Requires:
  pip install -r requirements-figures.txt

Usage:
  python scripts/make_map_figures.py \
    --checkpoint-metrla models/base_cyclic_s42.pth \
    --checkpoint-pemsbay models/pemsbay_base_cyclic_s42.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import contextily as ctx
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from data_core.data import get_loaders
from train import build_model

H = W = 16
STEPS = (2, 5, 11)  # 15, 30, 60 minutes


def read_locations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "latitude" not in df.columns:
        df = pd.read_csv(path, header=None,
                         names=["sensor_id", "latitude", "longitude"])
    df["sensor_id"] = df["sensor_id"].astype(float).astype(int).astype(str)
    return df


def sensor_cells(df: pd.DataFrame):
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    lat_min, lat_max = lat.min(), lat.max()
    lon_min, lon_max = lon.min(), lon.max()
    lat_span = (lat_max - lat_min) * 1.1
    lon_span = (lon_max - lon_min) * 1.1
    lat_origin = lat_max + (lat_max - lat_min) * 0.05
    lon_origin = lon_min - (lon_max - lon_min) * 0.05
    rows, cols = [], []
    for la, lo in zip(lat, lon):
        r = min(int((lat_origin - la) / lat_span * H), H - 1)
        c = min(int((lo - lon_origin) / lon_span * W), W - 1)
        rows.append(max(0, r))
        cols.append(max(0, c))
    return np.asarray(rows), np.asarray(cols)


def pick_sample(dataset):
    """Pick a test window with high speed variance at the 60-minute horizon."""
    best_i, best_score = 0, -1.0
    step = len(dataset) // 200 if len(dataset) > 200 else 1
    for i in range(0, len(dataset), max(step, 1)):
        s = int(dataset.starts[i])
        obs = dataset.observed[s + 12 + 11]
        vals = dataset.speed[s + 12 + 11][obs > 0]
        if vals.size:
            score = float(vals.std())
            if score > best_score:
                best_i, best_score = i, score
    return best_i


def timestamp_for(raw_h5: Path, index: int):
    if not raw_h5.exists():
        return None
    try:
        with h5py.File(raw_h5, "r") as f:
            key = "df" if "df" in f else "speed"
            values = f[key]["axis1"][:]
        return pd.to_datetime(values[index], unit="ns")
    except Exception:
        return None


def plot_dataset(name, checkpoint, data_dir, out_dir, raw_h5, zoom=11):
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    loaders = get_loaders(data_dir, dataset=name, batch_size=1,
                          time_mode=cfg["time_encoding"],
                          observed_channel=cfg["observed_channel"])
    model = build_model(cfg["model"], cfg["input_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = loaders["test"].dataset
    i = pick_sample(ds)
    x, y, obs = ds[i]
    with torch.no_grad():
        out = model(x[None])
        pred = out[0] if isinstance(out, tuple) else out
    scaler = loaders["scaler"]
    pred = scaler.inverse(pred[0]).numpy()     # (12,1,16,16)
    true = scaler.inverse(y).numpy()            # (12,1,16,16)
    obs = obs.numpy()                           # (12,1,16,16)

    loc_path = data_dir / (
        "graph_sensor_locations.csv" if name == "metrla"
        else "graph_sensor_locations_bay.csv"
    )
    loc = read_locations(loc_path)
    rows, cols = sensor_cells(loc)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform(loc["longitude"].to_numpy(),
                                   loc["latitude"].to_numpy())
    pad = 2500
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)

    vmax = 70.0 if name == "metrla" else 85.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180)
    for row, (grid, row_label) in enumerate(
        ((true, "Ground truth"), (pred, "Prediction"))
    ):
        for col, step in enumerate(STEPS):
            ax = axes[row, col]
            values = grid[step, 0, rows, cols]
            keep = obs[step, 0, rows, cols] > 0
            ax.scatter(
                xs[keep], ys[keep], c=values[keep],
                cmap="RdYlGn", vmin=0, vmax=vmax,
                s=16, edgecolors="white", linewidths=0.25, zorder=3,
            )
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.HOT,
                                zoom=zoom, attribution=False)
            except Exception as exc:
                print(f"{name}: basemap failed ({exc}); drawing points only")
                ax.set_facecolor("#f8fafc")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{row_label} · {(step + 1) * 5} min", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.025, pad=0.02)
    cbar.set_label("Speed (mph)")

    ts = timestamp_for(raw_h5, int(ds.starts[i]) + 12)
    suffix = f" — {ts:%Y-%m-%d %H:%M}" if ts is not None else ""
    fig.suptitle(f"{name.upper()} predicted speed on the road network{suffix}",
                 fontsize=14, fontweight="bold")
    fig.text(0.012, 0.70, "Ground truth", rotation=90, va="center", fontsize=11)
    fig.text(0.012, 0.28, "Prediction", rotation=90, va="center", fontsize=11)
    fig.text(0.99, 0.01, "Basemap © OpenStreetMap contributors, HOT",
             ha="right", fontsize=7, color="#475569")

    path = out_dir / f"map_prediction_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-metrla", default=None)
    p.add_argument("--checkpoint-pemsbay", default=None)
    p.add_argument("--data-dir", default=str(BASE / "data"))
    p.add_argument("--out-dir", default=str(BASE / "assets"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    if args.checkpoint_metrla:
        plot_dataset("metrla", Path(args.checkpoint_metrla).resolve(),
                     data_dir, out_dir, data_dir / "metr-la.h5", zoom=11)
    if args.checkpoint_pemsbay:
        plot_dataset("pemsbay", Path(args.checkpoint_pemsbay).resolve(),
                     data_dir, out_dir, data_dir / "pems-bay.h5", zoom=11)


if __name__ == "__main__":
    main()

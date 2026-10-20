"""Generate figures for the README.

Horizon curves are built from reference CSV files. Spatial error maps and
prediction examples require checkpoints and grid data.

Usage:
  python scripts/make_figures.py
  python scripts/make_figures.py \
    --checkpoint-metrla models/base_cyclic_s42.pth \
    --checkpoint-pemsbay models/pemsbay_base_cyclic_s42.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from data_core.data import get_loaders
from train import build_model

H = W = 16


def plot_architecture(out_dir: Path):
    from matplotlib.patches import FancyArrowPatch, Polygon

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=180)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    depth, rise = 0.38, 0.22

    def iso_box(x, y, w, h, title, sub, base, top, side):
        ax.add_patch(Polygon(
            [(x + 0.08, y - 0.10), (x + w + 0.08, y - 0.10),
             (x + w + depth + 0.08, y + rise - 0.10),
             (x + depth + 0.08, y + rise - 0.10)],
            closed=True, facecolor="#0f172a", alpha=0.08, edgecolor="none", zorder=1,
        ))
        ax.add_patch(Polygon(
            [(x + w, y), (x + w + depth, y + rise),
             (x + w + depth, y + h + rise), (x + w, y + h)],
            closed=True, facecolor=side, edgecolor="#334155", linewidth=1.0, zorder=2,
        ))
        ax.add_patch(Polygon(
            [(x, y + h), (x + depth, y + h + rise),
             (x + w + depth, y + h + rise), (x + w, y + h)],
            closed=True, facecolor=top, edgecolor="#334155", linewidth=1.0, zorder=3,
        ))
        ax.add_patch(Polygon(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            closed=True, facecolor=base, edgecolor="#334155", linewidth=1.1, zorder=4,
        ))
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
                fontsize=10, fontweight="semibold", color="#0f172a", zorder=5)
        if sub:
            ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                    fontsize=8, color="#334155", zorder=5)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.8, color="#475569",
            connectionstyle=f"arc3,rad={rad}", zorder=6,
        ))

    iso_box(0.5, 2.2, 1.8, 1.0, "Input", "12 x 16 x 16\nspeed + time",
            "#dbeafe", "#eff6ff", "#bfdbfe")
    iso_box(2.6, 2.2, 1.8, 1.0, "STAG + ConvLSTM", "layer 1",
            "#bfdbfe", "#dbeafe", "#93c5fd")
    iso_box(4.7, 2.2, 1.8, 1.0, "STAG + ConvLSTM", "layer 2",
            "#bfdbfe", "#dbeafe", "#93c5fd")
    iso_box(6.8, 2.2, 1.5, 1.0, "Encoder states", "H1 ... H12",
            "#e2e8f0", "#f1f5f9", "#cbd5e1")
    iso_box(6.8, 3.75, 1.6, 0.9, "Temporal attention", "query-key",
            "#fed7aa", "#ffedd5", "#fdba74")
    iso_box(6.8, 0.5, 1.6, 1.0, "ConvLSTM decoder", "12 recursive steps",
            "#bbf7d0", "#dcfce7", "#86efac")
    iso_box(9.4, 2.2, 1.5, 1.0, "Fusion", "1 x 1 conv",
            "#a7f3d0", "#d1fae5", "#6ee7b7")
    iso_box(11.4, 2.2, 1.5, 1.0, "Output", "12 x 16 x 16",
            "#e2e8f0", "#f1f5f9", "#cbd5e1")

    arrow(2.3, 2.7, 2.6, 2.7)
    arrow(4.4, 2.7, 4.7, 2.7)
    arrow(6.5, 2.7, 6.8, 2.7)
    arrow(7.55, 3.2, 7.55, 3.75)
    arrow(7.55, 2.2, 7.55, 1.5)
    arrow(8.4, 4.2, 9.5, 3.2, -0.12)
    arrow(8.4, 1.0, 9.5, 2.2, 0.12)
    arrow(10.9, 2.7, 11.4, 2.7)

    ax.text(3.5, 4.35, "ENCODER", ha="center", fontsize=9,
            color="#64748b", fontweight="semibold")
    ax.text(7.6, 4.95, "ATTENTION", ha="center", fontsize=9,
            color="#64748b", fontweight="semibold")
    ax.text(6.75, 5.65, "STAGC: ConvLSTM encoder-decoder with spatial gating",
            ha="center", fontsize=14, fontweight="semibold", color="#0f172a")
    ax.text(6.75, 0.12, "Masked Huber loss  ·  5-minute steps  ·  1-hour horizon",
            ha="center", fontsize=9, color="#64748b")

    fig.tight_layout()
    path = out_dir / "architecture.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_horizon(results_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    for ax, name in zip(axes, ("metrla", "pemsbay")):
        for cfg, label in (
            ("linear", "linear"),
            ("cyclic", "cyclic"),
            ("cyclic_residual", "cyclic + residual"),
        ):
            path = results_dir / f"{name}_{cfg}.csv"
            if not path.exists():
                continue
            data = np.genfromtxt(path, delimiter=",", names=True)
            horizon = data["horizon_min"][:-1]
            mae = data["mae"][:-1]
            ax.plot(horizon, mae, marker="o", markersize=3, label=label)
        ax.set_title(name.upper())
        ax.set_xlabel("Prediction horizon (min)")
        ax.set_ylabel("MAE (mph)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "horizon_mae.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_spatial(checkpoints: dict, data_dir: Path, out_dir: Path):
    device = torch.device("cpu")
    for dataset in ("metrla", "pemsbay"):
        if dataset not in checkpoints:
            continue
        ckpt = torch.load(checkpoints[dataset], map_location=device, weights_only=False)
        cfg = ckpt["config"]
        loaders = get_loaders(data_dir, dataset=dataset, batch_size=32,
                              time_mode=cfg["time_encoding"],
                              observed_channel=cfg["observed_channel"])
        model = build_model(cfg["model"], cfg["input_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        scaler = loaders["scaler"]
        strict = torch.from_numpy(loaders["strict"]).view(1, 1, 16, 16)

        err_sum = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)
        with torch.no_grad():
            for x, y, obs in loaders["test"]:
                out = model(x)
                pred = out[0] if isinstance(out, tuple) else out
                p = scaler.inverse(pred[:, 11, 0])
                t = scaler.inverse(y[:, 11, 0])
                mask = strict[:, 0] * obs[:, 11, 0]
                err_sum += ((p - t).abs() * mask).sum(dim=0).numpy()
                cnt += mask.sum(dim=0).numpy()
        mae = np.divide(err_sum, cnt, out=np.full_like(err_sum, np.nan), where=cnt > 0)

        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
        im = ax.imshow(mae, cmap="YlOrRd")
        ax.set_title(f"{dataset.upper()} spatial MAE at 60 min")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE (mph)")
        fig.tight_layout()
        path = out_dir / f"spatial_error_{dataset}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


def plot_example(checkpoints: dict, data_dir: Path, out_dir: Path):
    device = torch.device("cpu")
    for dataset in ("metrla", "pemsbay"):
        if dataset not in checkpoints:
            continue
        ckpt = torch.load(checkpoints[dataset], map_location=device, weights_only=False)
        cfg = ckpt["config"]
        loaders = get_loaders(data_dir, dataset=dataset, batch_size=1,
                              time_mode=cfg["time_encoding"],
                              observed_channel=cfg["observed_channel"])
        model = build_model(cfg["model"], cfg["input_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        scaler = loaders["scaler"]
        x, y, obs = next(iter(loaders["test"]))
        with torch.no_grad():
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
        p = scaler.inverse(pred[0, 11, 0]).numpy()
        t = scaler.inverse(y[0, 11, 0]).numpy()
        mask = (loaders["strict"] * obs[0, 11, 0].numpy()) > 0
        p = np.where(mask, p, np.nan)
        t = np.where(mask, t, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=160)
        for ax, arr, title in zip(axes, (p, t),
                                  ("Prediction, 60 min", "Ground truth, 60 min")):
            im = ax.imshow(arr, cmap="RdYlGn", vmin=0, vmax=85)
            ax.set_title(f"{dataset.upper()} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Speed (mph)")
        path = out_dir / f"prediction_example_{dataset}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="alias for --checkpoint-metrla")
    p.add_argument("--checkpoint-metrla", default=None)
    p.add_argument("--checkpoint-pemsbay", default=None)
    p.add_argument("--data-dir", default=str(BASE / "data"))
    p.add_argument("--results-dir", default=str(BASE / "results" / "reference"))
    p.add_argument("--out-dir", default=str(BASE / "assets"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_architecture(out_dir)
    plot_horizon(Path(args.results_dir), out_dir)

    checkpoints = {}
    metrla = args.checkpoint_metrla or args.checkpoint
    if metrla:
        checkpoints["metrla"] = Path(metrla).resolve()
    if args.checkpoint_pemsbay:
        checkpoints["pemsbay"] = Path(args.checkpoint_pemsbay).resolve()

    if checkpoints:
        data_dir = Path(args.data_dir)
        plot_spatial(checkpoints, data_dir, out_dir)
        plot_example(checkpoints, data_dir, out_dir)
    else:
        print("no checkpoints given; skipped spatial and prediction figures")


if __name__ == "__main__":
    main()

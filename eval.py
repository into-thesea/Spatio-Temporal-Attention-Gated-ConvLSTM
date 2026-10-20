"""Evaluate a trained STAGC checkpoint on observed sensor values.

Example:
  python eval.py --checkpoint models/base_cyclic_s42.pth --split test
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data_core.data import MaskedMetrics, get_loaders
from train import build_model

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    loaders = get_loaders(
        DATA, dataset=cfg.get("dataset", "metrla"), batch_size=args.batch_size,
        time_mode=cfg["time_encoding"],
        observed_channel=cfg["observed_channel"],
    )
    model = build_model(cfg["model"], cfg["input_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = loaders["scaler"]
    mask_hw = torch.from_numpy(loaders["strict"]).view(1, 1, 1, 16, 16).to(device)
    metrics = MaskedMetrics(12)

    with torch.no_grad():
        for x, y, obs in loaders[args.split]:
            x, y, obs = x.to(device), y.to(device), obs.to(device)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            if cfg.get("residual", False):
                pred = pred + x[:, -1, 0:1].unsqueeze(1)
            mask = mask_hw * obs
            metrics.update(scaler.inverse(pred), scaler.inverse(y), mask)

    m = metrics.compute()
    print(f"\n[{Path(args.checkpoint).stem}] split={args.split} n={int(m['count'][0])} per-horizon")
    print(f"{'step':>4} {'min':>4} {'MAE':>8} {'RMSE':>8} {'MAPE%':>7}")
    for i in (2, 5, 11):
        print(f"{i+1:>4} {(i+1)*5:>4} {m['mae'][i]:8.3f} {m['rmse'][i]:8.3f} {m['mape'][i]:7.2f}")
    print(f"mean      {m['mean_mae']:8.3f} {m['mean_rmse']:8.3f} {m['mean_mape']:7.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"eval_{Path(args.checkpoint).stem}_{args.split}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["horizon_min", "mae", "rmse", "mape", "count", "count_pos"])
        for i in range(12):
            w.writerow([(i + 1) * 5, m["mae"][i], m["rmse"][i], m["mape"][i],
                        m["count"][i], metrics.cnt_pos[i]])
        w.writerow(["mean", m["mean_mae"], m["mean_rmse"], m["mean_mape"],
                    m["count"].sum(), metrics.cnt_pos.sum()])
    print(f"saved: {path}")


if __name__ == "__main__":
    main()

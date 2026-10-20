"""Train STAGC on METR-LA / PeMS-BAY grid data.

Example:
  python train.py --dataset metrla --model base --time-encoding cyclic --residual
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_core.data import MaskedMetrics, get_loaders
from models.stagc import ACConvLSTM
from models.stagc_tanh import STAGCTanh

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODEL_DIR = BASE / "models"
LOG_DIR = BASE / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss(pred, target, mask):
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def build_model(name, input_dim):
    if name in ("base", "paper"):
        return ACConvLSTM(input_dim=input_dim, hidden_dim=64, kernel_size=3,
                          num_layers=2, output_len=12)
    if name in ("tanh", "improved"):
        return STAGCTanh(input_dim=input_dim, hidden_dim=64, kernel_size=3,
                         num_layers=2, output_len=12)
    raise ValueError(name)


def run_epoch(model, loader, mask_hw, scaler, optimizer=None, max_batches=None,
              conf_weight=0.0, residual=False, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    metrics = MaskedMetrics(12)
    for i, (x, y, obs) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, obs = x.to(device), y.to(device), obs.to(device)
        mask = mask_hw * obs
        last = x[:, -1, 0:1].unsqueeze(1)
        target = y - last if residual else y
        with torch.set_grad_enabled(train):
            out = model(x)
            if isinstance(out, tuple):
                pred, conf = out
            else:
                pred, conf = out, None
            loss = masked_loss(pred, target, mask)
            pred_abs = pred + last if residual else pred
            if train and conf is not None and conf_weight > 0:
                loss = loss + conf_weight * masked_loss(conf, (pred_abs.detach() - y).abs(), mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += loss.item()
        n += 1
        metrics.update(scaler.inverse(pred_abs.detach()), scaler.inverse(y), mask)
    return total / max(n, 1), metrics.compute()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["metrla", "pemsbay"], default="metrla")
    p.add_argument("--model", choices=["base", "tanh"], default="base")
    p.add_argument("--time-encoding", choices=["linear", "cyclic"], default="linear")
    p.add_argument("--observed-channel", action="store_true")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--conf-weight", type=float, default=0.0)
    p.add_argument("--residual", action="store_true")
    p.add_argument("--resume", action="store_true", help="兼容旧参数；默认已自动续训")
    p.add_argument("--fresh", action="store_true", help="忽略已有 last checkpoint，从头训练")
    p.add_argument("--patience", type=int, default=20)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    loaders = get_loaders(
        DATA,
        dataset=args.dataset,
        batch_size=args.batch_size,
        time_mode=args.time_encoding,
        observed_channel=args.observed_channel,
        stride=args.stride,
    )
    mask_hw = torch.from_numpy(loaders["strict"]).view(1, 1, 1, 16, 16).to(device)
    scaler = loaders["scaler"]
    model = build_model(args.model, loaders["input_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    prefix = "" if args.dataset == "metrla" else f"{args.dataset}_"
    tag = (f"{prefix}{args.model}_{args.time_encoding}"
           f"{'_obs' if args.observed_channel else ''}"
           f"{'_res' if args.residual else ''}_s{args.seed}")
    ckpt_path = MODEL_DIR / f"{tag}.pth"
    last_path = MODEL_DIR / f"{tag}_last.pth"
    log_path = LOG_DIR / f"{tag}_log.csv"
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "input_dim": loaders["input_dim"],
        "time_encoding": args.time_encoding,
        "observed_channel": args.observed_channel,
        "residual": args.residual,
        "seed": args.seed,
    }

    def save_last(ep, done=False):
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": ep,
            "best": best,
            "bad": bad,
            "done": done,
            "config": config,
            "scaler": scaler.state(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, last_path)

    start_epoch = 0
    best = float("inf")
    bad = 0
    if not args.fresh and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") != config:
            raise ValueError(f"resume config mismatch:\n{state.get('config')}\nvs\n{config}")
        if state.get("done", False) and args.epochs <= state["epoch"] + 1:
            print(f"[{tag}] 已完成，跳过。best={state.get('best'):.3f}")
            return
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best = state["best"]
        bad = state["bad"]
        rng = state.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"[{tag}] resume from epoch {start_epoch}, best={best:.3f}")

    log_mode = "a" if (not args.fresh and log_path.exists()) else "w"
    with open(log_path, log_mode, newline="", encoding="utf-8") as f:
        if log_mode == "w":
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_mean_mae", "val_60min_mae", "lr", "minutes"]
            )

    print(f"[{tag}] train={len(loaders['train'].dataset)} val={len(loaders['val'].dataset)} "
          f"test={len(loaders['test'].dataset)} input_dim={loaders['input_dim']} device={device}")
    for ep in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, loaders["train"], mask_hw, scaler, opt,
                               args.max_batches, args.conf_weight, args.residual, device)
        _, val = run_epoch(model, loaders["val"], mask_hw, scaler, None,
                           args.max_batches, args.conf_weight, args.residual, device)
        vmae, v60 = val["mean_mae"], float(val["mae"][11])
        sched.step(vmae)
        mins = (time.time() - t0) / 60
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.6f}", f"{vmae:.6f}", f"{v60:.6f}",
                                    opt.param_groups[0]["lr"], f"{mins:.2f}"])
        print(f"ep{ep:02d} loss={tr_loss:.5f} val_mean_MAE={vmae:.3f} val60={v60:.3f} "
              f"({mins:.1f}min) lr={opt.param_groups[0]['lr']:.1e}")
        if vmae < best:
            best, bad = vmae, 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "dataset": args.dataset,
                    "model": args.model,
                    "input_dim": loaders["input_dim"],
                    "time_encoding": args.time_encoding,
                    "observed_channel": args.observed_channel,
                    "residual": args.residual,
                    "seed": args.seed,
                },
                "scaler": loaders["scaler"].state(),
                "best_val_mean_mae": best,
            }, ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                save_last(ep, done=True)
                break
        save_last(ep)
    else:
        save_last(max(start_epoch, args.epochs - 1), done=True)
    print(f"done: best val mean MAE={best:.3f}, checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()

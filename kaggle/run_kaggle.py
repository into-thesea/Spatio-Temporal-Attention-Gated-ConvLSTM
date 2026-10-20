"""Run training stages on Kaggle.

Usage:
  python run_kaggle.py --dataset metrla --stage baseline
  python run_kaggle.py --dataset pemsbay --stage baseline
  python run_kaggle.py --dataset metrla --stage seeds
  python run_kaggle.py --dataset metrla --stage tanh
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(*args):
    print("\n>>>", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True)


def tag(dataset, model, encoding, seed, residual):
    prefix = "" if dataset == "metrla" else f"{dataset}_"
    return f"{prefix}{model}_{encoding}{'_res' if residual else ''}_s{seed}"


def train(dataset, model, encoding, seed, residual, epochs, patience):
    args = ["train.py", "--dataset", dataset, "--model", model,
            "--time-encoding", encoding, "--epochs", epochs,
            "--batch-size", 32, "--seed", seed, "--patience", patience]
    if residual:
        args.append("--residual")
    run(*args)


def evaluate(dataset, model, encoding, seed, residual, split="test"):
    run("eval.py", "--checkpoint",
        f"models/{tag(dataset, model, encoding, seed, residual)}.pth",
        "--split", split)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["metrla", "pemsbay"], default="metrla")
    p.add_argument("--stage", choices=["baseline", "seeds", "tanh", "all"],
                   default="baseline")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    if args.stage in ("baseline", "all"):
        for enc, res in (("linear", False), ("cyclic", False), ("cyclic", True)):
            train(args.dataset, "base", enc, 42, res, args.epochs, args.patience)
            evaluate(args.dataset, "base", enc, 42, res)

    if args.stage in ("seeds", "all"):
        for seed in (43, 44):
            train(args.dataset, "base", "cyclic", seed, True, args.epochs, args.patience)
            evaluate(args.dataset, "base", "cyclic", seed, True)

    if args.stage in ("tanh", "all"):
        train(args.dataset, "tanh", "cyclic", 42, True, args.epochs, args.patience)
        evaluate(args.dataset, "tanh", "cyclic", 42, True)


if __name__ == "__main__":
    main()

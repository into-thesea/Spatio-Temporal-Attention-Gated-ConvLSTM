"""Measure model size and CPU latency with PyTorch and ONNX Runtime.

Usage:
  python scripts/benchmark_infer.py --checkpoint models/base_cyclic_s42.pth
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE))

from models.stagc import ACConvLSTM


def bench_torch(model, x, runs=30, warmup=5):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def bench_ort(sess, x, runs=30, warmup=5):
    name = sess.get_inputs()[0].name
    arr = x.numpy()
    for _ in range(warmup):
        sess.run(None, {name: arr})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: arr})
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def summary(ts, batch_size=1):
    return {
        "mean_ms": float(np.mean(ts)),
        "p50_ms": float(np.percentile(ts, 50)),
        "p95_ms": float(np.percentile(ts, 95)),
        "samples_per_s": float(batch_size * 1000.0 / np.mean(ts)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/base_cyclic_s42.pth")
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()

    ckpt_path = BASE / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ACConvLSTM(
        input_dim=cfg["input_dim"], hidden_dim=64, kernel_size=3,
        num_layers=2, output_len=12,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    size_mb = ckpt_path.stat().st_size / 1e6
    out = {
        "checkpoint": str(ckpt_path.relative_to(BASE)),
        "params": int(params),
        "file_mb": round(size_mb, 2),
        "torch_cpu": {},
        "onnx_cpu": {},
    }

    print(f"params={params/1e6:.3f}M file={size_mb:.2f}MB")
    for bs in (1, 8, 32):
        x = torch.randn(bs, 12, cfg["input_dim"], 16, 16)
        ts = bench_torch(model, x, runs=args.runs)
        out["torch_cpu"][f"batch{bs}"] = summary(ts, bs)
        s = out["torch_cpu"][f"batch{bs}"]
        print(f"torch batch={bs:>2}: mean={s['mean_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
              f"{s['samples_per_s']:.1f} samples/s")

    try:
        import onnx
        import onnxruntime as ort

        onnx_path = ckpt_path.with_suffix(".onnx")
        dummy = torch.randn(1, 12, cfg["input_dim"], 16, 16)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["x"], output_names=["pred"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ts = bench_ort(sess, dummy, runs=args.runs)
        out["onnx_cpu"]["batch1"] = summary(ts, 1)
        out["onnx_file_mb"] = round(onnx_path.stat().st_size / 1e6, 2)
        s = out["onnx_cpu"]["batch1"]
        print(f"onnx  batch= 1: mean={s['mean_ms']:.2f}ms p95={s['p95_ms']:.2f}ms "
              f"{s['samples_per_s']:.1f} samples/s, file={out['onnx_file_mb']}MB")
    except Exception as e:
        out["onnx_error"] = repr(e)
        print("onnx failed:", repr(e))

    dst = BASE / "results" / "edge_benchmark.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", dst)


if __name__ == "__main__":
    main()

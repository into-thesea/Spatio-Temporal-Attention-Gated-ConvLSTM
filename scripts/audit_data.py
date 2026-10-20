"""Audit a generated grid dataset.

Usage:
  python scripts/audit_data.py --dataset metrla
  python scripts/audit_data.py --dataset pemsbay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
REPORTS = BASE / "reports"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["metrla", "pemsbay"], default="metrla")
    args = p.parse_args()
    prefix = f"{args.dataset}_grid"

    paths = {
        "speed": DATA / f"{prefix}_speed.npy",
        "observed": DATA / f"{prefix}_observed.npy",
        "strict": DATA / f"{prefix}_strict_mask.npy",
        "dilated": DATA / f"{prefix}_dilated_mask.npy",
        "meta": DATA / f"{prefix}_meta.json",
    }
    missing = [str(v) for v in paths.values() if not v.exists()]
    if missing:
        raise SystemExit("missing files:\n" + "\n".join(missing))

    speed = np.load(paths["speed"]).astype(np.float32)
    observed = np.load(paths["observed"]).astype(np.uint8)
    strict = np.load(paths["strict"]).astype(bool)
    dilated = np.load(paths["dilated"]).astype(bool)
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    speed_in = speed[:, dilated]
    per_frame = observed.sum(axis=(1, 2))
    valid_ratio = observed.sum() / max(strict.sum() * speed.shape[0], 1)

    report = f"""# Grid data audit: {args.dataset}

- source: {meta.get("source", "unknown")}
- shape: {speed.shape}
- dtype: {speed.dtype}
- strict cells: {int(strict.sum())}
- dilated cells: {int(dilated.sum())}
- dilation-only cells: {int(dilated.sum() - strict.sum())}
- observed ratio (strict): {valid_ratio:.4f}
- zero ratio inside dilated: {(speed_in == 0).mean():.4f}
- speed min / max / mean / std (inside dilated): {speed_in.min():.2f} / {speed_in.max():.2f} / {speed_in.mean():.2f} / {speed_in.std():.2f}
- observed count per frame: min {int(per_frame.min())}, mean {per_frame.mean():.2f}, max {int(per_frame.max())}
- frames with no observation: {int((per_frame == 0).sum())}
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"DATA_AUDIT_{args.dataset}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

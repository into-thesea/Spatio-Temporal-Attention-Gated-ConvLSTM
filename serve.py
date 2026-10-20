"""FastAPI + ONNX Runtime inference service.

Run:
  uvicorn serve:app --host 0.0.0.0 --port 8000

POST /predict expects 12 frames of 16x16 speed values (mph) and a start time.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
CKPT = BASE / "models" / "base_cyclic_s42.pth"
ONNX = BASE / "models" / "base_cyclic_s42.onnx"
MASK = BASE / "data" / "metrla_grid_dilated_mask.npy"
H = W = 16
SEQ = 12
OUT = 12


class PredictRequest(BaseModel):
    history: list[list[list[float]]]
    start_time: str
    include_grid: bool = False
    threshold_mph: float = 20.0


class PredictResponse(BaseModel):
    latency_ms: float
    mean_speed_by_horizon: list[float]
    min_speed_by_horizon: list[float]
    congested_cells_60min: list[dict]
    predictions: list | None = None


def load_state():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    mask = np.load(MASK).astype(np.float32)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    return scaler, mask, sess


SCALER, MASK, SESSION = load_state()
INPUT_NAME = SESSION.get_inputs()[0].name


def make_time_features(start_time: str):
    t0 = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    feats = []
    for k in range(SEQ):
        t = t0 + timedelta(minutes=5 * k)
        tod = (t.hour * 60 + t.minute) / 1440.0
        dow = t.weekday() / 7.0
        feats.append([
            np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
            np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow),
        ])
    return np.asarray(feats, dtype=np.float32)  # (12,4)


def preprocess(history, start_time):
    speed = np.asarray(history, dtype=np.float32)
    if speed.shape != (SEQ, H, W):
        raise HTTPException(400, f"history must be {SEQ}x{H}x{W}, got {speed.shape}")
    if not np.isfinite(speed).all():
        raise HTTPException(400, "history contains NaN/Inf")
    # 因果前向填充窗口内缺失值
    for t in range(1, SEQ):
        miss = (speed[t] == 0) & (MASK > 0)
        speed[t][miss] = speed[t - 1][miss]
    vmin, vmax = SCALER["vmin"], SCALER["vmax"]
    speed_norm = (speed - vmin) / (vmax - vmin + 1e-8)
    tf = make_time_features(start_time)  # (12,4)
    x = np.concatenate([
        speed_norm[:, None],
        np.broadcast_to(tf[:, :, None, None], (SEQ, 4, H, W)),
    ], axis=1) * MASK[None, None]
    return x[None].astype(np.float32)


def postprocess(pred_norm, threshold_mph):
    vmin, vmax = SCALER["vmin"], SCALER["vmax"]
    pred = pred_norm[0, :, 0] * (vmax - vmin + 1e-8) + vmin  # (12,16,16)
    pred = np.clip(pred * MASK[None], 0, None)
    valid = MASK > 0
    means = [float(pred[k][valid].mean()) for k in range(OUT)]
    mins = [float(pred[k][valid].min()) for k in range(OUT)]
    last = pred[-1]
    cells = [
        {"row": int(r), "col": int(c), "speed_mph": round(float(last[r, c]), 2)}
        for r, c in np.argwhere(valid & (last < threshold_mph))
    ]
    return pred, means, mins, cells


app = FastAPI(title="STAGC Traffic Speed Forecast", version="1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "base_cyclic_s42",
        "params": 920000,
        "onnx": str(ONNX.relative_to(BASE)),
        "window": f"{SEQ}->{OUT}, 5min",
        "grid": f"{H}x{W}",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    import time

    x = preprocess(req.history, req.start_time)
    t0 = time.perf_counter()
    pred_norm = SESSION.run(None, {INPUT_NAME: x})[0]
    latency = (time.perf_counter() - t0) * 1000
    pred, means, mins, cells = postprocess(pred_norm, req.threshold_mph)
    return PredictResponse(
        latency_ms=round(latency, 2),
        mean_speed_by_horizon=[round(v, 2) for v in means],
        min_speed_by_horizon=[round(v, 2) for v in mins],
        congested_cells_60min=cells,
        predictions=np.round(pred, 2).tolist() if req.include_grid else None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

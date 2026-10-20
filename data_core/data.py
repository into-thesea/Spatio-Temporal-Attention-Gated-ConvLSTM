"""Grid data loading, time-based splits, and masked metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

H = W = 16
STEPS_PER_DAY = 288


# ---------------------------------------------------------------- time features
def time_features(t: int, mode: str = "linear") -> np.ndarray:
    """Return (T, C) time features."""
    idx = np.arange(t, dtype=np.float32)
    tod = (idx % STEPS_PER_DAY) / STEPS_PER_DAY
    dow = ((idx // STEPS_PER_DAY) % 7) / 7.0
    if mode == "linear":
        return np.stack([tod, dow], axis=1).astype(np.float32)
    if mode == "cyclic":
        two_pi = 2.0 * np.pi
        return np.stack(
            [
                np.sin(two_pi * tod),
                np.cos(two_pi * tod),
                np.sin(two_pi * dow),
                np.cos(two_pi * dow),
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(f"unknown time mode: {mode}")


# ---------------------------------------------------------------- split
@dataclass(frozen=True)
class Bounds:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def split_bounds(
    t: int,
    input_len: int = 12,
    output_len: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    gap: int = 24,
) -> Bounds:
    """Chronological 7:1:2 split with a gap between segments."""
    train_end = int(t * train_ratio)
    val_end = int(t * (train_ratio + val_ratio))
    val_start = train_end + gap
    test_start = val_end + gap
    if val_start + input_len + output_len > val_end:
        raise ValueError("val segment too short for given gap")
    if test_start + input_len + output_len > t:
        raise ValueError("test segment too short for given gap")
    return Bounds((0, train_end), (val_start, val_end), (test_start, t))


def make_starts(bound: tuple[int, int], input_len: int, output_len: int) -> np.ndarray:
    start, end = bound
    last = end - input_len - output_len + 1
    if last <= start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, last, dtype=np.int64)


# ---------------------------------------------------------------- scaler
class SpeedScaler:
    """Min-max scaler fitted on observed values in the training split."""

    def __init__(self, vmin: float, vmax: float):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eps = 1e-8

    @classmethod
    def fit(cls, speed: np.ndarray, observed: np.ndarray, train_end: int):
        vals = speed[:train_end][observed[:train_end] > 0]
        if vals.size == 0:
            raise ValueError("no observed speed values in train segment")
        return cls(float(vals.min()), float(vals.max()))

    def transform(self, speed: np.ndarray) -> np.ndarray:
        return (speed - self.vmin) / (self.vmax - self.vmin + self.eps)

    def inverse(self, x: np.ndarray | torch.Tensor):
        return x * (self.vmax - self.vmin + self.eps) + self.vmin

    def state(self) -> dict:
        return {"vmin": self.vmin, "vmax": self.vmax}


# ---------------------------------------------------------------- dataset
class GridDataset(Dataset):
    """Windowed dataset returning input, target, and observed mask."""

    def __init__(
        self,
        speed_norm: np.ndarray,      # (T,H,W)
        observed: np.ndarray,        # (T,H,W) uint8
        time_feat: np.ndarray,       # (T,Ct)
        starts: np.ndarray,
        input_len: int,
        output_len: int,
        mask: np.ndarray,            # (H,W) input mask
        observed_channel: bool = False,
    ):
        self.speed = speed_norm
        self.observed = observed
        self.time_feat = time_feat
        self.starts = starts
        self.input_len = input_len
        self.output_len = output_len
        self.mask = mask.astype(np.float32)
        self.observed_channel = observed_channel

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        L, O = self.input_len, self.output_len

        speed_x = self.speed[s : s + L, None, :, :]                 # (L,1,H,W)
        tf = self.time_feat[s : s + L].T[:, :, None, None]          # (Ct,L,1,1)
        tf = np.broadcast_to(tf, (tf.shape[0], L, H, W))            # (Ct,L,H,W)
        parts = [speed_x, tf.transpose(1, 0, 2, 3)]                 # (L,1+Ct,H,W)
        if self.observed_channel:
            parts.append(self.observed[s : s + L, None, :, :].astype(np.float32))
        x = np.concatenate(parts, axis=1) * self.mask[None, None]

        y = self.speed[s + L : s + L + O, None, :, :]               # (O,1,H,W)
        obs_y = self.observed[s + L : s + L + O, None, :, :]        # (O,1,H,W)
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(obs_y.astype(np.float32)),
        )


def load_arrays(data_dir, dataset="metrla"):
    prefix = f"{dataset}_grid"
    speed = np.load(data_dir / f"{prefix}_speed.npy").astype(np.float32)
    observed = np.load(data_dir / f"{prefix}_observed.npy").astype(np.uint8)
    strict = np.load(data_dir / f"{prefix}_strict_mask.npy").astype(np.float32)
    dilated = np.load(data_dir / f"{prefix}_dilated_mask.npy").astype(np.float32)
    return speed, observed, strict, dilated


def get_loaders(
    data_dir,
    dataset="metrla",
    input_len=12,
    output_len=12,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    gap=24,
    time_mode="linear",
    observed_channel=False,
    stride=1,
):
    speed, observed, strict, dilated = load_arrays(data_dir, dataset)
    bounds = split_bounds(len(speed), input_len, output_len, train_ratio, val_ratio, gap)
    scaler = SpeedScaler.fit(speed, observed, bounds.train[1])
    speed_norm = scaler.transform(speed)
    tf = time_features(len(speed), time_mode)

    starts = {}
    for name, bound in (("train", bounds.train), ("val", bounds.val), ("test", bounds.test)):
        idx = make_starts(bound, input_len, output_len)
        if stride > 1 and name == "train":
            idx = idx[::stride]
        starts[name] = idx

    def ds(name):
        return GridDataset(
            speed_norm, observed, tf, starts[name], input_len, output_len,
            dilated, observed_channel,
        )

    return {
        "train": DataLoader(ds("train"), batch_size=batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(ds("val"), batch_size=batch_size, shuffle=False),
        "test": DataLoader(ds("test"), batch_size=batch_size, shuffle=False),
        "scaler": scaler,
        "strict": strict,
        "dilated": dilated,
        "bounds": bounds,
        "input_dim": 1 + tf.shape[1] + (1 if observed_channel else 0),
        "starts": starts,
    }


# ---------------------------------------------------------------- metrics
class MaskedMetrics:
    """Accumulate masked MAE/RMSE/MAPE by horizon."""

    def __init__(self, horizons: int):
        self.n = horizons
        self.sum_abs = np.zeros(horizons, dtype=np.float64)
        self.sum_sq = np.zeros(horizons, dtype=np.float64)
        self.sum_ape = np.zeros(horizons, dtype=np.float64)
        self.cnt = np.zeros(horizons, dtype=np.float64)
        self.cnt_pos = np.zeros(horizons, dtype=np.float64)

    @torch.no_grad()
    def update(self, pred, true, mask, min_speed=10.0):
        d = (pred - true).abs() * mask
        sq = (pred - true) ** 2 * mask
        pos = (true > min_speed).float() * mask
        ape = (pred - true).abs() / true.clamp(min=1e-6) * pos
        self.sum_abs += d.sum(dim=(0, 2, 3, 4)).cpu().numpy()
        self.sum_sq += sq.sum(dim=(0, 2, 3, 4)).cpu().numpy()
        self.sum_ape += ape.sum(dim=(0, 2, 3, 4)).cpu().numpy()
        self.cnt += mask.sum(dim=(0, 2, 3, 4)).cpu().numpy()
        self.cnt_pos += pos.sum(dim=(0, 2, 3, 4)).cpu().numpy()

    def compute(self):
        mae = self.sum_abs / np.maximum(self.cnt, 1)
        rmse = np.sqrt(self.sum_sq / np.maximum(self.cnt, 1))
        mape = self.sum_ape / np.maximum(self.cnt_pos, 1) * 100.0
        mean_mae = self.sum_abs.sum() / max(self.cnt.sum(), 1)
        mean_rmse = np.sqrt(self.sum_sq.sum() / max(self.cnt.sum(), 1))
        mean_mape = self.sum_ape.sum() / max(self.cnt_pos.sum(), 1) * 100.0
        return {
            "mae": mae, "rmse": rmse, "mape": mape,
            "mean_mae": float(mean_mae),
            "mean_rmse": float(mean_rmse),
            "mean_mape": float(mean_mape),
            "count": self.cnt.copy(),
        }

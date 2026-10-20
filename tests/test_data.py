"""Unit tests for splits, dataset shapes, and metrics.

Run:
  python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import torch

from data_core.data import (
    GridDataset, MaskedMetrics, make_starts, split_bounds, time_features,
)


class TestSplit(unittest.TestCase):
    def test_no_overlap(self):
        b = split_bounds(1000, input_len=12, output_len=12, gap=24)
        tr = make_starts(b.train, 12, 12)
        va = make_starts(b.val, 12, 12)
        te = make_starts(b.test, 12, 12)
        self.assertGreater(len(tr), 0)
        self.assertGreater(len(va), 0)
        self.assertGreater(len(te), 0)
        self.assertLessEqual(tr[-1] + 24, va[0])
        self.assertLessEqual(va[-1] + 24, te[0])
        self.assertLessEqual(te[-1] + 24, 1000)

    def test_gap_required(self):
        with self.assertRaises(ValueError):
            split_bounds(100, input_len=12, output_len=12, gap=24)


class TestDataset(unittest.TestCase):
    def test_shapes(self):
        T = 40
        speed = np.random.rand(T, 16, 16).astype(np.float32)
        obs = np.ones((T, 16, 16), dtype=np.uint8)
        tf = time_features(T, "linear")
        ds = GridDataset(speed, obs, tf, np.array([0]), 12, 12, np.ones((16, 16)))
        x, y, o = ds[0]
        self.assertEqual(tuple(x.shape), (12, 3, 16, 16))
        self.assertEqual(tuple(y.shape), (12, 1, 16, 16))
        self.assertEqual(tuple(o.shape), (12, 1, 16, 16))

    def test_cyclic(self):
        tf = time_features(288 * 8, "cyclic")
        self.assertEqual(tf.shape[1], 4)
        self.assertGreaterEqual(tf.min(), -1.0)
        self.assertLessEqual(tf.max(), 1.0)


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        m = MaskedMetrics(2)
        pred = torch.tensor([[[[[1.0]]], [[[2.0]]]]])   # (1,2,1,1,1)
        true = torch.zeros_like(pred)
        mask = torch.ones_like(pred)
        m.update(pred, true, mask, min_speed=10.0)
        out = m.compute()
        np.testing.assert_allclose(out["mae"], [1.0, 2.0])
        self.assertAlmostEqual(out["mean_mae"], 1.5)
        self.assertAlmostEqual(out["mean_rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(out["mape"].sum(), 0.0)

    def test_mask_excludes(self):
        m = MaskedMetrics(1)
        pred = torch.ones((1, 1, 1, 1, 1))
        true = torch.zeros_like(pred)
        mask = torch.zeros_like(pred)
        m.update(pred, true, mask)
        out = m.compute()
        self.assertEqual(out["count"][0], 0)
        self.assertEqual(out["mean_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

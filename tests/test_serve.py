"""Unit tests for the inference service.

Run:
  python -m unittest tests.test_serve -v
"""
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

WEIGHTS = Path("models/base_cyclic_s42.onnx")
DATA = Path("data/metrla_grid_speed.npy")
READY = WEIGHTS.exists() and DATA.exists()

if READY:
    from serve import app
else:
    app = None


@unittest.skipUnless(READY, "weights or sample data not available")
class TestServe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        speed = np.load(DATA)[:12].tolist()
        cls.payload = {
            "history": speed,
            "start_time": "2012-03-01T00:00:00",
            "include_grid": False,
            "threshold_mph": 20.0,
        }

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_predict(self):
        r = self.client.post("/predict", json=self.payload)
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(len(body["mean_speed_by_horizon"]), 12)
        self.assertEqual(len(body["min_speed_by_horizon"]), 12)
        self.assertGreater(body["latency_ms"], 0)
        self.assertIsNone(body["predictions"])

    def test_invalid_shape(self):
        bad = dict(self.payload, history=[[[1.0] * 16] * 16] * 11)
        r = self.client.post("/predict", json=bad)
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()

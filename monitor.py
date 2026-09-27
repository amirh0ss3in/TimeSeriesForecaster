"""Writes training snapshots to runs/<name>.json; monitor.html polls and plots them."""
import json
import os
import time

import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _clean(a):
    """Array -> JSON-safe list (rounded, NaN/inf -> null)."""
    a = np.asarray(a, dtype=float).ravel()
    return [round(float(v), 5) if np.isfinite(v) else None for v in a]


class Monitor:
    def __init__(self, name, x_train, y_train, x_test, y_test, config=None):
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.path = os.path.join(RUNS_DIR, f"{name}.json")
        self.y_test = np.asarray(y_test, dtype=float)
        self.data = {
            "name": name,
            "config": config or {},
            "done": False,
            "x_train": _clean(x_train), "y_train": _clean(y_train),
            "x_test": _clean(x_test), "y_test": _clean(y_test),
            "snapshots": [],
        }
        self._write()
        print(f"Monitor: http://localhost:8000/monitor.html?run={name}")

    def log(self, step, loss, forecast, band=None, samples=None):
        """Record one snapshot. Optional: band = (lower, upper) per point, samples = a few sampled paths.
        The test RMSE is shown in the monitor only and is never used for training."""
        f = np.asarray(forecast, dtype=float)
        rmse = float(np.sqrt(np.mean((f - self.y_test) ** 2)))
        snap = {
            "step": int(step),
            "loss": float(loss) if np.isfinite(loss) else None,
            "rmse": rmse if np.isfinite(rmse) else None,
            "forecast": _clean(f),
        }
        if band is not None:
            snap["lo"], snap["hi"] = _clean(band[0]), _clean(band[1])
        if samples is not None:
            snap["samples"] = [_clean(p) for p in samples]
        self.data["snapshots"].append(snap)
        self._write()
        return rmse

    def finish(self):
        self.data["done"] = True
        self._write()

    def _write(self):
        # Write to a temp file, then swap it in, so the page never reads a half-written file.
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.data, fh)
        for _ in range(50):  # on Windows the swap fails while the server is reading the file
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.02)

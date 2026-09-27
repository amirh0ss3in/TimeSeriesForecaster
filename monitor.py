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


def crps(paths, y):
    """CRPS per point from sampled paths (n_paths, T) against truth y (T,). Lower is better, same units as y.
    CRPS = E|X - y| - 0.5 E|X - X'|: rewards paths close to the truth, penalizes spread that doesn't help."""
    x = np.sort(np.asarray(paths, dtype=float), 0)
    n = len(x)
    near = np.abs(x - y).mean(0)
    spread = (2 * np.arange(1, n + 1) - n - 1) @ x * 2 / n ** 2   # E|X - X'| from sorted samples
    return near - 0.5 * spread


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

    def log(self, step, loss, forecast, band=None, samples=None, paths=None):
        """Record one snapshot. Optional: band = (lower, upper) per point, samples = a few paths to draw,
        paths = all sampled paths (then the test score is CRPS, otherwise RMSE of `forecast`).
        The test score is shown in the monitor only and is never used for training."""
        f = np.asarray(forecast, dtype=float)
        if paths is not None:
            score = float(crps(paths, self.y_test).mean())
            self.data["score_name"] = "test CRPS"
        else:
            score = float(np.sqrt(np.mean((f - self.y_test) ** 2)))
        snap = {
            "step": int(step),
            "loss": float(loss) if np.isfinite(loss) else None,
            "score": score if np.isfinite(score) else None,
            "forecast": _clean(f),
        }
        if band is not None:
            snap["lo"], snap["hi"] = _clean(band[0]), _clean(band[1])
        if samples is not None:
            snap["samples"] = [_clean(p) for p in samples]
        self.data["snapshots"].append(snap)
        self._write()
        return score

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

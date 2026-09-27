"""Writes training snapshots to runs/<name>.json; monitor.html polls and plots them.

A run has one or more series. Each snapshot stores, per series, the median forecast, the 95% band, a few sampled
paths and the test CRPS. Test scores are shown in the monitor only and are never used for training."""
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
    def __init__(self, name, series, config=None):
        """series: {name: dict(x_train, y_train, x_test, y_test, x_unit="pi"|"year", role="", reference=None)}
        role: shown in the panel title (e.g. "zero-shot"); reference: another model's CRPS to compare against."""
        os.makedirs(RUNS_DIR, exist_ok=True)
        self.path = os.path.join(RUNS_DIR, f"{name}.json")
        self.y_test = {k: np.asarray(s["y_test"], dtype=float) for k, s in series.items()}
        self.data = {"name": name, "config": config or {}, "done": False, "score_name": "test CRPS",
                     "series": {}, "snapshots": []}
        for k, s in series.items():
            y_tr = np.asarray(s["y_train"], dtype=float)
            # Climatology: every training value as the forecast distribution. Scores are shown relative to it.
            clim = float(crps(np.tile(y_tr[:, None], (1, len(self.y_test[k]))), self.y_test[k]).mean())
            self.data["series"][k] = {
                "x_train": _clean(s["x_train"]), "y_train": _clean(y_tr),
                "x_test": _clean(s["x_test"]), "y_test": _clean(self.y_test[k]),
                "x_unit": s.get("x_unit", "pi"), "role": s.get("role", ""),
                "reference": s.get("reference"), "clim": clim,
            }
        self._write()
        print(f"Monitor: http://localhost:8000/monitor.html?run={name}")

    def log(self, step, loss, paths):
        """paths: {series name: (n_paths, T) sampled forecast}. Returns {series name: test CRPS}."""
        snap, scores = {"step": int(step), "loss": float(loss) if np.isfinite(loss) else None, "series": {}}, {}
        for k, p in paths.items():
            p = np.asarray(p, dtype=float)
            y = self.y_test[k]
            lo, hi = np.percentile(p, [2.5, 97.5], 0)
            score = float(crps(p, y).mean())
            scores[k] = score
            snap["series"][k] = {
                "score": score if np.isfinite(score) else None,
                "cover": float(((y >= lo) & (y <= hi)).mean()),
                "forecast": _clean(np.median(p, 0)), "lo": _clean(lo), "hi": _clean(hi),
                "samples": [_clean(r) for r in p[np.linspace(0, len(p) - 1, 3).astype(int)]],  # spread across models
            }
        self.data["snapshots"].append(snap)
        self._write()
        return scores

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

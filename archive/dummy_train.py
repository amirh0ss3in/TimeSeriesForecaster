"""Fake training loop to test the monitor: loss decays, the 'forecast' paths slowly converge to the true curve."""
import time

import numpy as np

from monitor import Monitor

n, m, pts_per_pi = 4, 2, 50
x_train = np.arange(0, n * np.pi, np.pi / pts_per_pi)
x_test = np.arange(n * np.pi, (n + m) * np.pi + 1e-9, np.pi / pts_per_pi)
y_train, y_test = np.sin(x_train), np.sin(x_test)

mon = Monitor("dummy", {"sin": dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)},
              config={"n": n, "m": m, "note": "dummy loop"})
rng = np.random.default_rng(0)
steps, every = 1000, 20
for step in range(steps + 1):
    time.sleep(0.03)
    p = step / steps
    loss = np.exp(-5 * p) + 1e-3 * rng.random()
    if step % every == 0:
        # Fake sampled paths whose phase drift and spread shrink as training goes on.
        paths = np.sin(x_test + 1.5 * (1 - p) * (x_test - n * np.pi) / 3) \
                + 0.3 * (1 - p) * rng.standard_normal((20, len(x_test)))
        score = mon.log(step, loss, {"sin": paths})["sin"]
        print(f"step {step:5d}  loss {loss:.4f}  test crps {score:.4f}")
mon.finish()

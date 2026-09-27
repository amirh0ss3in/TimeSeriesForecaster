"""Fake training loop to test the monitor: loss decays, the 'forecast' slowly converges to the true curve."""
import time

import numpy as np

from monitor import Monitor

n, m, pts_per_pi = 4, 2, 50
x_train = np.arange(0, n * np.pi, np.pi / pts_per_pi)
x_test = np.arange(n * np.pi, (n + m) * np.pi + 1e-9, np.pi / pts_per_pi)
y_train, y_test = np.sin(x_train), np.sin(x_test)

mon = Monitor("dummy", x_train, y_train, x_test, y_test, config={"n": n, "m": m, "note": "dummy loop"})
rng = np.random.default_rng(0)
steps, every = 1000, 20
for step in range(steps + 1):
    time.sleep(0.03)
    p = step / steps
    loss = np.exp(-5 * p) + 1e-3 * rng.random()
    if step % every == 0:
        # A fake forecast whose phase drift, damping and noise all shrink as training goes on.
        forecast = (1 - 0.8 * (1 - p)) * np.sin(x_test + 1.5 * (1 - p) * (x_test - n * np.pi) / 3) \
                   + 0.2 * (1 - p) * rng.standard_normal(len(x_test))
        rmse = mon.log(step, loss, forecast)
        print(f"step {step:5d}  loss {loss:.4f}  test rmse {rmse:.4f}")
mon.finish()

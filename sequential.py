"""Sequential fine-tuning: one ensemble is trained on one series, then the same weights are trained further on the
next series, and so on. Before any training and after every stage it forecasts the test range of all series, which
shows transfer (series not trained on yet) and forgetting (series trained on earlier stages).

Usage: python sequential.py [--steps=800] [--horizon=200] [--replay=0]
  --replay=P: from the second stage on, P% of each batch are windows from the series trained on earlier (equal share
  each), to reduce forgetting; the rest come from the current series.

No leakage: each stage trains only on that series' training part, scaled with its own training statistics. Every
forecast starts from the series' last `ctx` training values and then feeds on its own samples.
"""
import sys
import time

import numpy as np
import torch

from data import all_series
from model import Ensemble, nll, sample_paths
from monitor import Monitor

# ---- config ----
flag = lambda key, default: int(next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{key}=")), default))
steps_per_stage, horizon, replay = flag("steps", 800), flag("horizon", 200), flag("replay", 0)
ORDER = ["sin", "trend", "amp (noise 0.3)", "sunspots", "co2"]
ctx, d_model, heads, layers = 128, 64, 4, 2
lr, n_models, batch, n_paths = 1e-3, 5, 48, 100
seed = 0
name = f"sequential_s{steps_per_stage}_h{horizon}" + (f"_replay{replay}" if replay else "")

torch.manual_seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data ----
SERIES = all_series(horizon)
stats = {k: (s[1].mean(), s[1].std()) for k, s in SERIES.items()}    # training-part statistics only
norm = lambda k, y: (y - stats[k][0]) / stats[k][1]
names = list(SERIES)
hist = torch.tensor(np.stack([norm(k, SERIES[k][1][-ctx:]) for k in names]), dtype=torch.float32, device=dev)
span = torch.arange(ctx + 1, device=dev)

# ---- model ----
model = Ensemble(n_models, ctx, d_model, heads, layers).to(dev)


def forecast():
    """Sampled paths for every series in one batched rollout, in each series' own units."""
    out = sample_paths(model, hist, horizon, n_paths).cpu().numpy()
    return {k: out[i, :, :len(SERIES[k][3])] * stats[k][1] + stats[k][0] for i, k in enumerate(names)}


def windows(k):
    """All training windows of series k, as (inputs, targets) of shape (n, ctx)."""
    z = torch.tensor(norm(k, SERIES[k][1]), dtype=torch.float32, device=dev)
    w = z.unfold(0, ctx + 1, 1)
    return w[:, :-1], w[:, 1:]


@torch.no_grad()
def one_step_mse(X, Y):
    return ((model(X[:256].expand(n_models, -1, -1))[0] - Y[:256]) ** 2).mean().item()


# ---- stages ----
cfg = dict(order=ORDER, steps_per_stage=steps_per_stage, horizon=horizon, replay=replay, ctx=ctx, d_model=d_model, heads=heads,
           layers=layers, lr=lr, n_models=n_models, batch=batch, n_paths=n_paths)
print(cfg)
mon = Monitor(name, {k: dict(x_train=s[0], y_train=s[1], x_test=s[2], y_test=s[3], x_unit=s[4]) for k, s in
                     SERIES.items()}, config=cfg)
t0, step = time.time(), 0
X, Y = windows(ORDER[0])
scores = mon.log(step, one_step_mse(X, Y), forecast(), label="untrained")
table = {"untrained": scores}
for stage, k in enumerate(ORDER):
    X, Y = windows(k)
    # Replay pool: windows of all earlier series, each series getting an equal share of the replayed part.
    old = [windows(j) for j in ORDER[:stage]]
    n_old = round(batch * replay / 100) if old else 0
    if n_old:
        RX, RY = torch.cat([w[0] for w in old]), torch.cat([w[1] for w in old])
        p_old = torch.cat([torch.full((len(w[0]),), 1 / len(w[0])) for w in old]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)          # fresh optimizer and schedule per stage
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_stage)
    for _ in range(steps_per_stage):
        idx = torch.randint(0, len(X), (n_models, batch - n_old), device=dev)   # own batch per model
        xb, yb = X[idx], Y[idx]
        if n_old:
            r = torch.multinomial(p_old, n_models * n_old, replacement=True).view(n_models, n_old)
            xb, yb = torch.cat([xb, RX[r]], 1), torch.cat([yb, RY[r]], 1)
        loss = nll(*model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
    scores = mon.log(step, one_step_mse(X, Y), forecast(), label=f"after {k}", trained=k)
    table[f"after {k}"] = scores
    print(f"after {k:16s} " + "  ".join(f"{s} {v:.2f}" for s, v in scores.items()) + f"  ({time.time() - t0:.0f}s)")
mon.finish()

# ---- final report: skill = CRPS / climatology CRPS (below 1 beats climatology) ----
clim = {k: mon.data["series"][k]["clim"] for k in names}
print("\nskill (CRPS / climatology); [x] = just trained on, above it = not trained on yet, below = trained earlier")
print(f"{'':22s}" + "".join(f"{k[:14]:>16s}" for k in names))
for row, sc in table.items():
    trained = row.removeprefix("after ")
    print(f"{row:22s}" + "".join(f"{(f'[{sc[k] / clim[k]:.2f}]' if k == trained else f'{sc[k] / clim[k]:.2f}'):>16s}"
                                   for k in names))

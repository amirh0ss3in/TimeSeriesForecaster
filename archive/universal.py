"""One model for many series: a single ensemble is trained on several series at once, then forecasts each of them,
including one it never saw at all (zero-shot).

Usage: python universal.py [--synth=N] [--d=64] [--layers=2] [--batch=48] [--steps=3000] [--log-every=500]
  --synth adds N random synthetic series to the training pool; the others set model width, depth, windows per
  model per step, training steps and how often a monitor snapshot (a full forecast) is taken.

No leakage: each series is scaled with its own training-part statistics. The held-out series contributes nothing to
training, not even its past; at forecast time the model only reads its last `ctx` training-part values as context.
Every training group (each series, and the synthetic pool as a whole) gets an equal share of each batch.
"""
import sys
import time

import numpy as np
import torch

from data import all_series
from model import Ensemble, nll, sample_paths
from monitor import Monitor, crps

# ---- config ----
flag = lambda key, default: int(next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{key}=")), default))
synth = flag("synth", 0)
d_model, layers, batch, steps = flag("d", 64), flag("layers", 2), flag("batch", 48), flag("steps", 3000)
log_every = flag("log-every", 500)       # each snapshot is a full rollout (801 steps for the synthetic series)
pool_share = flag("pool-share", -1)      # % of each batch for the synthetic pool; -1 = equal share per group
horizon_cap = flag("horizon", 0)         # forecast/score only the first N test points (quick checks); 0 = all
scale_prefix = flag("scale-prefix", 0)   # causal scale normalization from the window's first N values; 0 = off
name = ("universal" + (f"_synth{synth}" if synth else "")
        + (f"_d{d_model}_l{layers}_b{batch}" if (d_model, layers, batch) != (64, 2, 48) else "")
        + (f"_pool{pool_share}" if pool_share >= 0 else "") + (f"_h{horizon_cap}" if horizon_cap else "")
        + (f"_s{steps}" if steps != 3000 else "") + (f"_scale{scale_prefix}" if scale_prefix else ""))
ctx = 128                     # one shared context length for all series
heads, lr = 4, 1e-3
n_models, n_paths = 5, 100
seed = 0

torch.manual_seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ---- series: (x_train, y_train, x_test, y_test, x_unit) ----
SERIES = all_series(horizon_cap)
HELD_OUT = {"co2"}            # never used in training: zero-shot
REFERENCE = {"amp (noise 0.3)": 6.32, "sunspots": 21.45, "co2": 6.89}   # test CRPS of the single-series models
if horizon_cap:               # shorter test ranges: the full-range references no longer compare
    REFERENCE = {}


def random_series(rng, T=1024):
    """Synthetic pool: 1-3 sines with periods of 6-200 steps, sometimes growing amplitude, sometimes a trend, noise."""
    t = np.arange(T)
    y = sum(rng.uniform(0.2, 1) * np.sin(2 * np.pi * t / np.exp(rng.uniform(np.log(6), np.log(200)))
                                         + rng.uniform(0, 2 * np.pi)) for _ in range(rng.integers(1, 4)))
    if rng.random() < 0.5:
        y = y * (1 + rng.uniform(0, 3) * t / T)
    if rng.random() < 0.5:
        y = y + rng.normal(0, 0.02) * t
    return y + rng.uniform(0, 0.3) * rng.standard_normal(T)


# ---- training pool ----
stats = {k: (s[1].mean(), s[1].std()) for k, s in SERIES.items()}    # training-part statistics only
norm = lambda k, y: (y - stats[k][0]) / stats[k][1]
groups = {k: [norm(k, s[1])] for k, s in SERIES.items() if k not in HELD_OUT}
if synth:
    rng = np.random.default_rng(seed + 1)
    groups["synthetic pool"] = [(y - y.mean()) / y.std() for y in (random_series(rng) for _ in range(synth))]

# Share of each batch per group: equal, or the pool gets pool_share % and the real series split the rest.
n_real = len(groups) - ("synthetic pool" in groups)
share = {g: (pool_share / 100 if g == "synthetic pool" else (1 - pool_share / 100) / n_real)
         if pool_share >= 0 and "synthetic pool" in groups else 1 / len(groups) for g in groups}

# All series in one buffer; a window is ctx+1 consecutive values of one series, drawn with its group's share.
chunks, starts, probs, offset = [], [], [], 0
for g, arrays in groups.items():
    n_win = sum(len(a) - ctx for a in arrays)
    for a in arrays:
        chunks.append(a)
        starts.append(offset + np.arange(len(a) - ctx))
        probs.append(np.full(len(a) - ctx, share[g] / n_win))
        offset += len(a)
Z = torch.tensor(np.concatenate(chunks), dtype=torch.float32, device=dev)
starts = torch.tensor(np.concatenate(starts), device=dev)
p_win = torch.tensor(np.concatenate(probs), dtype=torch.float32, device=dev)
span = torch.arange(ctx + 1, device=dev)


def windows(i):
    w = Z[starts[i][..., None] + span]
    return w[..., :-1], w[..., 1:]


# ---- model ----
model = Ensemble(n_models, ctx, d_model, heads, layers, scale_prefix=scale_prefix).to(dev)
fv = model.first_valid        # positions before this would see offset/scale computed from their own targets
opt = torch.optim.AdamW(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

names = list(SERIES)
hist = torch.tensor(np.stack([norm(k, SERIES[k][1][-ctx:]) for k in names]), dtype=torch.float32, device=dev)
horizon = max(len(SERIES[k][3]) for k in names)


def forecast():
    """Sampled paths for every series in one batched rollout, in each series' own units."""
    out = sample_paths(model, hist, horizon, n_paths).cpu().numpy()
    return {k: out[i, :, :len(SERIES[k][3])] * stats[k][1] + stats[k][0] for i, k in enumerate(names)}


# ---- train ----
cfg = dict(train_groups=list(groups), batch_share={g: round(v, 3) for g, v in share.items()},
           held_out=sorted(HELD_OUT), synth=synth, horizon=horizon, scale_prefix=scale_prefix, ctx=ctx, d_model=d_model, heads=heads,
           layers=layers, steps=steps, lr=lr, n_models=n_models, batch=batch, n_paths=n_paths,
           train_windows=len(starts), params_per_model=sum(p.numel() for p in model.parameters()) // n_models)
print(cfg)
mon = Monitor(name, {k: dict(x_train=s[0], y_train=s[1], x_test=s[2], y_test=s[3], x_unit=s[4],
                             role="zero-shot (never seen)" if k in HELD_OUT else "seen in training (past only)",
                             reference=REFERENCE.get(k)) for k, s in SERIES.items()}, config=cfg)
mse_idx = torch.multinomial(p_win, 256, replacement=True)[None]   # fixed windows for the monitor's one-step MSE
t0 = time.time()
for step in range(steps + 1):
    idx = torch.multinomial(p_win, n_models * batch, replacement=True).view(n_models, batch)  # own batch per model
    x, y = windows(idx)
    mean, log_sigma = model(x)
    loss = nll(mean[..., fv:], log_sigma[..., fv:], y[..., fv:])
    if step % log_every == 0 or step == steps:
        paths = forecast()
        with torch.no_grad():
            xm, ym = windows(mse_idx.expand(n_models, -1))
            mse = ((model(xm)[0][..., fv:] - ym[..., fv:]) ** 2).mean().item()
        scores = mon.log(step, mse, paths)
        print(f"step {step:5d}  nll {loss.item():+.3f}  one-step mse {mse:.2e}  crps "
              + "  ".join(f"{k} {v:.2f}" for k, v in scores.items()) + f"  ({time.time() - t0:.0f}s)")
    if step == steps:
        break
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
mon.finish()

# ---- final report (same paths as the monitor's last snapshot) ----
print(f"\n{'series':18s} {'role':30s} {'CRPS':>7s} {'climat.':>8s} {'skill':>6s} {'in band':>8s} {'single-series':>14s}")
for k, s in SERIES.items():
    y, p = s[3], paths[k]
    lo, hi = np.percentile(p, [2.5, 97.5], 0)
    c = crps(p, y).mean()
    clim = mon.data["series"][k]["clim"]
    ref = REFERENCE.get(k)
    print(f"{k:18s} {mon.data['series'][k]['role']:30s} {c:7.2f} {clim:8.2f} {c / clim:6.2f} "
          f"{((y >= lo) & (y <= hi)).mean():8.0%} {'' if ref is None else f'{ref:.2f}':>14s}")
print("skill = CRPS / climatology CRPS (below 1 = better than climatology)")

"""Ensemble of decoder-only transformers, next-value prediction, forecast of an unseen range by sampling (like an LLM).

Usage: python train.py <signal> [--raw] [--noise=STD] [--half-life=H]
  --half-life: recency weighting — training windows are drawn with probability 0.5 ** (age / H), where age is how far
  the window's last value lies before the end of the training data (same expected gradient as weighting the loss).
  signal: sin | trend | amp (synthetic) or sunspots | co2 (real, see DATASETS);  --raw turns off per-window
  normalization;  --noise adds Gaussian observation noise to both training and test data of a synthetic signal
  (fixed seed). The forecast is scored against the noisy test values.
  sunspots: SILSO monthly mean total sunspot number (archive/data/sunspots.csv); train before 1990, test from 1990 on.
  co2: NOAA Mauna Loa monthly mean CO2 in ppm (archive/data/co2.csv); train before 2010, test from 2010 on.

Ensemble: `n_models` independent transformers (different random init), stored as one set of stacked weights so all
of them train and forecast together in single batched GPU calls.
Training: one-step (teacher-forced) Gaussian NLL — each model outputs a mean and a σ for the next value.
Forecast: `n_paths` paths split evenly across the models; each step samples next ~ N(mean, σ) from that path's
model and feeds the sample back in. The band is the spread of all paths (2.5%–97.5%), the line is their median.
"""
import math
import sys
import time

import numpy as np
import torch

from model import Ensemble, nll, sample_paths
from monitor import Monitor, crps

# ---- config ----
SIGNALS = {
    "sin": np.sin,
    "trend": lambda x: np.sin(x) + 0.1 * x,
    "amp": lambda x: 0.1 * x * np.sin(x),
}
DATASETS = {  # name: (file, delimiter, split year, context in months, batch) — columns 2, 3 = decimal year, value
    "sunspots": ("archive/data/sunspots.csv", ";", 1990, 132, 48),   # context ~ one 11-year solar cycle
    "co2": ("archive/data/co2.csv", ",", 2010, 36, 144),             # context = three yearly cycles
}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
signal_name = args[0] if args else "sin"
real = signal_name in DATASETS
window_norm = "--raw" not in sys.argv
flag = lambda key: float(next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{key}=")), 0))
noise = flag("noise")          # observation noise std
half_life = flag("half-life")  # recency weighting, in x units (years for real data); 0 = off
name = (signal_name + (f"_noise{noise:g}" if noise else "") + (f"_hl{half_life:g}" if half_life else "")
        + ("" if window_norm else "_raw"))
n, m = 24, 40          # synthetic: train on [0, nπ), test on [nπ, (n+m)π]
pts_per_pi = 20        # synthetic sampling: h = π/20, so one period = 40 points
ctx, batch = 40, 128   # synthetic: context = one period; random training windows per model per step
if real:
    data_file, delim, split_year, ctx, batch = DATASETS[signal_name]   # batch keeps ~same tokens per step
d_model, heads, layers = 64, 4, 2
steps, lr = 3000, 1e-3
log_every = 200 if real else 100   # each snapshot runs a full forecast
n_models = 5          # ensemble size (each model draws its own random batch)
n_paths = 100          # sampled forecast paths (split evenly across models)
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data ----
if real:
    rows = [line for line in open(data_file) if line[:1].isdigit()]    # skip comment and header lines
    raw = np.loadtxt(rows, delimiter=delim, usecols=(2, 3))            # decimal year, value
    tr = raw[:, 0] < split_year
    x_train, y_train, x_test, y_test = raw[tr, 0], raw[tr, 1], raw[~tr, 0], raw[~tr, 1]
    clean_test = None
else:
    h = math.pi / pts_per_pi
    x_train = np.arange(n * pts_per_pi) * h                           # [0, nπ)
    x_test = (n * pts_per_pi + np.arange(m * pts_per_pi + 1)) * h     # [nπ, (n+m)π]
    signal = SIGNALS[signal_name]
    clean_test = signal(x_test)
    rng = np.random.default_rng(seed)
    y_train = signal(x_train) + noise * rng.standard_normal(len(x_train))
    y_test = clean_test + noise * rng.standard_normal(len(x_test))
fmt_x = (lambda v: f"{v:.0f}") if real else (lambda v: f"{v / math.pi:.0f}π")

# Normalization uses training data only.
mu, sd = y_train.mean(), y_train.std()
z_train = torch.tensor((y_train - mu) / sd, dtype=torch.float32, device=dev)

# All windows of ctx+1 consecutive training values: input = first ctx, target = the next value at each position.
win = z_train.unfold(0, ctx + 1, 1)                  # (num_windows, ctx+1)
X, Y = win[:, :-1], win[:, 1:]

# Sampling probability of each window: uniform, or exponentially decaying with the age of its last value.
age = x_train[-1] - x_train[ctx:]                    # window i ends at x_train[i + ctx]
p_win = torch.tensor(0.5 ** (age / half_life) if half_life else np.ones(len(age)), dtype=torch.float32, device=dev)
p_win /= p_win.sum()
eff_windows = 1 / (p_win ** 2).sum().item()          # effective number of windows (= all of them when uniform)


# ---- model ----
model = Ensemble(n_models, ctx, d_model, heads, layers, window_norm).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


def forecast():
    """Sampled paths from the last training window, (n_paths, len(x_test)), in data units."""
    return sample_paths(model, z_train[None, -ctx:], len(x_test), n_paths)[0].cpu().numpy() * sd + mu


# ---- train ----
cfg = dict(signal=signal_name, x_unit="year" if real else "pi", noise=noise, window_norm=window_norm,
           **(dict(split_year=split_year) if real else dict(n=n, m=m, pts_per_pi=pts_per_pi)), ctx=ctx,
           d_model=d_model, heads=heads, layers=layers, steps=steps, lr=lr, n_models=n_models, n_paths=n_paths,
           batch=batch, half_life=half_life, train_windows=len(X), effective_windows=round(eff_windows), params_per_model=sum(p.numel() for p in model.parameters()) // n_models)
print(cfg)
mon = Monitor(name, {signal_name: dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                      x_unit="year" if real else "pi")}, config=cfg)
mse_idx = torch.randperm(len(X), device=dev)[:256]   # fixed windows for the monitor's one-step MSE
t0 = time.time()
for step in range(steps + 1):
    idx = torch.multinomial(p_win, n_models * batch, replacement=True).view(n_models, batch)  # own batch per model
    mean, log_sigma = model(X[idx])
    loss = nll(mean, log_sigma, Y[idx])
    if step % log_every == 0 or step == steps:
        paths = forecast()
        with torch.no_grad():                         # one-step MSE on a fixed subset of windows (cheap, comparable)
            mse = ((model(X[mse_idx].expand(n_models, -1, -1))[0] - Y[mse_idx]) ** 2).mean().item()
        score = mon.log(step, mse, {signal_name: paths})[signal_name]
        print(f"step {step:5d}  nll {loss.item():+.3f}  one-step mse {mse:.2e}  test crps {score:.4f}"
              f"  ({time.time() - t0:.0f}s)")
    if step == steps:
        break
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
mon.finish()

# ---- final report (same paths as the monitor's last snapshot) ----
f = np.median(paths, 0)
lo, hi = np.percentile(paths, [2.5, 97.5], 0)
err = f - y_test
inside = (y_test >= lo) & (y_test <= hi)
cr = crps(paths, y_test)
print(f"\nFinal test CRPS {cr.mean():.4f}   RMSE (median path) {np.sqrt(np.mean(err ** 2)):.4f}"
      f"   inside 95% band: {inside.mean():.0%}")
if noise and clean_test is not None:
    print(f"RMSE of median vs clean curve {np.sqrt(np.mean((f - clean_test) ** 2)):.4f}   (noise std {noise:g})")
for sl in np.array_split(np.arange(len(x_test)), 10):  # 10 segments: does the error grow, and the band with it?
    print(f"  [{fmt_x(x_test[sl[0]])}, {fmt_x(x_test[sl[-1]])}]  CRPS {cr[sl].mean():.4f}"
          f"   RMSE {np.sqrt(np.mean(err[sl] ** 2)):.4f}"
          f"   mean band width {(hi - lo)[sl].mean():.3f}   inside band {inside[sl].mean():.0%}")

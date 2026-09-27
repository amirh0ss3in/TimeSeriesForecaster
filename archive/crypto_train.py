"""Crypto, step 1: can the model learn one coin? Hourly log returns, one ensemble, 24-hour sampled forecasts from every
day's midnight, scored with CRPS against a recent-volatility baseline. Watch it in crypto.html?run=crypto_<COIN>.

Usage: python crypto_train.py [--coin=BTC] [--steps=3000]

Splits by date: train up to 2025-01-01, validation 2025 (tracked during training), test Jan-Aug 2026 (once, at the end).
No leakage: scaling uses training returns only; each forecast reads only the 168 hours up to its origin; the
volatility baseline is a running average of past squared returns whose decay is chosen on training data.
"""
import csv
import gzip
import math
import os
import sys
import time

import numpy as np
import torch

from model import Ensemble, nll, sample_paths
from monitor import RUNS_DIR, crps, write_json

# ---- config ----
arg = lambda key, default: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{key}=")), default)
coin, steps = arg("coin", "BTC"), int(arg("steps", 3000))
ctx, H = 168, 24              # read one week of hours, forecast the next 24
d_model, heads, layers, n_models, batch, lr = 64, 4, 2, 5, 48, 1e-3
n_paths, eval_every, chunk = 100, 300, 32
VAL, TEST = np.datetime64("2025-01-01T00:00"), np.datetime64("2026-01-01T00:00")
seed = 0
name = f"crypto_{coin}"

torch.manual_seed(seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
hour = np.timedelta64(1, "h")

# ---- data: hourly closes on a full hourly grid -> log returns in % ----
rows = list(csv.reader(gzip.open(f"data/crypto/{coin}.csv.gz", "rt")))[1:]
t = np.array([np.datetime64(r[0].replace(" ", "T")) for r in rows])
close = np.array([float(r[4]) for r in rows])
grid = np.arange(t[0], t[-1] + hour, hour)
price = close[np.searchsorted(t, grid, side="right") - 1]   # missing hours keep the previous close (a 0% move)
ret = 100 * np.diff(np.log(price))                            # ret[i]: move from grid[i] to grid[i+1]
rt = grid[1:]                                                 # ...known at the end of that hour
n_train = int((rt <= VAL).sum())                              # training = everything known by 2025-01-01 00:00
mu, sd = ret[:n_train].mean(), ret[:n_train].std()
z = torch.tensor((ret - mu) / sd, dtype=torch.float32, device=dev)
win = z[:n_train].unfold(0, ctx + 1, 1)
X, Y = win[:, :-1], win[:, 1:]


def origins(start, end):
    """Indices o of returns ending at midnight, from `start` on, whose next H hours all end by `end`."""
    midnight = rt.astype("datetime64[h]").astype(np.int64) % 24 == 0
    return np.where(midnight & (rt >= start) & (rt + H * hour <= end))[0]


val_o, test_o = origins(VAL, TEST), origins(TEST, rt[-1])
track_o = val_o[::3]          # every 3rd validation day is tracked during training (speed); all days at the end


# ---- baselines ----
_erf = np.vectorize(math.erf)


def gauss_crps(y, s):
    """CRPS of N(0, s^2) against y (closed form)."""
    w = y / s
    return s * (w * _erf(w / math.sqrt(2)) + 2 * np.exp(-w * w / 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))


def ewma_var(lam):
    """v[i] = variance estimate using returns up to and including i (past only)."""
    v, cur = np.empty(len(ret)), ret[:n_train].var()
    for i, r in enumerate(ret):
        cur = lam * cur + (1 - lam) * r * r
        v[i] = cur
    return v


# Decay chosen on training data only: best next-hour CRPS over the training period.
lam = min([0.7, 0.8, 0.85, 0.9, 0.94, 0.97, 0.98, 0.99, 0.995],
          key=lambda l: gauss_crps(ret[ctx:n_train], np.sqrt(ewma_var(l)[ctx - 1:n_train - 1])).mean())
v_ewma = ewma_var(lam)


def baseline_metrics(o_idx, sigma):
    """sigma: per-origin hourly σ. Forecast: N(0, σ²) for every hour, N(0, 24σ²) for the 24-hour move."""
    y = np.stack([ret[o + 1:o + 1 + H] for o in o_idx])
    s = np.asarray(sigma)[:, None]
    ycum, scum = y.sum(1), s[:, 0] * math.sqrt(H)
    return {"crps_h": gauss_crps(y, s).mean(0).tolist(), "crps_24": float(gauss_crps(ycum, scum).mean()),
            "cover50": float((np.abs(ycum) <= 0.6745 * scum).mean()),
            "cover95": float((np.abs(ycum) <= 1.96 * scum).mean())}


def baselines(o_idx):
    return {"volatility": baseline_metrics(o_idx, np.sqrt(v_ewma[o_idx])),
            "climatology": baseline_metrics(o_idx, np.full(len(o_idx), ret[:n_train].std()))}


# ---- model ----
model = Ensemble(n_models, ctx, d_model, heads, layers, window_norm=False).to(dev)   # returns have no level to remove
opt = torch.optim.AdamW(model.parameters(), lr=lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


def sample(o_idx):
    """(origins, n_paths, H) sampled hourly returns in %, each from the ctx hours up to its origin."""
    out = []
    for i in range(0, len(o_idx), chunk):
        hist = torch.stack([z[o - ctx + 1:o + 1] for o in o_idx[i:i + chunk]])
        out.append(sample_paths(model, hist, H, n_paths).cpu().numpy() * sd + mu)
    return np.concatenate(out)


def metrics(o_idx, paths):
    y = np.stack([ret[o + 1:o + 1 + H] for o in o_idx])
    O = len(o_idx)
    c_h = crps(paths.transpose(1, 0, 2).reshape(n_paths, -1), y.reshape(-1)).reshape(O, H).mean(0)
    cum, ycum = paths.sum(-1), y.sum(-1)                       # 24-hour move, sampled vs actual
    q = np.percentile(cum, [2.5, 25, 50, 75, 97.5], 1)
    return {"crps_h": c_h.tolist(), "crps_24": float(crps(cum.T, ycum).mean()),
            "cover50": float(((ycum >= q[1]) & (ycum <= q[3])).mean()),
            "cover95": float(((ycum >= q[0]) & (ycum <= q[4])).mean()),
            "direction": float((np.sign(q[2]) == np.sign(ycum)).mean())}


# Example days shown as price charts: 8 validation origins, spread over the year.
show = track_o[np.linspace(0, len(track_o) - 1, 8).astype(int)]


def fans(paths_show):
    """Price fans for the example days: quantiles and 3 sampled paths, from the origin price (h = 0..24)."""
    out = []
    for o, p in zip(show, paths_show):
        prices = price[o + 1] * np.exp(np.concatenate([np.zeros((n_paths, 1)), p.cumsum(1) / 100], 1))
        q = np.percentile(prices, [2.5, 25, 50, 75, 97.5], 0)
        out.append({"q": [np.round(r, 2).tolist() for r in q],
                    "samples": [np.round(prices[j], 2).tolist() for j in (0, n_paths // 2, n_paths - 1)]})
    return out


# ---- run ----
cfg = dict(coin=coin, ctx=ctx, horizon=H, d_model=d_model, heads=heads, layers=layers, n_models=n_models,
           batch=batch, lr=lr, steps=steps, n_paths=n_paths, train_until=str(VAL), val_until=str(TEST),
           train_hours=n_train, val_days=len(val_o), tracked_val_days=len(track_o), test_days=len(test_o), ewma_decay=lam,
           params_per_model=sum(p.numel() for p in model.parameters()) // n_models)
print(cfg)
data = {"name": name, "config": cfg, "done": False, "snapshots": [], "test": None,
        "baselines": {"val": baselines(track_o)}, "val_full": None,
        "show": [{"origin": str(rt[o]).replace("T", " "), "hist": np.round(price[o + 1 - 72:o + 2], 2).tolist(),
                  "truth": np.round(price[o + 1:o + 2 + H], 2).tolist()} for o in show]}
path = os.path.join(RUNS_DIR, f"{name}.json")
os.makedirs(RUNS_DIR, exist_ok=True)
write_json(path, data)
print(f"Monitor: http://localhost:8000/crypto.html?run={name}")

t0, losses = time.time(), []
for step in range(steps + 1):
    if step % eval_every == 0 or step == steps:
        paths = sample(track_o)
        m = metrics(track_o, paths)
        data["snapshots"].append({"step": step, "train_nll": float(np.mean(losses)) if losses else None, "val": m,
                                  "fans": fans(paths[np.searchsorted(track_o, show)])})
        write_json(path, data)
        b = data["baselines"]["val"]["volatility"]
        print(f"step {step:5d}  train nll {np.mean(losses) if losses else float('nan'):+.3f}  val CRPS 1h "
              f"{m['crps_h'][0]:.3f} (vol {b['crps_h'][0]:.3f})  24h {m['crps_24']:.3f} (vol {b['crps_24']:.3f})"
              f"  cover95 {m['cover95']:.0%}  ({time.time() - t0:.0f}s)")
        losses = []
    if step == steps:
        break
    idx = torch.randint(0, len(X), (n_models, batch), device=dev)   # own random batch per model
    loss = nll(*model(X[idx]), Y[idx])
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.item())

# ---- save weights, then the one-time test evaluation ----
os.makedirs("checkpoints", exist_ok=True)
torch.save({"state_dict": model.state_dict(), "config": cfg, "mu": mu, "sd": sd}, f"checkpoints/{name}.pt")
data["val_full"] = {"model": metrics(val_o, sample(val_o)), **baselines(val_o)}
data["test"] = {"model": metrics(test_o, sample(test_o)), **baselines(test_o)}
data["done"] = True
write_json(path, data)

print(f"\nsaved checkpoints/{name}.pt\n{'':12s}{'1h CRPS':>9s}{'24h CRPS':>10s}{'in 50%':>8s}{'in 95%':>8s}{'direction':>11s}")
for split in ("val", "test"):
    res = data["val_full"] if split == "val" else data["test"]
    for k, r in res.items():
        d = f"{r['direction']:.0%}" if "direction" in r else "-"
        print(f"{split + ' ' + k[:6]:12s}{r['crps_h'][0]:9.3f}{r['crps_24']:10.3f}{r['cover50']:8.0%}"
              f"{r['cover95']:8.0%}{d:>11s}")

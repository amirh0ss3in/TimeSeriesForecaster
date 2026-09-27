"""Market LLM: each hour is one step made of all 20 coins' candle numbers (move, range, volume, order flow). A causal
transformer ensemble reads the last `ctx` hours and predicts the full distribution of the next hour for every coin:
Student-t for the move (in units of the coin's volatility, so N(0, 1) is the recent-volatility baseline), Gaussian
for range, volume and order flow. Forecasts further ahead are made LLM-style: sample the next hour, feed it back.

Usage: python market_llm.py [--steps=3000] [--ctx=48] [--d=64] [--layers=2] [--lr=3e-4] [--dropout=0.2] [--aux=0.1]
                            [--test] [--name=...]
  --aux: weight of the other features' loss relative to the move's (the move is what we trade).
Watch it in direction.html?run=<name>. Splits as before: fit < 2024-10-01, Oct-Dec 2024 check, validation 2025,
test 2026 (only with --test). All inputs come from crypto_features.py (leak-tested); standardized with fit statistics.
"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from crypto_eval import direction_stats, line
from crypto_features import features
from model import Ensemble
from monitor import crps, write_json

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
steps, ctx, d_model, layers = int(arg("steps", 3000)), int(arg("ctx", 48)), int(arg("d", 64)), int(arg("layers", 2))
lr, wd, dropout, batch, n_models, heads = float(arg("lr", 3e-4)), 0.1, float(arg("dropout", 0.2)), 32, 5, 4
aux_w, eval_every, use_test = float(arg("aux", 0.1)), 250, "--test" in sys.argv
name = arg("name", f"market_llm_c{ctx}_d{d_model}_l{layers}")
ES, VAL, TEST = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
purge = pd.Timedelta(hours=25)
COINS = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ZEC", "DOGE", "LINK", "ADA", "XLM", "BCH", "NEAR", "UNI", "LTC",
         "SUI", "AVAX", "HBAR", "TAO", "SHIB"]            # market-cap order; the column order of every hour
FEAT = ["z0", "rng", "vol", "imb0", "clv0", "l15", "imb15"]   # per coin: move, range, volume, order flow, candle
                                                                # shape, last-15-minute move and order flow
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the market as a (hours, coins, features) array ----
df = features()
grid = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
wide = {f: df.pivot(index="time", columns="coin", values=f).reindex(index=grid, columns=COINS) for f in FEAT + ["y1"]}
raw = np.stack([wide[f].to_numpy(np.float64) for f in FEAT], -1)                 # (T, C, F)
present = np.isfinite(raw[..., :4]).all(-1)                                       # coin listed and warmed up
y1 = wide["y1"].to_numpy()                                                        # next hour's move (vol units)
n_fit = int(np.searchsorted(grid, ES - purge))                                    # hours usable for fitting
mu_f = np.nanmean(raw[:n_fit], (0, 1))
sd_f = np.nanstd(raw[:n_fit], (0, 1))
xs = np.clip((raw - mu_f) / sd_f, -8, 8)
xs[~present] = 0.0
xs[~np.isfinite(xs)] = 0.0                                                        # a few missing 5-minute values
cyc = lambda v, n: [np.sin(2 * np.pi * v / n), np.cos(2 * np.pi * v / n)]
tfeat = np.stack([*cyc(grid.hour, 24), *cyc(grid.weekday, 7)], -1)
T, C, F = xs.shape
inp = np.concatenate([xs.reshape(T, C * F), present.astype(float), tfeat], 1).astype(np.float32)
Xg = torch.tensor(np.vstack([inp, np.zeros((ctx, inp.shape[1]), np.float32)]), device=dev)
Tg = torch.tensor(np.vstack([xs, np.zeros((ctx, C, F))]).astype(np.float32), device=dev)          # targets
Mg = torch.tensor(np.vstack([present, np.zeros((ctx, C), bool)]), device=dev)
span = torch.arange(ctx, device=dev)
fit_starts = torch.arange(0, n_fit - ctx, device=dev)             # window [s, s+ctx) predicts hours s+1 .. s+ctx


NO = 3 + 2 * (F - 1)                                              # outputs per coin: Student-t (3) + Gaussians


def heads_of(out):
    """Split the (..., C * NO) output into the move's Student-t and the other features' Gaussians."""
    o = out.reshape(*out.shape[:-1], C, NO)
    loc, scale, dfree = o[..., 0], torch.nn.functional.softplus(o[..., 1]) + 1e-3, 2 + torch.nn.functional.softplus(o[..., 2])
    g_mu, g_ls = o[..., 3:3 + F - 1], o[..., 3 + F - 1:].clamp(-7, 3)
    return loc, scale, dfree, g_mu, g_ls


def loss_fn(out, target, mask):
    loc, scale, dfree, g_mu, g_ls = heads_of(out)
    nll_z = -torch.distributions.StudentT(dfree, loc, scale).log_prob(target[..., 0])
    nll_g = (g_ls + 0.5 * ((target[..., 1:] - g_mu) / g_ls.exp()) ** 2).sum(-1)
    m = mask.float()
    return ((nll_z + aux_w * nll_g) * m).sum() / m.sum()


model = Ensemble(n_models, ctx, d_model, heads, layers, window_norm=False, n_in=inp.shape[1], n_out=C * NO,
                 dropout=dropout).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


@torch.no_grad()
def next_hour(lo, hi, chunk=256):
    """For every hour t in [lo, hi): the model's next-hour distribution of each coin's move, from the ctx hours up to
    and including t. Windows step by ctx/2 and keep their second half (>= ctx/2 hours of context). Returns loc, scale,
    df in move units (vol-normalized), each (E, hi - lo, C)."""
    model.eval()
    half = ctx // 2
    starts = torch.arange(lo - half, hi - half + 1, half, device=dev)
    starts = starts[starts >= 0]
    outs = []
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        outs.append(model(Xg[s[:, None] + span].expand(n_models, -1, -1, -1))[:, :, half:])   # (E, n, half, C*NO)
    model.train()
    out = torch.cat(outs, 1)
    loc, scale, dfree, _, _ = heads_of(out)
    rows = (starts[:, None] + torch.arange(half, ctx, device=dev)).reshape(-1).cpu().numpy()
    keep = (rows >= lo) & (rows < hi)
    f = lambda a: a.reshape(n_models, -1, C).cpu().numpy()[:, keep][:, np.argsort(rows[keep], kind="stable")]
    loc, scale, dfree = f(loc) * sd_f[0] + mu_f[0], f(scale) * sd_f[0], f(dfree)      # back to move units
    return loc, scale, dfree                                                          # (E, hi - lo, C)


def summarize(lo, hi, crps_every=6, n_draw=100, seed=0):
    """P(up), expected move and CRPS (vs the N(0,1) recent-volatility baseline) for hours [lo, hi)."""
    loc, scale, dfree = next_hour(lo, hi)
    p_up = (1 - student_t.cdf(-loc / scale, dfree)).mean(0)                          # ensemble-average P(up)
    exp_move = loc.mean(0)
    y = y1[lo:hi]
    ok = present[lo:hi] & np.isfinite(y)
    # CRPS on every `crps_every`-th hour: sample the ensemble mixture (each draw picks a model at random).
    rng = np.random.default_rng(seed)
    sub = np.zeros_like(ok)
    sub[::crps_every] = ok[::crps_every]
    ii, cc = np.nonzero(sub)
    e = rng.integers(0, n_models, (n_draw, len(ii)))
    draws = loc[e, ii, cc] + scale[e, ii, cc] * rng.standard_t(dfree[e, ii, cc])
    base = rng.standard_normal((n_draw, len(ii)))
    return p_up, exp_move, ok, float(crps(draws, y[ii, cc]).mean()), float(crps(base, y[ii, cc]).mean())


def hour_index(ts):
    return int(np.searchsorted(grid, ts))


def report(lo, hi, label, boot=300):
    p_up, exp_move, ok, c_model, c_base = summarize(lo, hi)
    tt = np.broadcast_to(grid[lo:hi].to_numpy()[:, None], ok.shape)
    stats = direction_stats(tt[ok], y1[lo:hi][ok], (p_up - 0.5)[ok], prob=p_up[ok], boot=boot)
    stats.update(crps_model=c_model, crps_baseline=c_base)
    print(line(label, stats) + f"  CRPS {c_model:.4f} vs vol baseline {c_base:.4f} ({c_model / c_base:.3f})")
    return stats, p_up, exp_move, ok


# ---- train ----
v_lo, v_hi = hour_index(VAL), hour_index(TEST - purge)
s_lo, s_hi = hour_index(ES), hour_index(VAL - purge)
cfg = dict(ctx_hours=ctx, coins=COINS, token_features=FEAT, d_model=d_model, layers=layers, heads=heads,
           n_models=n_models, batch=batch, lr=lr, weight_decay=wd, dropout=dropout, aux_weight=aux_w, steps=steps,
           fit_hours=n_fit, params_per_model=sum(p.numel() for p in model.parameters()) // n_models)
print(cfg)
data = {"name": name, "kind": "market LLM (next-hour distribution for all coins)", "config": cfg, "done": False,
        "snapshots": []}
path = f"runs/{name}.json"
print(f"Monitor: http://localhost:8000/direction.html?run={name}")
losses, t0, best = [], time.time(), (1e9, None, 0)
for step in range(steps + 1):
    if step % eval_every == 0 or step == steps:
        with torch.no_grad():
            model.eval()
            sl = []
            for lo in range(s_lo, s_hi - ctx, ctx):
                w = (lo + span)[None, None].expand(n_models, 1, -1)
                sl.append(loss_fn(model(Xg[w]), Tg[w + 1], Mg[w + 1]).item())
            model.train()
        vs, *_ = report(v_lo, v_hi, f"step {step:5d} val", boot=100)
        data["snapshots"].append({"step": step, "train_loss": float(np.mean(losses)) if losses else None,
                                  "stop_loss": float(np.mean(sl)), "val": {k: vs[k] for k in ("hit", "top20", "big", "ic")},
                                  "crps_ratio": vs["crps_model"] / vs["crps_baseline"]})
        if np.mean(sl) < best[0]:          # keep the checkpoint with the best Oct-Dec 2024 loss
            best = (float(np.mean(sl)), {k: v.clone() for k, v in model.state_dict().items()}, step)
        data["best_step"] = best[2]
        write_json(path, data)
        print(f"            train {np.mean(losses) if losses else float('nan'):.4f}  Oct-Dec 2024 {np.mean(sl):.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        losses = []
    if step == steps:
        break
    s = fit_starts[torch.randint(0, len(fit_starts), (n_models, batch), device=dev)]
    rows = s[..., None] + span
    loss = loss_fn(model(Xg[rows]), Tg[rows + 1], Mg[rows + 1])
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.item())

# ---- final: best checkpoint; validation (and test) per coin; predictions saved for the backtest ----
model.load_state_dict(best[1])
print(f"best checkpoint (Oct-Dec 2024 loss): step {best[2]}")
os.makedirs("checkpoints", exist_ok=True)
torch.save({"state_dict": model.state_dict(), "config": cfg, "mu": mu_f, "sd": sd_f}, f"checkpoints/{name}.pt")


def final(lo, hi, tag):
    stats, p_up, exp_move, ok = report(lo, hi, f"{tag}, pooled")
    per_coin = {}
    tt = grid[lo:hi].to_numpy()
    for j, c in enumerate(COINS):
        m = ok[:, j]
        if m.sum() > 1000:
            per_coin[c] = direction_stats(tt[m], y1[lo:hi, j][m], p_up[m, j] - 0.5, prob=p_up[m, j], boot=200)
            print(line(f"  {c}", per_coin[c]))
    ii, jj = np.nonzero(ok)
    np.savez(f"checkpoints/preds_{name}_{tag.split()[0].lower()}.npz", coin=np.array(COINS)[jj], time=tt[ii],
             prob=p_up[ii, jj], exp_move=exp_move[ii, jj])
    return {"pooled": stats, "per_coin": per_coin}


data.update(done=True, val=final(v_lo, v_hi, "VAL 2025"))
if use_test:
    data["test"] = final(hour_index(TEST), len(grid) - 1, "TEST 2026")["pooled"]
write_json(path, data)
print(f"saved runs/{name}.json and checkpoints/preds_{name}_*.npz")

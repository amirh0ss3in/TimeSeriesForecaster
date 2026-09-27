"""Direction transformer: reads the last `ctx` hours of feature vectors (crypto_features.py) and predicts, at every
hour, the probability that the next hour's move is up (5-model ensemble, stacked weights, probabilities averaged).
Watch it in direction.html?run=<name>.

Usage: python crypto_seq.py [--coins=BTC,...] [--steps=4000] [--ctx=48] [--d=64] [--layers=2] [--lr=1e-3]
                            [--dropout=0] [--wd=0.05] [--eval-every=250] [--select=stop|last] [--perp] [--test] [--name=...]

Same splits as crypto_probe.py: fit < 2024-10-01, Oct-Dec 2024 picks the best checkpoint, validation = 2025,
test = 2026 (only with --test). Inputs are standardized with fit-period statistics; attention is causal, so each
prediction uses only its own hour and the hours before it.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from crypto_eval import direction_stats, line
from crypto_features import INPUTS, PERP_INPUTS, features
from model import Ensemble
from monitor import write_json

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
coins = arg("coins", "")
steps, ctx, d_model, layers = int(arg("steps", 4000)), int(arg("ctx", 48)), int(arg("d", 64)), int(arg("layers", 2))
heads, n_models, batch, lr, eval_every = 4, 5, 64, float(arg("lr", 1e-3)), int(arg("eval-every", 250))
dropout, wd = float(arg("dropout", 0)), float(arg("wd", 0.05))
select = arg("select", "stop")   # stop: best Oct-Dec 2024 loss; last: final step
use_test = "--test" in sys.argv
name = arg("name", f"direction_seq_{coins.replace(',', '_') or 'all'}_c{ctx}")
ES, VAL, TEST = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
purge = pd.Timedelta(hours=25)
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- data: one row per (coin, hour), sorted by coin then time ----
df = features()
if coins:
    df = df[df["coin"].isin(coins.split(","))]
df = df.sort_values(["coin", "time"]).reset_index(drop=True)
num = [c for c in INPUTS if c not in ("hour", "weekday")] + (PERP_INPUTS if "--perp" in sys.argv else [])
fit_rows = df["time"] < ES - purge
mu_f, sd_f = df.loc[fit_rows, num].mean(), df.loc[fit_rows, num].std()     # fit-period statistics only
cyc = lambda v, n: [np.sin(2 * np.pi * v / n), np.cos(2 * np.pi * v / n)]
X = np.column_stack([((df[num] - mu_f) / sd_f).clip(-5, 5).fillna(0).to_numpy(np.float32),
                     *cyc(df["hour"].to_numpy(), 24), *cyc(df["weekday"].to_numpy(), 7)]).astype(np.float32)
y = df["y1"].to_numpy(np.float32)
valid = np.isfinite(y) & (y != 0)
t = df["time"].to_numpy()
seg = np.flatnonzero(np.r_[True, df["coin"].to_numpy()[1:] != df["coin"].to_numpy()[:-1], True])   # coin boundaries

Xg = torch.tensor(np.vstack([X, np.zeros((ctx, X.shape[1]), np.float32)]), device=dev)    # padded at the end
Lg = torch.tensor(np.r_[(y > 0).astype(np.float32), np.zeros(ctx, np.float32)], device=dev)
Vg = torch.tensor(np.r_[valid, np.zeros(ctx, bool)], device=dev)
span = torch.arange(ctx, device=dev)

# Training windows: ctx consecutive rows of one coin, all in the fit period.
fit_starts = []
for a, b in zip(seg[:-1], seg[1:]):
    f = a + int(np.searchsorted(t[a:b], np.datetime64(ES - purge)))
    fit_starts.append(np.arange(a, f - ctx + 1))
fit_starts = torch.tensor(np.concatenate(fit_starts), device=dev)


def split_rows(lo, hi):
    """Per coin, the contiguous row range [p, q) with lo <= time < hi."""
    return [(a + int(np.searchsorted(t[a:b], np.datetime64(lo))), a + int(np.searchsorted(t[a:b], np.datetime64(hi))))
            for a, b in zip(seg[:-1], seg[1:])]


SPLITS = {"stop": split_rows(ES, VAL - purge), "val": split_rows(VAL, TEST - purge),
          "test": split_rows(TEST, pd.Timestamp("2100-01-01"))}

# ---- model ----
model = Ensemble(n_models, ctx, d_model, heads, layers, window_norm=False, n_in=X.shape[1], n_out=1,
                 dropout=dropout).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


@torch.no_grad()
def predict(ranges, chunk=512):
    model.eval()
    """P(up) for every row in the given ranges. Windows step by ctx/2 and keep their second half, so every row is
    predicted once, with at least ctx/2 hours of context; later rows in a window can't affect earlier ones (causal)."""
    half = ctx // 2
    starts, keep_idx = [], []
    for p, q in ranges:
        for w in range(p - half, q - half, half):
            starts.append(w)
    starts = torch.tensor(starts, device=dev)
    probs = []
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        logit = model(Xg[s[:, None] + span].expand(n_models, -1, -1, -1))[..., 0]   # (E, n, ctx)
        probs.append(torch.sigmoid(logit).mean(0)[:, half:])
    model.train()
    probs = torch.cat(probs).cpu().numpy().ravel()
    rows = (starts.cpu().numpy()[:, None] + np.arange(half, ctx)).ravel()
    inside = np.zeros(len(X) + ctx, bool)
    for p, q in ranges:
        inside[p:q] = True
    m = inside[rows]
    out = np.full(len(X), np.nan)
    out[rows[m]] = probs[m]
    return out


def evaluate(split):
    ranges = SPLITS[split]
    prob = predict(ranges)
    idx = np.concatenate([np.arange(p, q) for p, q in ranges])
    return idx, prob[idx]


def logloss(idx, prob):
    m = valid[idx]
    yy, pp = (y[idx][m] > 0), np.clip(prob[m], 1e-6, 1 - 1e-6)
    return float(-(yy * np.log(pp) + (~yy) * np.log(1 - pp)).mean())


# ---- train ----
cfg = dict(coins=coins or "all 20", ctx=ctx, d_model=d_model, heads=heads, layers=layers, n_models=n_models,
           batch=batch, lr=lr, select=select, dropout=dropout, weight_decay=wd, steps=steps, inputs=num + ["hour sin/cos", "weekday sin/cos"],
           fit_windows=len(fit_starts), params_per_model=sum(p.numel() for p in model.parameters()) // n_models)
print(cfg)
ref = "runs/direction_lgbm.json"
data = {"name": name, "kind": "transformer", "config": cfg, "done": False, "snapshots": [],
        "reference": {"lightgbm (all coins)": __import__("json").load(open(ref))["val"]["pooled"]} if os.path.exists(ref) else {}}
path = f"runs/{name}.json"
print(f"Monitor: http://localhost:8000/direction.html?run={name}")
best, best_state, losses, t0 = 1e9, None, [], time.time()
for step in range(steps + 1):
    if step % eval_every == 0 or step == steps:
        si, sp = evaluate("stop")
        vi, vp = evaluate("val")
        sl = logloss(si, sp)
        vs = direction_stats(t[vi], y[vi], vp - 0.5, boot=100)
        if sl < best or (select == "last" and step == steps):
            best, best_state, best_step = sl, {k: v.clone() for k, v in model.state_dict().items()}, step
        data["snapshots"].append({"step": step, "train_loss": float(np.mean(losses)) if losses else None,
                                  "stop_loss": sl, "val": {k: vs[k] for k in ("hit", "top20", "big", "ic")}})
        write_json(path, data)
        print(f"step {step:5d}  train {np.mean(losses) if losses else float('nan'):.4f}  stop {sl:.4f}"
              f"{' *' if best_step == step else '  '}  " + line("val", vs)[4:] + f"  ({time.time() - t0:.0f}s)")
        losses = []
    if step == steps:
        break
    s = fit_starts[torch.randint(0, len(fit_starts), (n_models, batch), device=dev)]
    rows = s[..., None] + span
    logit = model(Xg[rows])[..., 0]
    v = Vg[rows].float()
    loss = (F.binary_cross_entropy_with_logits(logit, Lg[rows], reduction="none") * v).sum() / v.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.item())

# ---- final: best checkpoint by the Oct-Dec 2024 loss ----
model.load_state_dict(best_state)
os.makedirs("checkpoints", exist_ok=True)
torch.save({"state_dict": best_state, "config": cfg, "mu": mu_f, "sd": sd_f, "step": best_step}, f"checkpoints/{name}.pt")
vi, vp = evaluate("val")
pooled = direction_stats(t[vi], y[vi], vp - 0.5, prob=vp)
print(f"\nbest checkpoint: step {best_step}\n" + line("val 2025, pooled", pooled))
per_coin = {}
cc = df["coin"].to_numpy()
for c in sorted(set(cc[vi])):
    m = cc[vi] == c
    per_coin[c] = direction_stats(t[vi][m], y[vi][m], vp[m] - 0.5, prob=vp[m], boot=200)
    print(line(f"  {c}", per_coin[c]))
data.update(done=True, best_step=best_step, val={"pooled": pooled, "per_coin": per_coin})
np.savez(f"checkpoints/preds_{name}.npz", coin=cc[vi], time=t[vi], prob=vp)     # for combining models later
if use_test:
    ti, tp = evaluate("test")
    data["test"] = direction_stats(t[ti], y[ti], tp - 0.5, prob=tp)
    print(line("TEST 2026, pooled", data["test"]))
    data["test_per_coin"] = {}
    for c in sorted(set(cc[ti])):
        m = cc[ti] == c
        data["test_per_coin"][c] = direction_stats(t[ti][m], y[ti][m], tp[m] - 0.5, boot=200)
        print(line(f"  TEST {c}", data["test_per_coin"][c]))
write_json(path, data)

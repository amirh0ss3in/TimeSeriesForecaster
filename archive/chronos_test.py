"""(Never run: replaced by kronos_test.py.) Zero-shot Chronos in our framework: for every coin and hour, give Chronos the last `ctx` hourly closes (all known
when that hour's candle has closed) and read P(next close > current close) from its forecast quantiles. No training.

Usage: python chronos_test.py [--model=amazon/chronos-bolt-small] [--from=2025-01-01] [--to=2026-09-01] [--ctx=512]
                              [--save=chronos_bolt_small] [--limit=N]
Writes checkpoints/preds_<save>.npz (coin, time, prob) for crypto_eval / crypto_backtest, and prints direction stats.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

from crypto_eval import direction_stats, line
from crypto_features import features

opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
model_id, ctx = opt("model", "amazon/chronos-bolt-small"), int(opt("ctx", 512))
start, end, save, limit = pd.Timestamp(opt("from", "2025-01-01")), pd.Timestamp(opt("to", "2026-09-01")), \
    opt("save", "chronos_bolt_small"), int(opt("limit", 0))
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

df = features()[["coin", "time", "y1"]]
df = df[(df["time"] >= start) & (df["time"] < end) & df["y1"].notna()]
pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float32)

rows = []
for coin, g in df.groupby("coin"):
    c = pd.read_csv(f"data/crypto/{coin}.csv.gz", parse_dates=["time"]).set_index("time")["close"]
    c = c.reindex(pd.date_range(c.index[0], c.index[-1], freq="h")).ffill()        # missing hours keep last close
    pos = c.index.get_indexer(g["time"])
    ok = pos >= ctx - 1
    rows.append(pd.DataFrame({"coin": coin, "time": g["time"].to_numpy()[ok], "y1": g["y1"].to_numpy()[ok],
                              "i": pos[ok], "series": [c.to_numpy()] * ok.sum()}))
df = pd.concat(rows, ignore_index=True)
if limit:
    df = df.sample(limit, random_state=0).sort_values(["coin", "time"]).reset_index(drop=True)
print(f"{model_id}: {len(df)} coin-hours, {df['time'].min()} .. {df['time'].max()}, context {ctx} hours")

probs, t0, B = np.empty(len(df)), time.time(), 1024
for s in range(0, len(df), B):
    part = df.iloc[s:s + B]
    ctxs = np.stack([ser[i - ctx + 1:i + 1] for ser, i in zip(part["series"], part["i"])]).astype(np.float32)
    q, _ = pipe.predict_quantiles(torch.tensor(ctxs), prediction_length=1, quantile_levels=QL)
    q = q[:, 0, :].cpu().numpy()                                     # (batch, 9): next close at each quantile
    last = ctxs[:, -1]
    # P(next <= last) by interpolating the quantile curve at the last close; beyond the 10%/90% quantiles, clamp.
    below = np.array([np.interp(l, qq, QL, left=0.05, right=0.95) for l, qq in zip(last, np.maximum.accumulate(q, 1))])
    probs[s:s + B] = 1 - below
    if s == 0 or (s // B) % 20 == 0:
        done = s + len(part)
        print(f"  {done}/{len(df)}  ({done / (time.time() - t0):.0f} forecasts/s)", flush=True)

np.savez(f"checkpoints/preds_{save}.npz", coin=df["coin"].to_numpy(), time=df["time"].to_numpy(), prob=probs)
print(line(f"{save}, pooled", direction_stats(df["time"].to_numpy(), df["y1"].to_numpy(), probs - 0.5, prob=probs)))
for per, (a, b) in {"2025": ("2025-01-01", "2026-01-01"), "2026 Jan-Aug": ("2026-01-01", "2026-09-01")}.items():
    m = ((df["time"] >= a) & (df["time"] < b)).to_numpy()
    if m.sum() > 1000:
        print(line(f"  {per}", direction_stats(df["time"].to_numpy()[m], df["y1"].to_numpy()[m], probs[m] - 0.5)))
print(f"saved checkpoints/preds_{save}.npz")

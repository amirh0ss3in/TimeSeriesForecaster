"""Zero-shot Kronos (github.com/shiyu-coder/Kronos, cloned to ../external/Kronos) in our framework: for a coin and hour t,
give Kronos the last `ctx` hourly candles (open, high, low, close, volume, amount = quote volume; all closed by the
decision time), sample the next candle `samples` times, and read P(up) = share of samples whose close is above the
current close. No training or fine-tuning.

Usage: python kronos_test.py [--model=NeoQuasar/Kronos-small] [--ctx=80] [--samples=10] [--every=12] [--no-volume]
                             [--from=2025-01-01] [--to=2026-09-01] [--save=kronos_small] [--speed]
  --every=N: one decision hour in N (per coin), to keep the run short; LightGBM is scored on the same coin-hours.
  Defaults follow the paper's hourly protocol (80-hour lookback, 10 samples). --no-volume: prices only, as the paper
  evaluated crypto. Two signals are scored: P(up) from the samples, and the paper's mean sampled close vs close.
Writes checkpoints/preds_<save>.npz (coin, time, prob) and prints direction stats next to LightGBM's.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "Kronos"))
from model import Kronos, KronosTokenizer                          # noqa: E402  (Kronos repo)
from model.kronos import auto_regressive_inference, calc_time_stamps  # noqa: E402

from crypto_eval import direction_stats, line                      # noqa: E402
from crypto_features import features                               # noqa: E402

opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
model_id, ctx, S, every = opt("model", "NeoQuasar/Kronos-small"), int(opt("ctx", 80)), int(opt("samples", 10)), \
    int(opt("every", 12))
start, end, save = pd.Timestamp(opt("from", "2025-01-01")), pd.Timestamp(opt("to", "2026-09-01")), opt("save", "kronos_small")
dev = "cuda"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").to(dev).eval()
mdl = Kronos.from_pretrained(model_id).to(dev).eval()
torch.manual_seed(0)

# Decision rows: every `every`-th hour of each coin in [start, end), where the target (next hour's move) exists.
f = features()[["coin", "time", "y1"]]
f = f[(f["time"] >= start) & (f["time"] < end) & f["y1"].notna()]
f = f[(f["time"] - start).dt.total_seconds().astype(int) // 3600 % every == 0]
candles = {}
for c in f["coin"].unique():
    d = pd.read_csv(f"data/crypto/{c}.csv.gz", parse_dates=["time"]).set_index("time")
    d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq="h"))
    d["close"] = d["close"].ffill()                                    # outage hours: flat candle, no volume
    for k in ("open", "high", "low"):
        d[k] = d[k].fillna(d["close"])
    d["volume"], d["amount"] = d["volume"].fillna(0.0), d["quote_volume"].fillna(0.0)
    if "--no-volume" in sys.argv:
        d["volume"], d["amount"] = 0.0, 0.0
    candles[c] = (d[["open", "high", "low", "close", "volume", "amount"]].to_numpy(np.float32),
                  calc_time_stamps(pd.Series(d.index)).to_numpy(np.float32), d.index)
pos = np.array([candles[c][2].get_loc(t) for c, t in zip(f["coin"], f["time"])])
keep = pos >= ctx - 1
f, pos = f[keep].reset_index(drop=True), pos[keep]
if "--speed" in sys.argv:
    f, pos = f.iloc[:256], pos[:256]
print(f"{model_id}: {len(f)} coin-hours (every {every}h), {f['time'].min()} .. {f['time'].max()}, "
      f"context {ctx}h, {S} samples each")


def batch_probs(idx):
    xs, xst, yst, last = [], [], [], []
    for i in idx:
        arr, stamps, _ = candles[f["coin"].iat[i]]
        p = pos[i]
        x = arr[p - ctx + 1:p + 1]
        m, s = x.mean(0), x.std(0)
        xs.append(np.clip((x - m) / (s + 1e-5), -5, 5))                # Kronos' own per-window normalization
        xst.append(stamps[p - ctx + 1:p + 1])
        yst.append(stamps[p + 1:p + 2] if p + 1 < len(stamps) else stamps[p:p + 1] * 0 + calc_time_stamps(
            pd.Series([candles[f["coin"].iat[i]][2][p] + pd.Timedelta(hours=1)])).to_numpy(np.float32))
        last.append(xs[-1][-1, 3])
    rep = lambda a: torch.tensor(np.repeat(np.stack(a), S, axis=0), device=dev)   # S copies = S independent samples
    out = auto_regressive_inference(tok, mdl, rep(xs), rep(xst), rep(yst), ctx, 1, clip=5, T=1.0, top_p=0.9,
                                    sample_count=1)                    # (len(idx) * S, ctx + 1, 6), normalized
    nxt = out[:, -1, 3].reshape(len(idx), S)                           # sampled next close (normalized units)
    last = np.array(last)[:, None]
    p_up = (nxt > last).mean(1) + 0.5 * (nxt == last).mean(1)
    return p_up, nxt.mean(1) - last[:, 0]                              # P(up), and the paper's mean-path signal


B, probs, meansig, t0 = max(1, 512 // S), np.empty(len(f)), np.empty(len(f)), time.time()
for s in range(0, len(f), B):
    probs[s:s + B], meansig[s:s + B] = batch_probs(range(s, min(s + B, len(f))))
    if (s // B) % 25 == 0:
        done = min(s + B, len(f))
        print(f"  {done}/{len(f)}  {done / (time.time() - t0):.1f} coin-hours/s", flush=True)
if "--speed" in sys.argv:
    sys.exit()

np.savez(f"checkpoints/preds_{save}.npz", coin=f["coin"].to_numpy(), time=f["time"].to_numpy(), prob=probs,
         mean_signal=meansig)
t, y = f["time"].to_numpy(), f["y1"].to_numpy()
# LightGBM (monthly refit, out of sample) on exactly the same coin-hours, for a fair comparison.
g = np.load("checkpoints/preds_lgbm_w24_walk.npz", allow_pickle=True)
g = pd.DataFrame({"coin": g["coin"], "time": pd.to_datetime(g["time"]), "lgbm": g["prob"]})
m = f.assign(kronos=probs, kronos_mean=meansig).merge(g, on=["coin", "time"], how="left")
for per, (a, b) in {"all": (start, end), "2025": ("2025-01-01", "2026-01-01"), "2026 Jan-Aug": ("2026-01-01", "2026-09-01")}.items():
    k = ((m["time"] >= a) & (m["time"] < b) & m["lgbm"].notna()).to_numpy()
    if k.sum() > 500:
        for col, center in (("kronos", 0.5), ("kronos_mean", 0.0), ("lgbm", 0.5)):
            print(line(f"{per:12s} {col}", direction_stats(m["time"].to_numpy()[k], m["y1"].to_numpy()[k],
                                                            m[col].to_numpy()[k] - center, boot=200)))
print(f"correlation of the two models' P(up): {m[['kronos', 'lgbm']].corr().iloc[0, 1]:.3f}")
print(f"saved checkpoints/preds_{save}.npz")

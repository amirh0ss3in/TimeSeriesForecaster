"""Backtest of hourly direction predictions: does the edge survive trading costs? Goal: > 3% net per month.

Usage: python crypto_backtest.py --dev=NAME[,NAME...] [--holdout=NAME[,NAME...]] [--signal=prob|move]
                                 [--alloc=slices|active] [--final] [--save=NAME]
  NAMEs are checkpoints/preds_<NAME>.npz files (P(up) per coin-hour, optionally exp_move). Development = all their
  hours before HOLDOUT (2025 validation + 2026 walk-forward); holdout = hours from HOLDOUT on, from --holdout files
  (default: the dev files). The holdout is only evaluated with --final.

Rules, fixed before looking at results:
- Perpetual futures (shorting allowed), no leverage. --alloc=slices: each coin always gets 1 / n_coins of the capital
  (idle when flat). --alloc=active: each hour the capital is split equally among the coins with a position.
- A coin goes long/short when its edge |P(up) - 0.5| reaches the entry threshold, and stays in while the edge keeps
  the same sign and is at least the exit threshold (= exit ratio x entry threshold); otherwise it is flat.
  --signal=move: the edge is the expected move converted to a return with the coin's current volatility, in
  round-trip fees.
- Each hour's position earns the coin's next-hour log return. Fees on every change of capital weight: maker 0.02%,
  taker 0.05% per side (Binance USD-M futures). Funding ignored.
- Entry threshold and exit ratio are chosen on development data only: the pair whose worse period (2025 or 2026)
  has the best mean monthly net return with maker fees. 95% ranges resample whole days.
"""
import sys

import numpy as np
import pandas as pd

from crypto_features import features
from monitor import write_json

FEES = {"maker 0.02%": 0.0002, "taker 0.05%": 0.0005}
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
EXIT_RATIOS = [1.0, 0.5, 0.25, 0.0]
HOLDOUT = pd.Timestamp("2026-09-01")
PERIODS = {"2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
           "2026 Jan-Aug": (pd.Timestamp("2026-01-01"), HOLDOUT)}


def load(names):
    return pd.concat([load_one(n) for n in names.split(",")]).drop_duplicates(["coin", "time"], keep="last")


def load_one(name):
    d = np.load(f"checkpoints/preds_{name}.npz", allow_pickle=True)
    p = pd.DataFrame({"coin": d["coin"], "time": pd.to_datetime(d["time"]), "prob": d["prob"]})
    if "exp_move" in d:
        p["exp_move"] = d["exp_move"]
    return p.merge(features()[["coin", "time", "r1", "sig"]], on=["coin", "time"])


def positions(edge, thr_in, thr_out):
    """Enter at |edge| >= thr_in, stay while the sign holds and |edge| >= thr_out, else flat."""
    e = np.nan_to_num(edge.to_numpy())
    pos, cur = np.zeros_like(e), np.zeros(e.shape[1])
    for i in range(len(e)):
        s = np.sign(e[i])
        keep = (cur != 0) & (s == cur) & (np.abs(e[i]) >= thr_out)
        cur = np.where(np.abs(e[i]) >= thr_in, s, np.where(keep, cur, 0.0))
        cur[e[i] == 0] = 0.0
        pos[i] = cur
    return pd.DataFrame(pos, index=edge.index, columns=edge.columns)


def run(p, thr, fee, boot=300, seed=0, alloc="slices", signal="prob", exit_ratio=1.0):
    if signal == "move":                     # edge measured in round-trip fees: expected return / (2 x fee)
        p = p.assign(prob=0.5 + p["exp_move"] * p["sig"] / (2 * fee))
    prob = p.pivot(index="time", columns="coin", values="prob")
    r1 = p.pivot(index="time", columns="coin", values="r1").reindex_like(prob)
    grid = pd.date_range(prob.index.min(), prob.index.max(), freq="h")
    prob, r1 = prob.reindex(grid), r1.reindex(grid)
    edge = prob - 0.5
    pos = positions(edge, thr, thr * exit_ratio)
    n_active = (pos != 0).sum(1)
    weight = pos.div(n_active.where(n_active > 0, 1), axis=0) if alloc == "active" else pos / pos.shape[1]
    cost = (weight - weight.shift(1).fillna(0.0)).abs() * fee   # fees on every change of capital weight
    pnl = weight * r1.fillna(0.0) - cost                        # contribution of each coin to the portfolio
    port = pnl.sum(1)
    traded = pos != 0
    n_tr = int(traded.sum().sum())
    hit = float((np.sign(r1[traded]) == pos[traded]).sum().sum() / max(n_tr, 1))
    monthly = port.groupby(port.index.to_period("M")).sum()
    days = port.groupby(port.index.floor("D")).sum().to_numpy()
    draws = np.random.default_rng(seed).integers(0, len(days), (boot, len(days)))
    per_month = days[draws].mean(1) * 30.44                    # mean monthly return, resampled by day
    eq = port.cumsum()
    return {"threshold": thr, "exit_ratio": exit_ratio, "fee": fee, "monthly_mean": float(monthly.mean()),
            "monthly_ci": [float(x) for x in np.percentile(per_month, [2.5, 97.5])],
            "monthly": {str(k): float(v) for k, v in monthly.items()}, "trades_per_day": n_tr / (len(grid) / 24),
            "hit_on_trades": hit, "net_per_trade": float((pos * r1.fillna(0.0) - (pos - pos.shift(1).fillna(0.0)).abs()
                                                          * fee).sum().sum() / max(n_tr, 1)),
            "hours_in_market": float((n_active > 0).mean()),
            "max_drawdown": float((eq - eq.cummax()).min()), "total": float(eq.iloc[-1]),
            "equity": [[str(t), float(v)] for t, v in eq.iloc[::24].items()]}


def show(tag, r):
    lo, hi = r["monthly_ci"]
    print(f"{tag:34s} in {r['threshold']:.2f} out x{r['exit_ratio']:.2f}  monthly {r['monthly_mean']:+.2%} [{lo:+.2%}, {hi:+.2%}]  "
          f"trades/day {r['trades_per_day']:6.1f}  hit {r['hit_on_trades']:.1%}  per trade {r['net_per_trade']:+.4%}  "
          f"max DD {r['max_drawdown']:+.1%}  in market {r['hours_in_market']:.0%}")


if __name__ == "__main__":
    opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    save, alloc, signal = opt("save", ""), opt("alloc", "slices"), opt("signal", "prob")
    if signal == "move":
        THRESHOLDS[:] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]    # expected move of at least k round-trip fees
    dev_all = load(opt("dev", ""))
    dev = {k: dev_all[(dev_all["time"] >= a) & (dev_all["time"] < b)] for k, (a, b) in PERIODS.items()}
    for k, d in dev.items():
        print(f"development {k}: {len(d)} coin-hours, {d['time'].min()} .. {d['time'].max()}")
    fee_m = FEES["maker 0.02%"]
    print("\n-- entry threshold x exit ratio on DEVELOPMENT data (maker fees): mean monthly net per period --")
    grid = []
    for t in THRESHOLDS:
        for xr in EXIT_RATIOS:
            res = {k: run(d, t, fee_m, boot=50, alloc=alloc, signal=signal, exit_ratio=xr) for k, d in dev.items()}
            g = {"threshold": t, "exit_ratio": xr, **{k: r["monthly_mean"] for k, r in res.items()},
                 "trades_per_day": np.mean([r["trades_per_day"] for r in res.values()]),
                 "net_per_trade": np.mean([r["net_per_trade"] for r in res.values()]),
                 "monthly_mean": min(r["monthly_mean"] for r in res.values())}
            grid.append(g)
            print(f"in {t:5.2f} out x{xr:.2f}   " + "   ".join(f"{k} {g[k]:+.2%}" for k in dev) +
                  f"   worse period {g['monthly_mean']:+.2%}   trades/day {g['trades_per_day']:6.1f}"
                  f"   per trade {g['net_per_trade']:+.4%}")
    best = max(grid, key=lambda g: g["monthly_mean"])
    print(f"\nchosen on development data: entry {best['threshold']}, exit ratio {best['exit_ratio']}\n")
    out = {}
    for fname, fee in FEES.items():
        for k, d in dev.items():
            out[f"dev {k} {fname}"] = r = run(d, best["threshold"], fee, alloc=alloc, signal=signal,
                                              exit_ratio=best["exit_ratio"])
            show(f"DEV {k} {fname}", r)
    if "--final" in sys.argv:                              # the one-time holdout evaluation
        ho = load(opt("holdout", opt("dev", "")))
        ho = ho[ho["time"] >= HOLDOUT]
        print(f"\nHOLDOUT: {len(ho)} coin-hours, {ho['time'].min()} .. {ho['time'].max()}")
        for fname, fee in FEES.items():
            out[f"holdout Sep 2026 {fname}"] = r = run(ho, best["threshold"], fee, alloc=alloc, signal=signal,
                                                       exit_ratio=best["exit_ratio"])
            show(f"HOLDOUT {fname}", r)
    if save:
        write_json(f"runs/{save}.json", {"name": save, "val_preds": opt("dev", ""), "test_preds": opt("holdout", opt("dev", "")),
                                         "alloc": alloc, "signal": signal, "threshold": best["threshold"],
                                         "exit_ratio": best["exit_ratio"], "grid": grid, "results": out,
                                         "goal_monthly": 0.03})
        print(f"saved runs/{save}.json")

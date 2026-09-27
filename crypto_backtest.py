"""Backtest of hourly direction predictions: does the edge survive trading costs? Goal: > 3% net per month.

Usage: python crypto_backtest.py VAL_PREDS TEST_PREDS [--alloc=slices|active] [--signal=prob|move] [--save=NAME]
  --signal=move (needs exp_move in the files): trade when the expected move, converted to a return with the coin's
  current volatility, exceeds a threshold given in round-trip fees; the threshold grid is then in those units.
  VAL_PREDS / TEST_PREDS name checkpoints/preds_<NAME>.npz files (P(up) per coin-hour, saved by the model scripts).

Rules, fixed before looking at results:
- Perpetual futures (shorting allowed), no leverage. --alloc=slices: each coin always gets 1 / n_coins of the capital
  (idle when flat). --alloc=active: each hour the capital is split equally among the coins with a position (all of it
  on one coin if only one signal fires); fees are charged on the change of each coin's capital weight.
- Every hour each coin is long (+1), short (-1) or flat (0): long/short when |P(up) - 0.5| >= threshold.
- The position is held for that hour and earns the coin's next-hour log return (known only after the hour).
- Fees are paid only when the position changes: |new - old| x fee per side. Maker 0.02%, taker 0.05% (Binance
  USD-M futures, regular tier). Funding payments are ignored (small for hour-long holds).
- The threshold is chosen on the validation file (best mean monthly net return, maker fees) and then used unchanged on
  the test file. 95% ranges resample whole days.
"""
import sys

import numpy as np
import pandas as pd

from crypto_features import features
from monitor import write_json

FEES = {"maker 0.02%": 0.0002, "taker 0.05%": 0.0005}
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]


def load(name):
    d = np.load(f"checkpoints/preds_{name}.npz", allow_pickle=True)
    p = pd.DataFrame({"coin": d["coin"], "time": pd.to_datetime(d["time"]), "prob": d["prob"]})
    if "exp_move" in d:
        p["exp_move"] = d["exp_move"]
    return p.merge(features()[["coin", "time", "r1", "sig"]], on=["coin", "time"])


def run(p, thr, fee, boot=300, seed=0, alloc="slices", signal="prob"):
    if signal == "move":                     # edge measured in round-trip fees: expected return / (2 x fee)
        p = p.assign(prob=0.5 + p["exp_move"] * p["sig"] / (2 * fee))
    prob = p.pivot(index="time", columns="coin", values="prob")
    r1 = p.pivot(index="time", columns="coin", values="r1").reindex_like(prob)
    grid = pd.date_range(prob.index.min(), prob.index.max(), freq="h")
    prob, r1 = prob.reindex(grid), r1.reindex(grid)
    edge = prob - 0.5
    pos = np.sign(edge).where(edge.abs() >= thr, 0.0).fillna(0.0)
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
    return {"threshold": thr, "fee": fee, "monthly_mean": float(monthly.mean()),
            "monthly_ci": [float(x) for x in np.percentile(per_month, [2.5, 97.5])],
            "monthly": {str(k): float(v) for k, v in monthly.items()}, "trades_per_day": n_tr / (len(grid) / 24),
            "hit_on_trades": hit, "net_per_trade": float((pos * r1.fillna(0.0) - (pos - pos.shift(1).fillna(0.0)).abs()
                                                          * fee).sum().sum() / max(n_tr, 1)),
            "hours_in_market": float((n_active > 0).mean()),
            "max_drawdown": float((eq - eq.cummax()).min()), "total": float(eq.iloc[-1]),
            "equity": [[str(t), float(v)] for t, v in eq.iloc[::24].items()]}


def show(tag, r):
    lo, hi = r["monthly_ci"]
    print(f"{tag:34s} thr {r['threshold']:.2f}  monthly {r['monthly_mean']:+.2%} [{lo:+.2%}, {hi:+.2%}]  "
          f"trades/day {r['trades_per_day']:6.1f}  hit {r['hit_on_trades']:.1%}  per trade {r['net_per_trade']:+.4%}  "
          f"max DD {r['max_drawdown']:+.1%}  in market {r['hours_in_market']:.0%}")


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--save=")), "")
    alloc = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--alloc=")), "slices")
    signal = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--signal=")), "prob")
    if signal == "move":
        THRESHOLDS[:] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]    # expected move of at least k round-trip fees
    val, test = load(names[0]), load(names[1])
    print(f"validation: {names[0]} ({val['time'].min():%Y-%m} .. {val['time'].max():%Y-%m}), "
          f"test: {names[1]} ({test['time'].min():%Y-%m} .. {test['time'].max():%Y-%m})")
    print("\n-- threshold grid on VALIDATION (maker fees) --")
    grid = [run(val, t, FEES["maker 0.02%"], boot=100, alloc=alloc, signal=signal) for t in THRESHOLDS]
    for r in grid:
        show("val", r)
    best = max(grid, key=lambda r: r["monthly_mean"])
    print(f"\nchosen on validation: threshold {best['threshold']:.2f}\n")
    out = {}
    for fname, fee in FEES.items():
        for split, p in (("val", val), ("test", test)):
            out[f"{split} {fname}"] = r = run(p, best["threshold"], fee, alloc=alloc, signal=signal)
            show(f"{split.upper()} {fname}", r)
    print("\nTEST monthly (maker):", "  ".join(f"{k} {v:+.2%}" for k, v in out["test maker 0.02%"]["monthly"].items()))
    if save:
        write_json(f"runs/{save}.json", {"name": save, "val_preds": names[0], "test_preds": names[1], "alloc": alloc, "signal": signal,
                                         "threshold": best["threshold"], "grid": [{k: r[k] for k in (
                                             "threshold", "monthly_mean", "trades_per_day", "hit_on_trades",
                                             "net_per_trade")} for r in grid], "results": out, "goal_monthly": 0.03})
        print(f"saved runs/{save}.json")

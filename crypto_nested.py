"""Honest test of the whole procedure: at the start of every month, choose the trading setting using only earlier
months, then trade that month with it. Every reported month is out of sample, including the choice itself.

Usage: python crypto_nested.py [--preds=lgbm_w24_walk] [--start=2025-01-01] [--save=nested]

Predictions: LightGBM refit at the start of every month on all earlier data (crypto_probe.py --walk
--walk-from=2024-01-01), so each month's predictions are out of sample too.
Candidates: allocation (4) x entry threshold (6) x exit ratio (4) = 96 settings, funding included.
Choice for month M, fixed before looking at results: the setting with the best mean monthly net return (maker fees)
over all months before M, where each setting's score is averaged with its neighbouring entry thresholds (same
allocation and exit ratio), so a stable range beats a lone peak. If no smoothed score is above 0, stay flat for M.
"""
import sys

import numpy as np
import pandas as pd

from crypto_backtest import FEES, load, run
from monitor import write_json

opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
preds, start, save = opt("preds", "lgbm_w24_walk"), pd.Period(opt("start", "2025-01-01")[:7], "M"), opt("save", "nested")
ALLOCS = ["active", "active_cap2", "fixed5", "slices"]
ENTRIES = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
EXITS = [1.0, 0.5, 0.25, 0.0]

p = load(preds)
print(f"predictions {preds}: {p['time'].min()} .. {p['time'].max()}, {len(p)} coin-hours")
# Every setting once over the whole period: its monthly net returns and its daily equity, for both fee levels.
table = {}
for alloc in ALLOCS:
    for entry in ENTRIES:
        for ex in EXITS:
            for fname, fee in FEES.items():
                r = run(p, entry, fee, boot=10, alloc=alloc, exit_ratio=ex, funding=True)
                eq = pd.Series({pd.Timestamp(t): v for t, v in r["equity"]})
                table[(alloc, entry, ex, fname)] = {"monthly": {pd.Period(k, "M"): v for k, v in r["monthly"].items()},
                                                    "daily": eq.diff().fillna(eq.iloc[0])}
    print(f"  evaluated {alloc}", flush=True)
months = sorted(next(iter(table.values()))["monthly"])
maker, taker = list(FEES)

# The monthly choice, using only months before M.
choices, oos = [], {f: {} for f in FEES}
for m in [x for x in months if x >= start]:
    past = [x for x in months if x < m]
    score = {(a, e, x): np.mean([table[(a, e, x, maker)]["monthly"][q] for q in past])
             for a in ALLOCS for e in ENTRIES for x in EXITS}
    smooth = {(a, e, x): np.mean([score[(a, ENTRIES[j], x)] for j in range(max(i - 1, 0), min(i + 2, len(ENTRIES)))])
              for a in ALLOCS for i, e in enumerate(ENTRIES) for x in EXITS}
    best = max(smooth, key=smooth.get)
    trade = smooth[best] > 0
    for f in FEES:
        oos[f][m] = table[(*best, f)]["monthly"][m] if trade else 0.0
    choices.append({"month": str(m), "alloc": best[0] if trade else "flat", "entry": best[1], "exit_ratio": best[2],
                    "past_score": float(smooth[best]), "past_months": len(past),
                    "maker": oos[maker][m], "taker": oos[taker][m]})
    print(f"{m}  chose {best[0]:11s} entry {best[1]:.2f} exit x{best[2]:.2f} (past score {smooth[best]:+.2%})"
          f"{'' if trade else '  -> flat'}   this month: maker {oos[maker][m]:+.2%}  taker {oos[taker][m]:+.2%}")


def stitched(fname):
    """Daily out-of-sample equity: each month uses the daily gains of the setting chosen for that month."""
    parts = []
    for c in choices:
        m = pd.Period(c["month"], "M")
        if c["alloc"] == "flat":
            days = pd.date_range(m.start_time, m.end_time.floor("D"), freq="D")
            parts.append(pd.Series(0.0, index=days))
        else:
            d = table[(c["alloc"], c["entry"], c["exit_ratio"], fname)]["daily"]
            parts.append(d[(d.index >= m.start_time) & (d.index <= m.end_time)])
    return pd.concat(parts).cumsum()


results, rng = {}, np.random.default_rng(0)
print("\nOUT-OF-SAMPLE (settings re-chosen every month from earlier months only)")
for f in FEES:
    vals = np.array(list(oos[f].values()))
    boot = rng.choice(vals, (2000, len(vals))).mean(1)
    eq = stitched(f)
    results[f"out-of-sample {f}"] = {
        "monthly_mean": float(vals.mean()), "monthly_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "monthly": {str(k): float(v) for k, v in oos[f].items()}, "trades_per_day": None, "hit_on_trades": None,
        "net_per_trade": None, "max_drawdown": float((eq - eq.cummax()).min()), "total": float(eq.iloc[-1]),
        "equity": [[str(t.date()), float(v)] for t, v in eq.items()], "positive_months": int((vals > 0).sum())}
    r = results[f"out-of-sample {f}"]
    print(f"  {f}: mean month {r['monthly_mean']:+.2%} [{r['monthly_ci'][0]:+.2%}, {r['monthly_ci'][1]:+.2%}]  "
          f"total {r['total']:+.1%}  positive months {r['positive_months']}/{len(vals)}  worst drawdown {r['max_drawdown']:+.1%}")
write_json(f"runs/{save}.json", {"name": save, "val_preds": preds, "test_preds": preds, "alloc": "re-chosen monthly",
                                 "signal": "prob", "threshold": "re-chosen monthly", "results": results,
                                 "goal_monthly": 0.03, "nested": {"choices": choices, "candidates": len(ALLOCS) *
                                                                  len(ENTRIES) * len(EXITS)}})
print(f"saved runs/{save}.json  ->  http://localhost:8000/backtest.html?run={save}")

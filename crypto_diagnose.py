"""Diagnostics for the chosen trading setup on development data only (2025 + 2026 Jan-Aug); nothing here is tuned.

Usage: python crypto_diagnose.py [--preds=lgbm_plain,lgbm_plain_walk] [--entry=0.12] [--exit=0.25]
1. sizing and costs: allocation rules x funding payments, maker fees, per period
2. where the profit comes from: net contribution per coin and per month
3. neighbourhood: do nearby entry / exit settings also make money (worse period, active allocation)?
Writes runs/diagnose.json.
"""
import sys

import numpy as np
import pandas as pd

from crypto_backtest import FEES, PERIODS, load, run
from monitor import write_json

opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
preds, entry, exit_r = opt("preds", "lgbm_plain,lgbm_plain_walk"), float(opt("entry", 0.12)), float(opt("exit", 0.25))
fee = FEES["maker 0.02%"]
d_all = load(preds)
dev = {k: d_all[(d_all["time"] >= a) & (d_all["time"] < b)] for k, (a, b) in PERIODS.items()}
out = {"preds": preds, "entry": entry, "exit_ratio": exit_r}

print("1. sizing and costs (maker fees): mean monthly net per period | worst drawdown | coin-hours in position/day")
out["variants"] = []
for alloc in ["active", "active_cap2", "fixed5", "fixed3", "slices"]:
    for funding in (False, True):
        res = {k: run(d, entry, fee, boot=50, alloc=alloc, exit_ratio=exit_r, funding=funding) for k, d in dev.items()}
        row = {"alloc": alloc, "funding": funding, **{k: r["monthly_mean"] for k, r in res.items()},
               "max_dd": min(r["max_drawdown"] for r in res.values()),
               "per_day": float(np.mean([r["trades_per_day"] for r in res.values()]))}
        out["variants"].append(row)
        print(f"  {alloc:12s} funding {'on ' if funding else 'off'}  " + "  ".join(f"{k} {row[k]:+.2%}" for k in dev)
              + f"  | DD {row['max_dd']:+.1%} | {row['per_day']:.1f}/day")

print("\n2. where the profit comes from (active allocation, funding on, maker)")
res = {k: run(d, entry, fee, alloc="active", exit_ratio=exit_r, funding=True) for k, d in dev.items()}
by_coin = pd.DataFrame({k: r["by_coin"] for k, r in res.items()}).fillna(0.0)
by_coin["total"] = by_coin.sum(1)
by_coin = by_coin.sort_values("total", ascending=False)
print(by_coin.map(lambda v: f"{v:+.2%}").to_string())
top2 = by_coin["total"].head(2).sum() / by_coin["total"].sum()
print(f"  top 2 coins = {top2:.0%} of the total")
months = {m: v for r in res.values() for m, v in r["monthly"].items()}
print("  by month: " + "  ".join(f"{m} {v:+.1%}" for m, v in months.items()))
print(f"  positive months: {sum(v > 0 for v in months.values())} of {len(months)}")
out.update(by_coin=by_coin["total"].to_dict(), by_month=months, top2_share=float(top2))

print("\n3. neighbourhood (active allocation, funding on, maker): worse-period mean monthly net")
out["neighbourhood"] = []
entries, exits = [0.08, 0.10, 0.12, 0.15], [1.0, 0.5, 0.25, 0.0]
print("  entry | exit  " + "".join(f"{x:>9.2f}" for x in exits))
for t in entries:
    cells = []
    for x in exits:
        v = min(run(d, t, fee, boot=20, alloc="active", exit_ratio=x, funding=True)["monthly_mean"] for d in dev.values())
        cells.append(v)
        out["neighbourhood"].append({"entry": t, "exit_ratio": x, "worse_period_monthly": v})
    print(f"  {t:12.2f}  " + "".join(f"{v:+9.2%}" for v in cells))
write_json("runs/diagnose.json", out)
print("\nsaved runs/diagnose.json")

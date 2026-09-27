"""Daily cross-sectional ranking: each day at 00:00 UTC, LightGBM ranks the tradable coins by their expected return over
the next H days relative to the coin average; long the top K, short the bottom K (half the capital each side, so the
market's direction cancels). Retrained at the start of every month on all earlier data (H-day gap), so every
prediction is out of sample. Positions are the average of the last H days' portfolios (overlapping holdings).
Positions pay (long) or receive (short) each coin's perpetual funding rate.

Usage: python crypto_rank.py [--save=rank] [--universe=20] [--drop=ZEC,TAO] [--delay=1]
                             [--walk-from=2024-01-01] [--nested-from=2025-01] [--seeds=5]
  --universe=N: point-in-time coins (fetch_universe.py): at the start of every month, the N coins with the highest
      USDT volume over the previous 30 days, among coins with 90+ days of history and a live perpetual. Without it:
      the 20 coins in data/crypto/ (today's largest, so chosen with hindsight).
  --seeds=N: average N LightGBM models with different random row/input sampling (one model's picks vary a lot).
  --drop: leave these coins out entirely.
  --delay=N: trade at the close N hours after the decision (hourly closes; data/universe_1h/ with --universe).
Honest test: each month's (H, K) is chosen from earlier months only; flat if none was positive.
Also prints each coin's share of the out-of-sample profit.
"""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from monitor import write_json

opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
save, drop, delay, univ = opt("save", "rank"), [c for c in opt("drop", "").split(",") if c], int(opt("delay", 0)), \
    int(opt("universe", 0))
FEES = {"maker 0.02%": 0.0002, "taker 0.05%": 0.0005}
HS, KS = [1, 3, 7], [3, 5]
WALK_FROM, NESTED_FROM = pd.Timestamp(opt("walk-from", "2024-01-01")), pd.Period(opt("nested-from", "2025-01"), "M")
END, SEEDS = pd.Timestamp("2026-09-01"), int(opt("seeds", 5))
SRC, FSRC = ("data/universe", "data/universe_funding") if univ else ("data/crypto", "data/crypto_funding")

# Daily candles per coin (a day's row is known at the next 00:00 UTC).
close, qvol, execp, feats = {}, {}, {}, []
for path in sorted(glob.glob(f"{SRC}/*.csv.gz")):
    c = os.path.basename(path).split(".")[0]
    h = pd.read_csv(path, parse_dates=["time"]).set_index("time")
    if c in drop or h.empty:
        continue
    d = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last", "quote_volume": "sum",
                              "volume": "sum", "taker_buy_volume": "sum", "trades": "sum"})
    d = d[d["quote_volume"] > 0]
    if len(d) < 2:
        continue
    d = d.reindex(pd.date_range(d.index[0], d.index[-1], freq="D"))   # calendar days: a trading gap stays a gap
    close[c], qvol[c] = d["close"], d["quote_volume"]
    r = np.log(d["close"]).diff()
    f = pd.DataFrame(index=d.index)
    for n in (1, 3, 7, 14, 30, 60):
        f[f"r{n}"] = np.log(d["close"] / d["close"].shift(n))
    f["vol7"], f["vol30"] = r.rolling(7).std(), r.rolling(30).std()
    for n in (1, 7, 30):
        f[f"rs{n}"] = f[f"r{n}"] / (f["vol30"] * np.sqrt(n) + 1e-9)
    f["vr"] = np.log(d["quote_volume"].rolling(7).mean() / d["quote_volume"].rolling(60).mean())
    f["tb1"] = d["taker_buy_volume"] / d["volume"].replace(0, np.nan)
    f["tb7"] = d["taker_buy_volume"].rolling(7).sum() / d["volume"].rolling(7).sum()
    f["clv"] = (d["close"] - d["low"]) / (d["high"] - d["low"] + 1e-12)
    hi30, lo30 = d["high"].rolling(30).max(), d["low"].rolling(30).min()
    f["pos30"] = (d["close"] - lo30) / (hi30 - lo30 + 1e-12)
    f["trades7"] = np.log(d["trades"].rolling(7).mean() / d["trades"].rolling(60).mean())
    f["coin"] = c
    feats.append(f.dropna(subset=["r1"]).reset_index(names="time"))
f = pd.concat(feats, ignore_index=True)
f["time"] = f["time"] + pd.Timedelta(days=1)            # decision time: the day's candle has closed
px, qv = pd.DataFrame(close), pd.DataFrame(qvol)
px.index = qv.index = px.index + pd.Timedelta(days=1)     # known at decision time
if delay:                                                  # trade at the close `delay` hours after the decision
    for c in close:
        path = f"{'data/universe_1h' if univ else SRC}/{c}.csv.gz"
        ex = pd.read_csv(path, parse_dates=["time"]).set_index("time")["close"] if os.path.exists(path) else []
        if len(ex):
            ex.index = ex.index + pd.Timedelta(hours=1)    # close time of each hourly candle
            ex = ex[ex.index.hour == delay]
            ex.index = ex.index.floor("D")                  # trade price for the decision at that day's 00:00
            execp[c] = ex[~ex.index.duplicated(keep="last")]
    ep = pd.DataFrame(execp).reindex(index=px.index, columns=px.columns)
    ret1 = np.log(ep.shift(-1) / ep)                       # next day's move at the trade price, per coin
else:
    ret1 = np.log(px.shift(-1) / px)

ft = pd.DataFrame({os.path.basename(p)[:-7]: pd.read_csv(p, parse_dates=["time"]).drop_duplicates("time")
                   .set_index("time")["funding_rate"] for p in glob.glob(f"{FSRC}/*.csv.gz")})
ft = ft.reindex(columns=px.columns)
fund = ft.groupby((ft.index - pd.Timedelta(1, "ns")).floor("D")).sum().reindex(px.index).fillna(0.0)  # (t, t + 1d]

# Tradable coins per day.
if univ:
    live = ft.notna().groupby(ft.index.floor("D")).sum().reindex(px.index, fill_value=0)
    live = live.rolling(30, min_periods=1).sum().shift(1) > 0            # a funding payment in the last 30 days
    hist = px.notna().cumsum()
    mem = pd.DataFrame(False, index=px.index, columns=px.columns)
    for m in pd.period_range(px.index[0], END, freq="M"):
        a = m.start_time
        if a not in px.index:
            continue
        vol30 = qv[(qv.index > a - pd.Timedelta(days=30)) & (qv.index <= a)].mean()
        top = vol30[(hist.loc[a] >= 90) & live.loc[a] & vol30.notna()].nlargest(univ).index
        mem.loc[(mem.index >= a) & (mem.index < (m + 1).start_time), top] = True
else:
    mem = px.notna()
mem = mem.rename_axis("time").rename_axis("coin", axis=1).stack().rename("in").reset_index()
f = f.merge(mem, on=["time", "coin"])
f = f[f["in"]].drop(columns="in")
INPUTS = [k for k in f.columns if k not in ("time", "coin")]
# Cross-sectional versions: each input relative to that day's coin average, and its rank.
for k in INPUTS[:]:
    g = f.groupby("time")[k]
    f[k + "_x"] = f[k] - g.transform("mean")
    f[k + "_rk"] = g.rank(pct=True)
INPUTS = [k for k in f.columns if k not in ("time", "coin")]
f = f[f["time"] >= "2019-01-01"]
if univ:
    first = f.groupby(f["time"].dt.to_period("M"))["coin"].unique()
    print(f"universe: {f['coin'].nunique()} different coins over time; e.g. " +
          "  ".join(f"{m}: {' '.join(sorted(first[pd.Period(m, 'M')]))}" for m in ("2021-01", "2023-01", "2026-08")
                    if pd.Period(m, "M") in first.index), flush=True)


def walk(H):
    """Monthly refit from WALK_FROM; returns a (day x coin) table of out-of-sample scores."""
    fwd = np.log(px.shift(-H) / px).rename_axis("time").rename_axis("coin", axis=1).stack().rename("y").reset_index()
    df = f.merge(fwd, on=["time", "coin"], how="left")
    df["y"] = df["y"] - df.groupby("time")["y"].transform("mean")        # relative to the tradable coins' average
    out = []
    for m in pd.period_range(WALK_FROM, END, freq="M"):
        a, b = m.start_time, (m + 1).start_time
        tr = df[(df["time"] < a - pd.Timedelta(days=H)) & df["y"].notna()]
        te = df[(df["time"] >= a) & (df["time"] < b)]
        if te.empty:
            continue
        s = 0
        for seed in range(SEEDS):
            mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=300,
                                    subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10,
                                    random_state=seed, verbose=-1)
            s = s + mdl.fit(tr[INPUTS], tr["y"]).predict(te[INPUTS]) / SEEDS
        out.append(te[["time", "coin"]].assign(s=s))
    return pd.concat(out).pivot(index="time", columns="coin", values="s")


def book(score, H, K):
    """Daily net returns (per fee level) of the long-short book."""
    w = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    rk = score.rank(axis=1)
    n = score.notna().sum(1)
    w[rk.le(K, axis=0)] = -0.5 / K
    w[rk.gt(n - K, axis=0)] = 0.5 / K
    w = w.rolling(H, min_periods=1).mean()                 # overlapping H-day holdings
    dw = w.diff().abs().fillna(w.abs())
    simple = np.expm1(ret1.reindex(w.index)[w.columns]).fillna(0)                # a position earns weight x simple move
    per_coin = w * simple - w * fund.reindex(w.index)[w.columns].fillna(0)
    gross, turn = per_coin.sum(1), dw.sum(1)
    return {fn: gross - fee * turn for fn, fee in FEES.items()}, turn.mean(), per_coin - FEES["maker 0.02%"] * dw


scores = {H: walk(H) for H in HS}
daily, coin_pnl = {}, {}
for H in HS:
    for K in KS:
        daily[(H, K)], turn, coin_pnl[(H, K)] = book(scores[H], H, K)
        m = daily[(H, K)]["maker 0.02%"].groupby(daily[(H, K)]["maker 0.02%"].index.to_period("M")).sum()
        t = daily[(H, K)]["taker 0.05%"].groupby(daily[(H, K)]["taker 0.05%"].index.to_period("M")).sum()
        m, t = m[m.index < pd.Period("2026-09", "M")], t[t.index < pd.Period("2026-09", "M")]
        print(f"H={H} K={K}  turnover/day {turn:.2f}  " + "  ".join(f"{y} {m[m.index.year == y].mean():+.2%}"
                                                                   for y in sorted(set(m.index.year))) +
              f"  (taker all {t.mean():+.2%})  positive {int((m > 0).sum())}/{len(m)}", flush=True)

# Honest test: choose (H, K) each month from earlier months only.
monthly = {k: {fn: v[fn].groupby(v[fn].index.to_period("M")).sum() for fn in FEES} for k, v in daily.items()}
months = [m for m in monthly[(1, 3)]["maker 0.02%"].index if NESTED_FROM <= m < pd.Period("2026-09", "M")]
oos, choices = {fn: {} for fn in FEES}, []
for m in months:
    past = {k: v["maker 0.02%"][v["maker 0.02%"].index < m].mean() for k, v in monthly.items()}
    best = max(past, key=past.get)
    for fn in FEES:
        oos[fn][str(m)] = float(monthly[best][fn][m]) if past[best] > 0 else 0.0
    choices.append({"month": str(m), "alloc": f"H={best[0]} K={best[1]}" if past[best] > 0 else "flat",
                    "entry": best[0], "exit_ratio": best[1], "past_score": float(past[best]),
                    "maker": oos["maker 0.02%"][str(m)], "taker": oos["taker 0.05%"][str(m)]})
    print(f"{m}  chose H={best[0]} K={best[1]} (past {past[best]:+.2%})  this month: maker "
          f"{oos['maker 0.02%'][str(m)]:+.2%}  taker {oos['taker 0.05%'][str(m)]:+.2%}")

# Each coin's share of the out-of-sample profit (maker fees, funding included), per month's chosen setting.
parts = [coin_pnl[(c["entry"], c["exit_ratio"])][lambda x: x.index.to_period("M") == pd.Period(c["month"], "M")]
         for c in choices if c["alloc"] != "flat"]
by_coin = pd.concat(parts).sum().sort_values(ascending=False) if parts else pd.Series(dtype=float)
by_coin = by_coin[by_coin != 0]
print("\nprofit by coin, out of sample (sum of monthly returns, maker fees): best 10 / worst 5")
print("  " + "  ".join(f"{k} {v:+.1%}" for k, v in by_coin.head(10).items()) + "   ...   " +
      "  ".join(f"{k} {v:+.1%}" for k, v in by_coin.tail(5).items()))
results, rng = {}, np.random.default_rng(0)
print(f"\nOUT-OF-SAMPLE (H, K re-chosen every month from earlier months only), {months[0]} .. {months[-1]}")
for fn in FEES:
    vals = np.array(list(oos[fn].values()))
    eq = np.cumsum(vals)
    ci = np.percentile(rng.choice(vals, (2000, len(vals))).mean(1), [2.5, 97.5])
    results[f"out-of-sample {fn}"] = {"monthly_mean": float(vals.mean()), "monthly_ci": [float(x) for x in ci],
                                      "monthly": oos[fn], "total": float(eq[-1]), "positive_months": int((vals > 0).sum()),
                                      "max_drawdown": float((eq - np.maximum.accumulate(eq)).min()),
                                      "equity": [[k + "-28", float(v)] for k, v in zip(oos[fn], eq)],
                                      "trades_per_day": None, "hit_on_trades": None, "net_per_trade": None}
    by_year = pd.Series(oos[fn]).groupby(lambda k: k[:4]).mean()
    print(f"  {fn}: mean month {vals.mean():+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}]  total {eq[-1]:+.1%}  "
          f"positive {int((vals > 0).sum())}/{len(vals)}  worst drawdown {results[f'out-of-sample {fn}']['max_drawdown']:+.1%}"
          f"\n      by year: " + "  ".join(f"{y} {v:+.2%}" for y, v in by_year.items()))
write_json(f"runs/{save}.json", {"name": save, "val_preds": "daily ranking", "test_preds": "daily ranking",
                                 "alloc": "long-short, re-chosen monthly", "signal": "rank", "threshold": "re-chosen monthly",
                                 "results": results, "goal_monthly": 0.02, "nested": {"choices": choices,
                                                                                      "candidates": len(HS) * len(KS)}})
print(f"saved runs/{save}.json  ->  http://localhost:8000/backtest.html?run={save}")

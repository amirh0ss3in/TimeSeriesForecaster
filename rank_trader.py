"""Frozen daily ranking system, for paper trading. Settings are the ones the honest monthly test chose in every month
since mid-2025 (crypto_rank.py): each month's coins are the 20 with the highest USDT volume over the previous 30 days
(90+ days of history, live perpetual); LightGBM (5 seeds) is retrained at the start of every month; each day at 00:00
UTC it goes long the top 3 and short the bottom 3 (1/6 of capital each), filled at the 01:00 UTC close and held one
day. Fees (maker 0.02% or taker 0.05%) are paid on every weight change, and funding on open positions.

Usage:
  python rank_trader.py update             latest daily candles, hourly closes and funding rates from Binance's public
                                           API into data/universe*/ (a coin not on disk is added once it is in the top
                                           40 by 24-hour volume)
  python rank_trader.py step [--now=2026-09-28] [--from=2026-09-28] [--log=rank_paper]
                                           the decisions for every day not logged yet, up to that day (default: today,
                                           UTC; run it after `update`); --from = first day of a new log. Books earlier
                                           days once their next 01:00 price is known
  python rank_trader.py check              replays Jul 2026 through this code and compares it with the backtest
  python rank_trader.py replay --from=2026-09-01 --to=2026-09-26 [--log=rank_replay]
                                           the same code path over past days
Models: checkpoints/rank_trader/<YYYY-MM>_seed<S>.txt, trained once per month and then reused.
Log: runs/<log>.json, viewable at backtest.html?run=<log>.
"""
import csv
import datetime as dt
import glob
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto_rank import FEES, fit, load, pnl, positions, targets
from fetch_universe import NOT_COINS
from monitor import write_json

UNIVERSE, H, K, SEEDS, DELAY = 20, 1, 3, 5, 1
MODEL_DIR = "checkpoints/rank_trader"
# Public market-data addresses; GitHub's (US) servers are refused by these two, so the workflow points them at
# data-api.binance.vision and www.binance.com, which serve the same data.
SPOT = os.environ.get("BINANCE_SPOT", "https://api.binance.com")
FUT = os.environ.get("BINANCE_FUTURES", "https://fapi.binance.com")
KLINE_HEADER = ["time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]
opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


# ---------------------------------------------------------------------------------------------------- live data feed
def api(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404) or attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)


stamp = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")


def merge(path, header, rows):
    """Append rows for times not on disk yet (rows already on disk, e.g. from the archive, are kept as they are)."""
    old = []
    if os.path.exists(path):
        with gzip.open(path, "rt", newline="") as fh:
            old = list(csv.reader(fh))[1:]
    have = {r[0] for r in old}
    new = [r for r in rows if r[0] not in have]
    if new or not os.path.exists(path):
        with gzip.open(path, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(sorted(old + new, key=lambda r: r[0]))
    return old[-1][0] if old else None, len(new)


def update():
    now = time.time() * 1000
    spot = {s["baseAsset"]: s["symbol"] for s in api(f"{SPOT}/api/v3/exchangeInfo")["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    perps = {s["symbol"] for s in api(f"{FUT}/fapi/v1/exchangeInfo")["symbols"]
             if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING" and s["quoteAsset"] == "USDT"}
    perp = {b: next((p + b + "USDT" for p in ("", "1000", "1000000", "1M") if p + b + "USDT" in perps), None)
            for b in spot}
    vol = {t["symbol"]: float(t["quoteVolume"]) for t in api(f"{SPOT}/api/v3/ticker/24hr")}
    ok = [b for b in spot if perp[b] and b.isascii() and b.isalnum() and b not in NOT_COINS]
    on_disk = {os.path.basename(p)[:-7] for p in glob.glob("data/universe/*.csv.gz")}
    new = [b for b in sorted(ok, key=lambda b: -vol.get(spot[b], 0.0))[:40] if b not in on_disk]
    coins = sorted((on_disk & set(spot)) | set(new))

    def refresh(b):
        k = api(f"{SPOT}/api/v3/klines?symbol={spot[b]}&interval=1d&limit={1000 if b in new else 60}")
        last, n = merge(f"data/universe/{b}.csv.gz", KLINE_HEADER,
                        [[stamp(x[0]), *x[1:6], x[7], x[8], x[9]] for x in k if x[6] < now])
        k = api(f"{SPOT}/api/v3/klines?symbol={spot[b]}&interval=1h&limit=1000")
        merge(f"data/universe_1h/{b}.csv.gz", ["time", "close"], [[stamp(x[0]), x[4]] for x in k if x[6] < now])
        if perp.get(b):
            fr = api(f"{FUT}/fapi/v1/fundingRate?symbol={perp[b]}&limit=1000")
            merge(f"data/universe_funding/{b}.csv.gz", ["time", "funding_rate"],
                  [[stamp(x["fundingTime"]), x["fundingRate"]] for x in fr])
        return b, n

    with ThreadPoolExecutor(8) as pool:
        added = dict(pool.map(refresh, coins))
    print(f"updated {len(coins)} coins ({len(new)} new: {' '.join(new) or '-'}); "
          f"new days: {sum(added.values())}; latest daily candle: "
          f"{pd.read_csv('data/universe/BTC.csv.gz')['time'].iloc[-1]}")


# ---------------------------------------------------------------------------------------------------- decisions
def models_for(month, df, inputs):
    """The month's frozen models: trained once (on everything before the month) and then read from disk."""
    paths = [f"{MODEL_DIR}/{month}_seed{s}.txt" for s in range(SEEDS)]
    if not all(os.path.exists(p) for p in paths):
        os.makedirs(MODEL_DIR, exist_ok=True)
        for p, md in zip(paths, fit(df, month.start_time, H, inputs, SEEDS)):
            md.booster_.save_model(p)
    return [lgb.Booster(model_file=p) for p in paths]


def decide(data, days):
    """Scores and target weights for these decision days, each from what was known at that day's 00:00 UTC."""
    f, inputs, px, _, _ = data
    df = targets(f, px, H)
    parts = []
    for m in sorted({d.to_period("M") for d in days}):
        te = df[df["time"].isin([d for d in days if d.to_period("M") == m])]
        if not te.empty:
            parts.append(te[["time", "coin"]].assign(s=sum(b.predict(te[inputs]) for b in models_for(m, df, inputs))
                                                     / SEEDS))
    missing = sorted(set(days) - set(pd.concat(parts)["time"])) if parts else days
    if missing:
        sys.exit(f"no candles for the decisions of {', '.join(str(d.date()) for d in missing)}: "
                 f"run `python rank_trader.py update` first")
    score = pd.concat(parts).pivot(index="time", columns="coin", values="s")
    return score, positions(score, H, K)


def record(log, score, w):
    for d in w.index:
        s, wd = score.loc[d].dropna(), w.loc[d]
        log["days"][str(d.date())] = {"coins": sorted(s.index), "weights": {c: float(v) for c, v in wd.items() if v},
                                      "scores": {c: round(float(v), 6) for c, v in s.sort_values().items()}}


def book(log, data):
    """Book every logged day whose next 01:00 price is known, and add a backtest.html-style summary."""
    _, _, _, ret1, fund = data
    days = pd.to_datetime(sorted(log["days"]))
    w = pd.DataFrame([log["days"][str(d.date())]["weights"] for d in days], index=days).fillna(0.0)
    w = w.reindex(pd.date_range(days[0], days[-1], freq="D")).ffill()             # a missed day keeps its positions
    known = ret1.index[ret1.notna().any(axis=1)].max()                              # next 01:00 price is in
    net, turn, _ = pnl(w[w.index <= known], ret1, fund)
    start = pd.Timestamp(log["start"])
    booked = {fn: v[v.index >= start] for fn, v in net.items()}
    log["daily"] = [{"date": str(d.date()), "coins": len(log["days"].get(str(d.date()), {}).get("coins", [])),
                     "long": sorted(c for c, v in w.loc[d].items() if v > 0),
                     "short": sorted(c for c, v in w.loc[d].items() if v < 0),
                     **{k: (float(booked[fn][d]) if d in booked[fn].index else None)
                        for k, fn in (("maker", "maker 0.02%"), ("taker", "taker 0.05%"))}}
                    for d in w.index if d >= start]
    rng, log["results"] = np.random.default_rng(0), {}
    for fn, v in booked.items():
        if v.empty:
            continue
        mon = v.groupby(v.index.to_period("M")).sum()
        eq = v.cumsum()
        ci = np.percentile(rng.choice(mon.to_numpy(), (2000, len(mon))).mean(1), [2.5, 97.5])
        log["results"][f"paper {fn}"] = {
            "monthly_mean": float(mon.mean()), "monthly_ci": [float(x) for x in ci],
            "monthly": {str(k): float(x) for k, x in mon.items()}, "total": float(eq.iloc[-1]),
            "max_drawdown": float((eq - eq.cummax()).min()), "positive_months": int((mon > 0).sum()),
            "equity": [[str(d.date()), float(x)] for d, x in eq.items()],
            "trades_per_day": float(turn[turn.index >= start].mean()), "hit_on_trades": None, "net_per_trade": None}
    write_json(f"runs/{log['name']}.json", log)
    return booked


def new_log(name, start):
    return {"name": name, "val_preds": "frozen daily ranking", "test_preds": "live" if name == "rank_paper" else "replay",
            "alloc": f"long top {K} / short bottom {K} of the point-in-time top {UNIVERSE}, {0.5 / K:.3f} each",
            "signal": f"LightGBM rank ({SEEDS} seeds, refit monthly)", "threshold": "—", "goal_monthly": 0.02,
            "start": str(start.date()), "days": {}}


def open_log(name, start):
    path = f"runs/{name}.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return new_log(name, start)


def replay(a, b, name):
    """Decide every day from a to b (plus the day before, so a's fees count the change from it) and book them."""
    data = load(UNIVERSE, delay=DELAY, end=b)
    score, w = decide(data, list(pd.date_range(a - pd.Timedelta(days=1), b, freq="D")))
    log = new_log(name, a)
    record(log, score, w)
    return book(log, data)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "update":
        update()
    elif cmd == "step":
        now = pd.Timestamp(opt("now", str(dt.datetime.now(dt.UTC).date())))
        name = opt("log", "rank_paper")
        log = open_log(name, pd.Timestamp(opt("from", str(now.date()))))
        # Every day not logged yet, up to now (catches up missed days; logged days are never recomputed).
        first = pd.Timestamp(max(log["days"])) + pd.Timedelta(days=1) if log["days"] else pd.Timestamp(log["start"])
        days = list(pd.date_range(first, now, freq="D"))
        data = load(UNIVERSE, delay=DELAY, end=now)
        if days:
            record(log, *decide(data, days))
        booked = book(log, data)
        d = log["days"][str(now.date())]
        print(f"{now.date()}: long {' '.join(c for c, v in d['weights'].items() if v > 0)}   short "
              f"{' '.join(c for c, v in d['weights'].items() if v < 0)}   ({0.5 / K:.1%} of capital each, "
              f"at the 01:00 UTC close)")
        print(f"coins today: {' '.join(d['coins'])}")
        for fn, v in booked.items():
            if len(v):
                print(f"booked {fn}: {len(v)} days, last {v.index[-1].date()} {v.iloc[-1]:+.2%}, total {v.sum():+.2%}")
        print(f"log: runs/{name}.json  ->  http://localhost:8000/backtest.html?run={name}")
    elif cmd == "check":
        booked = replay(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"), "rank_check")
        with open("runs/rank_pit20_d1.json", encoding="utf-8") as fh:
            ref = json.load(fh)["results"]
        for fn, v in booked.items():
            mine, theirs = v.sum(), ref[f"out-of-sample {fn}"]["monthly"]["2026-07"]
            print(f"Jul 2026 {fn}: this code {mine:+.6%}   backtest {theirs:+.6%}   "
                  f"{'MATCH' if abs(mine - theirs) < 1e-9 else 'DIFFERENT'}")
    elif cmd == "replay":
        name = opt("log", "rank_replay")
        booked = replay(pd.Timestamp(opt("from", "2026-09-01")), pd.Timestamp(opt("to", "2026-09-26")), name)
        for fn, v in booked.items():
            print(f"{fn}: {len(v)} days booked, total {v.sum():+.2%}, per 30 days {v.mean() * 30:+.2%}, "
                  f"positive days {(v > 0).sum()}/{len(v)}")
        print(f"log: runs/{name}.json  ->  http://localhost:8000/backtest.html?run={name}")
    else:
        print(__doc__)

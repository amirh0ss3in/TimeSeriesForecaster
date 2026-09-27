"""Frozen hourly trading system: LightGBM next-hour direction model + entry/exit rule, logged for paper trading.

Usage:
  python trader.py step [--now=2026-09-15T21:00] [--log=paper_log]
                                                    one hourly decision on the data on disk; appends to the paper log
  python trader.py replay --from=2026-09-01 --to=2026-09-27 [--log=paper_replay]
                                                    replay past hours through the same code path (checks, demo log)
  python trader.py check                            reproduce the backtest's saved September predictions

Frozen settings (chosen on development data, see SETUP.md): inputs = crypto_features.INPUTS, LightGBM as below with
TREES trees, retrained at the start of every month on all rows before it (25-hour purge); entry edge 0.12, exit at
0.25 x entry. Several capital-allocation variants are logged side by side (VARIANTS). Every decision at hour t uses
only candles closed by t + 1h; its outcome is booked when hour t + 1 closes. Paper log: runs/<log>.json.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto_backtest import FEES, funding_table, positions, weights
from crypto_features import INPUTS, build, features
from monitor import write_json

PARAMS = dict(learning_rate=0.02, num_leaves=15, min_child_samples=2000, subsample=0.5, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=10, verbose=-1)
TREES = 278                                   # early stopping on Oct-Dec 2024 (253 trees) x 1.1, as in crypto_probe
ENTRY, EXIT_RATIO = 0.12, 0.25
VARIANTS = ["active", "active_cap2", "fixed5", "slices"]
PURGE = pd.Timedelta(hours=25)
MODEL_DIR = "checkpoints/trader"


def model_for(month):
    """The LightGBM model used during `month` (a Timestamp at the month start): trained on every row before it."""
    path = f"{MODEL_DIR}/model_{month:%Y-%m}.txt"
    if os.path.exists(path):
        return lgb.Booster(model_file=path)
    df = features()
    base = df[(df["time"] < month - PURGE) & (df["y1"] != 0)].dropna(subset=INPUTS + ["y1"])
    m = lgb.LGBMClassifier(**PARAMS, n_estimators=TREES).fit(base[INPUTS], base["y1"] > 0)
    os.makedirs(MODEL_DIR, exist_ok=True)
    m.booster_.save_model(path)
    print(f"trained model for {month:%Y-%m} on {len(base)} rows")
    return m.booster_


def predict(rows):
    """P(up) for feature rows (any hours), each with the model of its own month."""
    out = pd.Series(np.nan, index=rows.index)
    for month, g in rows.groupby(rows["time"].dt.to_period("M").dt.start_time):
        out[g.index] = model_for(month).predict(g[INPUTS])
    return out


def new_log(name):
    return {"name": name, "settings": {"entry": ENTRY, "exit_ratio": EXIT_RATIO, "trees": TREES,
                                       "variants": VARIANTS, "fees": FEES},
            "hours": [], "variants": {v: {"equity": {f: 0.0 for f in FEES}, "weights": {}, "pos": {}} for v in VARIANTS}}


def book(log, t, prob, realized, fund):
    """One hourly step for every variant: book the hour that just closed, then set the new positions.
    prob: P(up) per coin decided at hour t; realized: each coin's move over the hour that just closed (the one the
    previous decision earned); fund: funding rate per coin stamped at the end of that hour (0 if none)."""
    edge = (prob - 0.5).to_frame().T
    entry = {"time": str(t), "prob": prob.round(4).to_dict(), "variants": {}}
    for v in VARIANTS:
        st = log["variants"][v]
        prev_pos = pd.Series(st["pos"], dtype=float).reindex(edge.columns).fillna(0.0)
        prev_w = pd.Series(st["weights"], dtype=float).reindex(edge.columns).fillna(0.0)
        # positions(): the same entry/exit rule as the backtest, continued from the previous hour's positions
        pos = positions(edge, ENTRY, ENTRY * EXIT_RATIO, start=prev_pos.to_numpy())
        pos, w = weights(pos, edge, v)
        w = w.iloc[0]
        gross = float((prev_w * realized.reindex(prev_w.index).fillna(0.0)).sum())
        fnd = float((prev_w * fund.reindex(prev_w.index).fillna(0.0)).sum())
        turnover = float((w - prev_w).abs().sum())
        for f, fee in FEES.items():
            st["equity"][f] += gross - fnd - turnover * fee
        st["pos"], st["weights"] = pos.iloc[0].to_dict(), w.to_dict()
        entry["variants"][v] = {"gross": gross, "funding": fnd, "turnover": turnover,
                                "equity": dict(st["equity"]), "positions": {c: p for c, p in st["pos"].items() if p}}
    log["hours"].append(entry)


def replay(start, end, log_name):
    """Replay past hours through the same decision and booking code (features from the leak-tested full build,
    which equals building at each hour, see leak_check.py)."""
    df = features()
    rows = df[(df["time"] >= start) & (df["time"] < end)].dropna(subset=INPUTS)
    prob = predict(rows)
    p = pd.DataFrame({"coin": rows["coin"], "time": rows["time"], "prob": prob, "r1": rows["r1"]})
    P = p.pivot(index="time", columns="coin", values="prob")
    R = p.pivot(index="time", columns="coin", values="r1").reindex_like(P)
    F = funding_table().reindex(index=P.index + pd.Timedelta(hours=1), columns=P.columns).fillna(0.0)
    log = new_log(log_name)
    prev_r = pd.Series(0.0, index=P.columns)
    for i, t in enumerate(P.index):
        # realized move of the hour that just closed = r1 of the previous decision row (its next candle)
        realized = R.iloc[i - 1] if i else prev_r
        book(log, t, P.loc[t].dropna(), realized, F.iloc[i] if i else prev_r)
    write_json(f"runs/{log_name}.json", log)
    for v in VARIANTS:
        print(f"{v:12s} " + "  ".join(f"{f}: {e:+.2%}" for f, e in log["variants"][v]["equity"].items()))
    return log


def step(now, log_name="paper_log"):
    """Paper trading: the decision for hour `now` from data on disk (rebuilt up to now + 1h)."""
    path = f"runs/{log_name}.json"
    log = __import__("json").load(open(path)) if os.path.exists(path) else new_log(log_name)
    df = build(cutoff=now + pd.Timedelta(hours=1))         # only candles closed by the decision time
    rows = df[df["time"] == now].dropna(subset=INPUTS)
    prob = pd.Series(predict(rows).to_numpy(), index=rows["coin"])
    # candle `now` just closed; its move is the previous row's r1 (the move that row's decision earned)
    realized = df[df["time"] == now - pd.Timedelta(hours=1)].set_index("coin")["r1"]
    ft = funding_table()
    fund = ft.loc[now + pd.Timedelta(hours=1)] if now + pd.Timedelta(hours=1) in ft.index else pd.Series(dtype=float)
    book(log, now, prob, realized.fillna(0.0), fund)
    write_json(path, log)
    print(f"{now}: " + ", ".join(f"{c} {p:.3f}" for c, p in prob.sort_values().items()))


if __name__ == "__main__":
    opt = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "step":
        step(pd.Timestamp(opt("now", str(pd.Timestamp.utcnow().floor("h").tz_localize(None) - pd.Timedelta(hours=1)))),
             opt("log", "paper_log"))
    elif cmd == "replay":
        replay(pd.Timestamp(opt("from", "2026-09-01")), pd.Timestamp(opt("to", "2026-09-27")), opt("log", "paper_replay"))
    elif cmd == "check":
        saved = np.load("checkpoints/preds_lgbm_plain_walk.npz", allow_pickle=True)
        s = pd.DataFrame({"coin": saved["coin"], "time": pd.to_datetime(saved["time"]), "saved": saved["prob"]})
        s = s[s["time"] >= "2026-09-01"]
        df = features()
        rows = df[df["time"] >= "2026-09-01"].dropna(subset=INPUTS)
        mine = pd.DataFrame({"coin": rows["coin"], "time": rows["time"], "mine": predict(rows).to_numpy()})
        m = s.merge(mine, on=["coin", "time"])
        print(f"September: {len(m)} coin-hours, max |difference| vs the backtest's saved predictions: "
              f"{(m['saved'] - m['mine']).abs().max():.2e}")

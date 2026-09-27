"""Where is the direction signal? Fast models on the hourly feature table, scored on 2025 validation.

Usage: python crypto_probe.py [--target=y1|y24] [--features=a,b,c] [--coins=BTC,ETH] [--obj=binary] [--test]
                              [--seeds=N] [--refit] [--perp] [--wide] [--weight=move] [--walk]
  --weight=move: each training hour counts in proportion to the size of its move (capped at 3 sigma).
  --walk: refit monthly from --walk-from (default 2026-01-01); months from HOLDOUT on are predicted and saved but
  never scored or printed.
                              [--save=NAME]   (writes runs/NAME.json for direction.html)

Splits by time with a 25-hour purge between them, so no target overlaps the next split:
  fit   < 2024-10-01 (early stopping on Oct-Dec 2024), validation = 2025, test = 2026 (only with --test).
Scores: direction hit rate (next move's sign, zero moves excluded), hit rate of the most confident 20%, and the
information coefficient (correlation of prediction and outcome). 95% ranges come from resampling whole days, since
coins move together within an hour.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from crypto_eval import direction_stats, line
from crypto_features import INPUTS, PERP_INPUTS, features
from monitor import write_json

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
target = arg("target", "y1")
inputs = arg("features", ",".join(INPUTS)).split(",")
coins = arg("coins", "")
use_test = "--test" in sys.argv
ES, VAL, TEST = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")
HOLDOUT = pd.Timestamp("2026-09-01")                # final holdout: never scored here
purge = pd.Timedelta(hours=25)

df = features()
if "--wide" in sys.argv:       # every coin's latest move and candle shape as inputs for every row (NaN before listing)
    wide = df.pivot(index="time", columns="coin", values=["z0", "clv0"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    df = df.merge(wide, left_on="time", right_index=True, how="left")
    inputs = inputs + list(wide.columns)
if "--perp" in sys.argv:      # perpetual-futures inputs (NaN before a coin's perp existed; LightGBM handles it)
    inputs = inputs + PERP_INPUTS
if coins:
    df = df[df["coin"].isin(coins.split(","))]
df = df.dropna(subset=[c for c in inputs if c in INPUTS] + [target])   # wide and perp columns may be NaN
fit = df[df["time"] < ES - purge]
stop = df[(df["time"] >= ES) & (df["time"] < VAL - purge)]
val = df[(df["time"] >= VAL) & (df["time"] < TEST - purge)]
test = df[(df["time"] >= TEST) & (df["time"] < HOLDOUT - purge)]
WALK_FROM = pd.Timestamp(arg("walk-from", str(TEST.date())))
walk_rows = df[df["time"] >= WALK_FROM]             # includes the holdout, for saved walk-forward predictions only
clip = lambda y: y.clip(-5, 5)                       # heavy tails: fit on clipped targets, score on raw ones


def score(d, pred, name, prob=None, boot=300):
    s = direction_stats(d["time"].to_numpy(), d[target].to_numpy(), pred, prob, boot=boot)
    print(line(name, s))
    return s


t0 = time.time()
print(f"target {target}  fit {len(fit)}  stop {len(stop)}  val {len(val)}  inputs {len(inputs)}")
# Simple rules first: they show whether plain momentum or reversal is in the data at all.
rules = {f"follow {col}": score(val, val[col], f"rule: follow {col}")
         for col in [c for c in ("z0", "mkt0", "clv0", "mkt_clv", "imb0", "mkt_imb") if c in inputs]}

params = dict(n_estimators=4000, learning_rate=0.02, num_leaves=int(arg("leaves", 15)),
              min_child_samples=int(arg("min-child", 2000)), subsample=0.5, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=10, verbose=-1)
binary = arg("obj", "regression") == "binary"          # binary: predict P(up), trained on the sign only
half_life = float(arg("half-life", 0))                  # years; recency weighting of training rows (0 = off)
age = (ES - fit["time"]).dt.days / 365.25
weight = 0.5 ** (age / half_life) if half_life else None
if arg("weight", "") == "move":                     # profit-aware: big moves matter more than small ones
    weight = (weight if weight is not None else 1.0) * fit[target].abs().clip(upper=3)
if binary:
    model = lgb.LGBMClassifier(**params)
    keep_f, keep_s = fit[target] != 0, stop[target] != 0
    model.fit(fit[inputs][keep_f], fit[target][keep_f] > 0, eval_set=[(stop[inputs][keep_s], stop[target][keep_s] > 0)],
              sample_weight=None if weight is None else weight[keep_f],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    model.predict = (lambda m: lambda X: m.predict_proba(X)[:, 1] - 0.5)(model)   # centred: sign = direction
    # --seeds=N: average N models with different random seeds; --refit: after early stopping picks the number of
    # trees, retrain on fit + Oct-Dec 2024 (all data before 2025) with that many trees.
    n_seeds, refit = int(arg("seeds", 1)), "--refit" in sys.argv
    if n_seeds > 1 or refit:
        base = pd.concat([fit, stop]) if refit else fit
        base = base[base[target] != 0]
        trees = int(model.best_iteration_ * (1.1 if refit else 1))
        members = [lgb.LGBMClassifier(**{**params, "n_estimators": trees, "random_state": s}).fit(
            base[inputs], base[target] > 0) for s in range(n_seeds)]
        model.predict = lambda X: np.mean([m.predict_proba(X)[:, 1] for m in members], 0) - 0.5
        print(f"{n_seeds} model(s) with {trees} trees on {'fit + Oct-Dec 2024' if refit else 'fit'}")
else:
    model = lgb.LGBMRegressor(**params)
    model.fit(fit[inputs], clip(fit[target]), eval_set=[(stop[inputs], clip(stop[target]))],
              callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"lightgbm ({'binary' if binary else 'regression'}) stopped at {model.best_iteration_} trees "
      f"({time.time() - t0:.0f}s)")
pv = model.predict(val[inputs])
pooled = score(val, pv, "lightgbm, all coins pooled", prob=pv + 0.5 if binary else None)
per_coin = {}
for c in sorted(val["coin"].unique()):
    m = (val["coin"] == c).to_numpy()
    per_coin[c] = score(val[m], pv[m], f"  {c}", boot=200)
imp = pd.Series(model.booster_.feature_importance("gain"), inputs).sort_values(ascending=False)
print("feature gain share:", ", ".join(f"{k} {v:.0%}" for k, v in (imp / imp.sum()).head(10).items()))
test_stats = None
if use_test:
    pt = model.predict(test[inputs])
    test_stats = score(test, pt, "TEST 2026, lightgbm pooled", prob=pt + 0.5 if binary else None)
    for c in sorted(test["coin"].unique()):
        m = (test["coin"] == c).to_numpy()
        score(test[m], pt[m], f"  TEST {c}", boot=200)
if arg("save", ""):
    name = arg("save", "")
    if binary:
        np.savez(f"checkpoints/preds_{name}.npz", coin=val["coin"].to_numpy(), time=val["time"].to_numpy(), prob=pv + 0.5)
    write_json(f"runs/{name}.json", {
        "name": name, "kind": "lightgbm", "done": True, "snapshots": [],
        "config": dict(target=target, objective="binary" if binary else "regression", coins=coins or "all 20",
                       trees=model.best_iteration_, inputs=inputs, half_life=half_life),
        "val": {"pooled": pooled, "per_coin": per_coin}, "rules": rules, "test": test_stats,
        "importance": {k: float(v) for k, v in (imp / imp.sum()).head(15).items()}})
    print(f"saved runs/{name}.json  ->  http://localhost:8000/direction.html?run={name}")

if "--walk" in sys.argv and binary:
    # Walk-forward: at the start of each month from 2026-01, refit on everything before it (25-hour purge) with the tree
    # count chosen above, then predict that month. Months from HOLDOUT on are predicted and saved, never scored.
    trees = int(model.best_iteration_ * 1.1)
    parts, monthly = [], []
    for M in pd.date_range(WALK_FROM, df["time"].max(), freq="MS"):
        base = df[(df["time"] < M - purge) & (df[target] != 0)]
        w = None if arg("weight", "") != "move" else base[target].abs().clip(upper=3)
        mdl = lgb.LGBMClassifier(**{**params, "n_estimators": trees}).fit(base[inputs], base[target] > 0, sample_weight=w)
        mm = ((walk_rows["time"] >= M) & (walk_rows["time"] < M + pd.offsets.MonthBegin(1))).to_numpy()
        pw = mdl.predict_proba(walk_rows.loc[mm, inputs])[:, 1]
        parts.append(pd.DataFrame({"coin": walk_rows["coin"].to_numpy()[mm], "time": walk_rows["time"].to_numpy()[mm], "prob": pw}))
        if M < HOLDOUT:
            sw = direction_stats(walk_rows["time"].to_numpy()[mm], walk_rows[target].to_numpy()[mm], pw - 0.5, boot=100)
            monthly.append({"month": str(M)[:7], "hit": sw["hit"], "top20": sw["top20"], "n": sw["n"]})
            print(f"{str(M)[:7]}  refit {sw['hit'][0]:.2%} (top20 {sw['top20'][0]:.2%})", flush=True)
        else:
            print(f"{str(M)[:7]}  (holdout: predicted and saved, not scored)", flush=True)
    wp = pd.concat(parts)
    dev = wp[wp["time"] < HOLDOUT - purge].merge(df[["coin", "time", target]], on=["coin", "time"])
    score(dev, dev["prob"].to_numpy() - 0.5, "WALK-FORWARD 2026 (before holdout)", prob=dev["prob"].to_numpy())
    if arg("save", ""):
        np.savez(f"checkpoints/preds_{arg('save', '')}_walk.npz", coin=wp["coin"].to_numpy(), time=wp["time"].to_numpy(),
                 prob=wp["prob"].to_numpy())
        print(f"saved checkpoints/preds_{arg('save', '')}_walk.npz")

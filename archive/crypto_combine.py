"""Average the predicted P(up) of several direction models (saved by crypto_probe.py / crypto_seq.py) and score it.

Usage: python crypto_combine.py NAME1 NAME2 ... [--save=NAME]
Reads checkpoints/preds_<NAME>.npz (2025 validation predictions); only coin-hours predicted by every model are used.
"""
import sys

import numpy as np
import pandas as pd

from crypto_eval import direction_stats, line
from crypto_features import features
from monitor import write_json

names = [a for a in sys.argv[1:] if not a.startswith("--")]
save = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--save=")), "")
frames = []
for n in names:
    d = np.load(f"checkpoints/preds_{n}.npz", allow_pickle=True)
    frames.append(pd.DataFrame({"coin": d["coin"], "time": pd.to_datetime(d["time"]), n: d["prob"]}))
m = frames[0]
for f in frames[1:]:
    m = m.merge(f, on=["coin", "time"])
m = m.merge(features()[["coin", "time", "y1"]], on=["coin", "time"])
m["prob"] = m[names].mean(1)
print(f"{len(m)} coin-hours predicted by all of {names}; correlation of the models' P(up):")
print(m[names].corr().round(3).to_string())
for n in names + ["prob"]:
    print(line(n if n != "prob" else "AVERAGE", direction_stats(m["time"].to_numpy(), m["y1"].to_numpy(), m[n] - 0.5)))
if save:
    pooled = direction_stats(m["time"].to_numpy(), m["y1"].to_numpy(), m["prob"] - 0.5, prob=m["prob"])
    per_coin = {c: direction_stats(g["time"].to_numpy(), g["y1"].to_numpy(), g["prob"] - 0.5, prob=g["prob"], boot=200)
                for c, g in m.groupby("coin")}
    write_json(f"runs/{save}.json", {"name": save, "kind": "average of " + ", ".join(names), "done": True,
                                     "snapshots": [], "config": {"models": names},
                                     "val": {"pooled": pooled, "per_coin": per_coin}})
    m[["coin", "time", "prob"]].pipe(lambda d: np.savez(f"checkpoints/preds_{save}.npz", coin=d["coin"].to_numpy(),
                                                        time=d["time"].to_numpy(), prob=d["prob"].to_numpy()))
    print(f"saved runs/{save}.json  ->  http://localhost:8000/direction.html?run={save}")

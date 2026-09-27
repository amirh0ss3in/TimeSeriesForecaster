"""The series used across experiments. Each is (x_train, y_train, x_test, y_test, x_unit)."""
import math

import numpy as np


def synthetic(f, noise=0.0, seed=0):
    """Same data as train.py's synthetic signals: train on [0, 24π), test on [24π, 64π]."""
    h = math.pi / 20
    x_tr, x_te = np.arange(480) * h, (480 + np.arange(801)) * h
    rng = np.random.default_rng(seed)
    y_tr = f(x_tr) + noise * rng.standard_normal(len(x_tr))
    return x_tr, y_tr, x_te, f(x_te) + noise * rng.standard_normal(len(x_te)), "pi"


def real(file, delim, split_year):
    rows = [line for line in open(file) if line[:1].isdigit()]
    r = np.loadtxt(rows, delimiter=delim, usecols=(2, 3))    # decimal year, value
    tr = r[:, 0] < split_year
    return r[tr, 0], r[tr, 1], r[~tr, 0], r[~tr, 1], "year"


def all_series(horizon=0):
    """horizon > 0 keeps only the first `horizon` test points of each series (quick checks)."""
    series = {
        "sin": synthetic(np.sin),
        "trend": synthetic(lambda x: np.sin(x) + 0.1 * x),
        "amp (noise 0.3)": synthetic(lambda x: 0.1 * x * np.sin(x), 0.3),
        "sunspots": real("archive/data/sunspots.csv", ";", 1990),
        "co2": real("archive/data/co2.csv", ",", 2010),
    }
    if horizon:
        series = {k: (s[0], s[1], s[2][:horizon], s[3][:horizon], s[4]) for k, s in series.items()}
    return series

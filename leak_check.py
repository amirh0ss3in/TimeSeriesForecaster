"""Leak test for crypto_features.py: build the feature table from data cut off at a date, and check that every input of
every hour before the cutoff equals the full build. An input that secretly used later data would differ.

Usage: python leak_check.py [2025-06-01T13:00]
"""
import sys

import numpy as np
import pandas as pd

from crypto_features import INPUTS, PERP_INPUTS, build

INPUTS = INPUTS + PERP_INPUTS

cut = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2025-06-01T13:00")
full, part = build(), build(cutoff=cut)
key = ["coin", "time"]
m = full.merge(part, on=key, suffixes=("", "_cut"))
m = m[m["time"] < cut - pd.Timedelta(hours=1)]      # the last hour before the cut has no complete next hour anyway
bad = {c: int((~np.isclose(m[c], m[c + "_cut"], equal_nan=True)).sum()) for c in INPUTS}
print(f"cut at {cut}: compared {len(m)} coin-hours before it, {len(INPUTS)} inputs")
print("inputs that changed:", {k: v for k, v in bad.items() if v} or "none - no input depends on later data")

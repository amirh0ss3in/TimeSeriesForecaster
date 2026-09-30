"""Direction scoring shared by crypto_probe.py and crypto_seq.py.

hit rate: share of next moves whose sign the prediction got right (zero moves excluded, zero predictions = half).
top20%: hit rate of the 20% most confident predictions. big moves: hit rate on moves > 0.5 sigma, where quote bounce
can't decide the sign. IC: correlation of prediction and (clipped) outcome. 95% ranges resample whole days, since
coins move together within an hour.
"""
import numpy as np


def direction_stats(time, y, pred, prob=None, boot=300, seed=0):
    """time: datetime64 array; y: outcome in sigma units; pred: signed score (sign = predicted direction);
    prob: optional P(up) for the calibration curve."""
    y, pred, time = np.asarray(y, float), np.asarray(pred, float), np.asarray(time)
    keep = (y != 0) & np.isfinite(y) & np.isfinite(pred)
    y, p, day = y[keep], pred[keep], time[keep].astype("datetime64[D]")
    hit = np.where(p == 0, 0.5, (np.sign(p) == np.sign(y)).astype(float))
    conf = np.abs(p) >= np.quantile(np.abs(p), 0.8)
    big = np.abs(y) > 0.5
    days, inv = np.unique(day, return_inverse=True)
    draws = np.random.default_rng(seed).integers(0, len(days), (boot, len(days)))

    def rate(mask):
        h, n = np.bincount(inv, hit * mask, len(days)), np.bincount(inv, mask.astype(float), len(days))
        lo, hi = np.percentile(h[draws].sum(1) / n[draws].sum(1), [2.5, 97.5])
        return [float(hit[mask].mean()), float(lo), float(hi)]

    # Hit rate by confidence decile (0 = least confident 10%, 9 = most confident 10%).
    dec = np.minimum((np.argsort(np.argsort(np.abs(p))) * 10) // len(p), 9)
    out = {"hit": rate(np.ones_like(hit, bool)), "top20": rate(conf), "big": rate(big),
           "ic": float(np.corrcoef(p, np.clip(y, -5, 5))[0, 1]), "n": int(keep.sum()),
           "by_confidence": [float(hit[dec == k].mean()) for k in range(10)]}
    if prob is not None:
        pr = np.asarray(prob, float)[keep]
        edges = np.quantile(pr, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, pr, side="right") - 1, 0, 9)
        out["calibration"] = [[float(pr[b == k].mean()), float((y[b == k] > 0).mean())] for k in range(10)]
    return out


def line(name, s):
    f = lambda r: f"{r[0]:.2%} [{r[1]:.1%},{r[2]:.1%}]"
    return (f"{name:34s} hit {f(s['hit'])}  top20% {f(s['top20'])}  big moves {f(s['big'])}  IC {s['ic']:+.4f}"
            f"  n={s['n']}")

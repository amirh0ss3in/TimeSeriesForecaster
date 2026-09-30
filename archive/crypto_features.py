"""Hourly feature table for all downloaded coins on one UTC hour grid, cached to data/crypto/features.pkl.

Row (coin, t) describes the candle that opened at hour t. Every input feature of that row is known once that candle
has closed (t + 1h). Targets are the NEXT candle(s) and are named y_*; they must never be used as inputs.
Returns are divided by a causal volatility estimate (EWMA of past squared returns), so coins are comparable.
"""
import glob
import os

import numpy as np
import pandas as pd

CACHE = "data/crypto/features.pkl"
LAM = 0.94                   # EWMA decay for the volatility used to normalize returns


def load(cutoff=None):
    coins = sorted(os.path.basename(f)[:-7] for f in glob.glob("data/crypto/*.csv.gz"))
    raw = {c: pd.read_csv(f"data/crypto/{c}.csv.gz", parse_dates=["time"]).set_index("time") for c in coins}
    if cutoff is not None:                                   # leak test: pretend no data exists from `cutoff` on
        raw = {c: d[d.index < cutoff] for c, d in raw.items()}
    grid = pd.date_range(min(d.index[0] for d in raw.values()), max(d.index[-1] for d in raw.values()), freq="h")
    wide = {k: pd.DataFrame({c: raw[c][k].reindex(grid) for c in coins}) for k in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume")}
    return coins, grid, wide


def intra_hour(coins, grid, cutoff=None):
    """From 5-minute candles (data/crypto_5m, see fetch_crypto.py --interval=5m), per coin and hour t: the log move of
    the hour's last 5 and last 15 minutes, the realized volatility of its 5-minute moves, the order-flow imbalance of
    its last 15 minutes and their share of the hour's volume. The last 5-minute candle closes when hour t closes."""
    out = {k: {} for k in ("l5", "l15", "rv5", "imb15", "vsh15")}
    for c in coins:
        d = pd.read_csv(f"data/crypto_5m/{c}.csv.gz", parse_dates=["time"],
                        usecols=["time", "close", "volume", "taker_buy_volume"])
        if cutoff is not None:
            d = d[d["time"] < cutoff]
        d["hour"], d["minute"] = d["time"].dt.floor("h"), d["time"].dt.minute
        d = d[d["minute"] % 5 == 0]                  # a 2018 exchange restart left a few candles on odd minutes
        p = {k: d.pivot_table(index="hour", columns="minute", values=k, aggfunc="last")
                 .reindex(columns=range(0, 60, 5)) for k in ("close", "volume", "taker_buy_volume")}
        cl = np.log(p["close"])
        last = [45, 50, 55]
        v15 = p["volume"][last].sum(1)
        out["l5"][c] = cl[55] - cl[50]
        out["l15"][c] = cl[55] - cl[40]
        out["rv5"][c] = np.sqrt((cl.diff(axis=1) ** 2).sum(1, min_count=6))
        out["imb15"][c] = (2 * p["taker_buy_volume"][last].sum(1) / v15 - 1).where(v15 > 0)
        out["vsh15"][c] = (v15 / p["volume"].sum(1)).where(p["volume"].sum(1) > 0)
    return {k: pd.DataFrame(v).reindex(grid)[coins] for k, v in out.items()}


def other_market(folder, cols, coins, grid, cutoff=None):
    """Hourly columns from another Binance market (perp futures or premium index), wide per coin; NaN where the coin
    has no such market (yet). Rows are aligned by candle open time, like spot."""
    out = {k: {} for k in cols}
    for c in coins:
        f = f"{folder}/{c}.csv.gz"
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, parse_dates=["time"]).set_index("time")
        d = d[~d.index.duplicated()]
        if cutoff is not None:
            d = d[d.index < cutoff]
        for k in cols:
            out[k][c] = d[k]
    return {k: pd.DataFrame(v).reindex(index=grid, columns=coins) for k, v in out.items()}


def build(cutoff=None):
    coins, grid, w = load(cutoff)
    listed = w["close"].notna().cummax()                     # from each coin's first candle on
    gap = listed & w["close"].isna()                         # exchange outages inside a coin's history
    close = w["close"].ffill().where(listed)
    ret = np.log(close).diff()                               # r_t: move during candle t (0 across outages)
    rng = np.log(w["high"] / w["low"]).where(~gap, 0.0)      # high-low range of candle t
    qvol = np.log1p(w["quote_volume"]).where(~gap)            # log quote volume of candle t
    # Order flow: +1 = all volume was aggressive buying, -1 = all aggressive selling.
    imb = (2 * w["taker_buy_volume"] / w["volume"] - 1).where(w["volume"] > 0, 0.0).where(~gap, 0.0)
    trades = np.log1p(w["trades"]).where(~gap)
    sig = np.sqrt((ret ** 2).ewm(alpha=1 - LAM, adjust=False).mean())   # sigma_t uses returns up to t
    sig = sig.clip(lower=sig.rolling(24 * 30, min_periods=24).median() * 0.2)   # guard against ~0 after flat stretches
    z = ret / sig.shift(1)                                   # r_t in units of the volatility known before t
    zc = z.clip(-10, 10)
    mkt = zc.mean(1)                                         # market move: average normalized return of listed coins
    # Candle shape: where the close sits in the high-low range (-1 = at the low, +1 = at the high), and the wicks.
    hl = (w["high"] - w["low"]).where(~gap)
    clv = ((2 * w["close"] - w["high"] - w["low"]) / hl).where(hl > 0, 0.0)
    top, bot = np.maximum(w["open"], w["close"]), np.minimum(w["open"], w["close"])
    wick_up = (np.log(w["high"] / top) / sig.shift(1)).where(~gap, 0.0).clip(0, 10)
    wick_dn = (np.log(bot / w["low"]) / sig.shift(1)).where(~gap, 0.0).clip(0, 10)
    ups = np.sign(ret).rolling(6, min_periods=6).mean()     # share of up hours minus down hours, last 6
    ih = intra_hour(coins, grid, cutoff)
    l5, l15 = (ih["l5"] / sig.shift(1)).clip(-10, 10), (ih["l15"] / sig.shift(1)).clip(-10, 10)
    # Perpetual futures: their order flow, their share of volume, how their price moved against spot, and the premium
    # of the perp over the spot index (level vs the past week, and its last change).
    pf = other_market("data/crypto_perp", ["close", "volume", "quote_volume", "taker_buy_volume"], coins, grid, cutoff)
    pr = other_market("data/crypto_premium", ["close"], coins, grid, cutoff)["close"]
    pimb = (2 * pf["taker_buy_volume"] / pf["volume"] - 1).where(pf["volume"] > 0)
    pshare = np.log((pf["quote_volume"] + 1) / (w["quote_volume"] + 1))
    pbasis = ((np.log(pf["close"]).diff() - ret) / sig.shift(1)).clip(-10, 10)
    prem_sd = pr.rolling(168, min_periods=24).std() + 1e-6

    def roll(df, n):                                         # past n hours including t, scaled like one z
        return df.rolling(n, min_periods=n).sum() / np.sqrt(n)

    feats = {
        "z0": zc, "z1": zc.shift(1), "z2": zc.shift(2), "z3": zc.shift(3),
        "z6": roll(zc, 6), "z24": roll(zc, 24), "z168": roll(zc, 168),
        "rng": np.log((rng + 1e-5) / (rng.rolling(168, min_periods=24).mean() + 1e-5)),   # 1e-5: flat candles
        "vol": qvol - qvol.rolling(168, min_periods=24).mean(),
        "regime": np.log(sig / sig.rolling(24 * 30, min_periods=24 * 7).mean()),
        "rel0": zc.sub(mkt, axis=0), "rel24": roll(zc, 24).sub(roll(mkt.to_frame("m"), 24)["m"], axis=0),
        "clv0": clv, "clv1": clv.shift(1), "wick_up": wick_up, "wick_dn": wick_dn, "ups6": ups,
        "imb0": imb, "imb1": imb.shift(1), "imb6": imb.rolling(6, min_periods=6).mean(),
        "imb_rel": imb - imb.rolling(168, min_periods=24).mean(),
        "trades": trades - trades.rolling(168, min_periods=24).mean(),
        "l5": l5, "l15": l15, "rv5": np.log((ih["rv5"] + 1e-6) / (sig.shift(1) + 1e-6)),
        "imb15": ih["imb15"], "vsh15": ih["vsh15"],
        "pimb0": pimb, "pimb6": pimb.rolling(6, min_periods=6).mean(),
        "pshare": pshare - pshare.rolling(168, min_periods=24).mean(), "pbasis": pbasis,
        "prem": (pr - pr.rolling(168, min_periods=24).mean()) / prem_sd, "dprem": pr.diff() / prem_sd,
    }
    shared = pd.DataFrame({
        "mkt0": mkt, "mkt1": mkt.shift(1), "mkt6": roll(mkt.to_frame("m"), 6)["m"],
        "mkt24": roll(mkt.to_frame("m"), 24)["m"],
        "btc0": zc["BTC"], "btc1": zc["BTC"].shift(1), "eth0": zc["ETH"], "eth1": zc["ETH"].shift(1),
        "mkt_clv": clv.mean(1), "btc_clv": clv["BTC"], "mkt_imb": imb.mean(1), "btc_imb": imb["BTC"],
        "mkt_l5": l5.mean(1), "mkt_l15": l15.mean(1), "btc_l5": l5["BTC"], "btc_l15": l15["BTC"],
        "btc_imb15": ih["imb15"]["BTC"],
        "mkt_pimb": pimb.mean(1), "btc_pimb": pimb["BTC"], "mkt_prem": ((pr - pr.rolling(168, min_periods=24).mean())
                                                                       / prem_sd).mean(1),
        "hour": grid.hour, "weekday": grid.weekday,
    }, index=grid)
    # Targets: next candle's return, and the sum of the next 24, in units of sigma_t (known at t).
    y1 = ret.shift(-1) / sig
    y24 = ret[::-1].rolling(24, min_periods=24).sum()[::-1].shift(-1) / (sig * np.sqrt(24))

    rows = []
    for c in coins:
        d = pd.DataFrame({k: v[c] for k, v in feats.items()}, index=grid).join(shared)
        d["coin"], d["y1"], d["y24"], d["r1"] = c, y1[c], y24[c], ret[c].shift(-1)
        d["sig"] = sig[c]                                    # volatility known at t (converts vol units to returns)
        d["gap_next"] = gap[c].shift(-1, fill_value=False)
        rows.append(d[listed[c]].iloc[24 * 7:])              # skip each coin's first week (features warming up)
    df = pd.concat(rows).rename_axis("time").reset_index()
    df = df[~df["gap_next"]].drop(columns="gap_next")        # no targets that are outage filler
    if cutoff is None:
        df.to_pickle(CACHE)
    return df


def features():
    return pd.read_pickle(CACHE) if os.path.exists(CACHE) else build()


INPUTS = ["z0", "z1", "z2", "z3", "z6", "z24", "z168", "rng", "vol", "regime", "rel0", "rel24",
          "mkt0", "mkt1", "mkt6", "mkt24", "btc0", "btc1", "eth0", "eth1", "hour", "weekday",
          "clv0", "clv1", "wick_up", "wick_dn", "ups6", "mkt_clv", "btc_clv",
          "imb0", "imb1", "imb6", "imb_rel", "trades", "mkt_imb", "btc_imb",
          "l5", "l15", "rv5", "imb15", "vsh15", "mkt_l5", "mkt_l15", "btc_l5", "btc_l15", "btc_imb15"]
PERP_INPUTS = ["pimb0", "pimb6", "pshare", "pbasis", "prem", "dprem", "mkt_pimb", "btc_pimb", "mkt_prem"]   # NaN before a coin's perp

if __name__ == "__main__":
    df = build()
    print(df.shape, df["time"].min(), df["time"].max())
    print(df.groupby("coin").size().to_string())
    print(df[INPUTS + ["y1", "y24"]].describe().T[["mean", "std", "min", "max"]].round(3).to_string())

"""Coins for a point-in-time universe: daily candles and perpetual funding for every Binance USDT coin that was ever
among the most traded, delisted ones included, so each month's tradable list can be rebuilt from what was known then.

1. List every <COIN>USDT spot pair and every USD-M perpetual in the archive (including delisted ones).
2. Sample each pair's USDT volume at one month per quarter; keep coins that were in the top 40 at some snapshot and
   have a perpetual (so they can be shorted). Stablecoins, fiat, wrapped and leveraged tokens are left out.
3. Download their full daily candles (data/universe/<COIN>.csv.gz, same columns as data/crypto/) and funding rates
   (data/universe_funding/<COIN>.csv.gz: time, funding_rate).
Usage: python fetch_universe.py [--hourly]
  --hourly: instead, hourly closes (time, close) from Dec 2020 for the coins already in data/universe/, to test trades
  placed some hours after the daily decision (data/universe_1h/<COIN>.csv.gz).
"""
import csv
import datetime as dt
import glob
import gzip
import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = "https://data.binance.vision/data/"
LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix={p}&marker={m}"
NOT_COINS = {"USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "UST", "EUR", "GBP", "AUD",
             "BRL", "TRY", "RUB", "AEUR", "EURI", "XUSD", "USD1", "BFUSD", "RLUSD", "PYUSD", "USDE", "PAXG", "XAUT",
             "WBTC", "WBETH", "BETH", "BNSOL", "BTCB", "STETH", "BKRW", "IDRT", "BIDR", "NGN", "UAH", "ZAR"}
TOP, FIRST, LAST = 40, (2019, 1), (2026, 8)
months = [f"{y}-{m:02d}" for y in range(FIRST[0], LAST[0] + 1) for m in range(1, 13) if FIRST <= (y, m) <= LAST]


def listing(prefix):
    out, marker = [], ""
    while True:
        x = urllib.request.urlopen(LIST.format(p=prefix, m=marker), timeout=60).read().decode()
        p = re.findall(r"<Prefix>" + re.escape(prefix) + r"([^/<]+)/</Prefix>", x)
        out += p
        if "<IsTruncated>true" not in x or not p:
            return out
        marker = prefix + p[-1] + "/"


def get(url):
    """Rows of one archive file, or [] if it doesn't exist."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            if attempt == 5:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
        time.sleep(2 ** attempt)
    rows = []
    for line in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
        if line[0].isdigit():
            t = int(line[0])
            t = t // 1000 if t > 10 ** 14 else t       # spot files switched to microseconds in 2025
            rows.append([dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")] + line[1:])
    return rows


kline = lambda s, m: f"{ROOT}spot/monthly/klines/{s}USDT/1d/{s}USDT-1d-{m}.zip"
funding = lambda s, m: f"{ROOT}futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"

if __name__ == "__main__":
    if "--hourly" in sys.argv:
        hourly = lambda s, m: f"{ROOT}spot/monthly/klines/{s}USDT/1h/{s}USDT-1h-{m}.zip"
        os.makedirs("data/universe_1h", exist_ok=True)
        with ThreadPoolExecutor(32) as pool:
            for p in sorted(glob.glob("data/universe/*.csv.gz")):
                b = os.path.basename(p)[:-7]
                if os.path.exists(f"data/universe_1h/{b}.csv.gz"):
                    continue                               # resume after an interruption
                with gzip.open(p, "rt") as fh:
                    start = max("2020-12", (list(csv.reader(fh))[1:2] or [["9999"]])[0][0][:7])
                rows = [[r[0], r[4]] for c in pool.map(lambda m: get(hourly(b, m)), [m for m in months if m >= start])
                        for r in c]
                with gzip.open(f"data/universe_1h/{b}.csv.gz", "wt", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["time", "close"])
                    w.writerows(rows)
                print(f"{b:10s} {len(rows):6d} hours", flush=True)
        sys.exit()

    # 1. Every USDT coin and its perpetual (small-price coins trade per 1000 or per million units on futures).
    spot = [s[:-4] for s in listing("data/spot/monthly/klines/") if s.endswith("USDT")]
    perps = set(listing("data/futures/um/monthly/fundingRate/"))
    lev = {b for b in spot for suf in ("UP", "DOWN", "BULL", "BEAR") if b.endswith(suf) and b[:-len(suf)] in spot}
    coins = [b for b in spot if b.isascii() and b.isalnum() and b not in NOT_COINS and b not in lev]
    perp = {b: next((p + b + "USDT" for p in ("", "1000", "1000000", "1M") if p + b + "USDT" in perps), None) for b in coins}
    coins = [b for b in coins if perp[b]]
    print(f"{len(spot)} USDT spot coins, {len(coins)} with a perpetual (after leaving out stablecoins and tokens)", flush=True)

    # 2. Quarterly volume snapshots -> coins that were ever in the top 40.
    snaps = [m for m in months if int(m[5:]) in (1, 4, 7, 10)]
    with ThreadPoolExecutor(32) as pool:
        vol = dict(zip([(b, m) for b in coins for m in snaps],
                       pool.map(lambda bm: sum(float(r[7]) for r in get(kline(*bm))), [(b, m) for b in coins for m in snaps])))
    keep = set()
    for m in snaps:
        keep |= set(sorted(coins, key=lambda b: -vol[(b, m)])[:TOP])
    keep = sorted(keep)
    print(f"{len(keep)} coins were in the top {TOP} by USDT volume at some quarterly snapshot:", " ".join(keep), flush=True)

    # 3. Full daily candles and funding rates.
    os.makedirs("data/universe", exist_ok=True)
    os.makedirs("data/universe_funding", exist_ok=True)
    with ThreadPoolExecutor(32) as pool:
        for b in keep:
            if os.path.exists(f"data/universe_funding/{b}.csv.gz"):
                continue                                   # resume after an interruption
            rows = [r[:6] + r[7:10] for c in pool.map(lambda m: get(kline(b, m)), months) for r in c]
            with gzip.open(f"data/universe/{b}.csv.gz", "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"])
                w.writerows(rows)
            fr = [[r[0], r[-1]] for c in pool.map(lambda m: get(funding(perp[b], m)), months) for r in c]
            with gzip.open(f"data/universe_funding/{b}.csv.gz", "wt", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["time", "funding_rate"])
                w.writerows(fr)
            print(f"{b:10s} {len(rows):5d} days {rows[0][0][:10] if rows else '-'} .. {rows[-1][0][:10] if rows else '-'}"
                  f"   funding {len(fr)} ({perp[b]})", flush=True)

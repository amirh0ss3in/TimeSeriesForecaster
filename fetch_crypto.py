"""Download hourly candles for the largest coins from Binance's public archive (data.binance.vision).

Candidates are in market-cap order (CoinGecko, 2026-09-27) without stablecoins, gold tokens and exchange tokens not
traded on Binance; the first 20 with a <COIN>USDT hourly archive are kept.
Saves data/crypto/<COIN>.csv.gz with columns: time (UTC, start of the hour), open, high, low, close, volume.
"""
import csv
import datetime as dt
import gzip
import io
import os
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

CANDIDATES = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "ZEC", "HYPE", "DOGE", "LINK", "XMR", "ADA", "XLM", "BCH",
              "NEAR", "UNI", "LTC", "SUI", "AVAX", "TON", "HBAR", "TAO", "SHIB", "ENA", "ONDO", "AAVE", "DOT", "ICP"]
N_COINS, FIRST, OUT = 20, (2017, 8), "data/crypto"
URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{y}-{m:02d}.zip"

today = dt.date.today()
last = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)   # last complete month
months = [(y, m) for y in range(FIRST[0], last[0] + 1) for m in range(1, 13) if FIRST <= (y, m) <= last]


def get(sym, y, m):
    """Rows of one monthly file, or [] if the pair had no data that month."""
    try:
        with urllib.request.urlopen(URL.format(s=sym, y=y, m=m), timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    rows = []
    for line in csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))):
        if not line[0].isdigit():                    # some files have a header line
            continue
        t = int(line[0])
        t = t // 1000 if t > 10 ** 14 else t         # spot files switched from milliseconds to microseconds in 2025
        rows.append([dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")] + line[1:6])
    return rows


def available(coin):
    return bool(get(f"{coin}USDT", *last))


os.makedirs(OUT, exist_ok=True)
with ThreadPoolExecutor(16) as pool:
    ok = list(pool.map(available, CANDIDATES))
coins = [c for c, a in zip(CANDIDATES, ok) if a][:N_COINS]
print("skipped (no Binance USDT hourly archive):", [c for c, a in zip(CANDIDATES, ok) if not a])
print("coins:", coins)

with ThreadPoolExecutor(16) as pool:
    for coin in coins:
        chunks = list(pool.map(lambda ym: get(f"{coin}USDT", *ym), months))
        rows = [r for c in chunks for r in c]
        with gzip.open(f"{OUT}/{coin}.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "open", "high", "low", "close", "volume"])
            w.writerows(rows)
        print(f"{coin:5s} {len(rows):6d} hours  {rows[0][0]} .. {rows[-1][0]}")

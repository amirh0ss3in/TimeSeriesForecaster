"""Download hourly candles for the largest coins from Binance's public archive (data.binance.vision).

Candidates are in market-cap order (CoinGecko, 2026-09-27) without stablecoins, gold tokens and exchange tokens not
traded on Binance; the first 20 with a <COIN>USDT hourly archive are kept.
Usage: python fetch_crypto.py [--interval=1h|5m] [--market=spot|perp|premium] [--daily]
  --daily: fetch the current month's daily files up to yesterday and merge them into the existing files.
  5m goes to data/crypto_5m/. perp = USDT-margined perpetual futures candles (data/crypto_perp/), premium = their
  premium-index candles, i.e. (perp - spot index) / index (data/crypto_premium/); both use the same 20 coins as spot.
Saves data/crypto/<COIN>.csv.gz with columns: time (UTC, start of the hour), open, high, low, close, volume,
quote_volume, trades, taker_buy_volume (base-asset volume bought by takers, i.e. aggressive buyers).
"""
import csv
import sys
import time
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
arg = lambda k, d: next((a.split("=")[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
IV, MARKET = arg("interval", "1h"), arg("market", "spot")
PATH = {"spot": "spot/monthly/klines", "perp": "futures/um/monthly/klines",
        "premium": "futures/um/monthly/premiumIndexKlines"}[MARKET]
N_COINS, FIRST = 20, (2017, 8)
OUT = "data/crypto" + ("" if MARKET == "spot" else f"_{MARKET}") + ("" if IV == "1h" else f"_{IV}")
DAILY = "--daily" in sys.argv
if DAILY:
    PATH = PATH.replace("/monthly/", "/daily/")
URL = f"https://data.binance.vision/data/{PATH}/{{s}}/{IV}/{{s}}-{IV}-{{p}}.zip"   # p: 2026-08 or 2026-09-01
PERP_NAME = {"SHIB": "1000SHIB"}                     # futures list some small-price coins per 1000 units
symbol = lambda coin: (coin if MARKET == "spot" else PERP_NAME.get(coin, coin)) + "USDT"

today = dt.date.today()
last = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)   # last complete month
months = [f"{y}-{m:02d}" for y in range(FIRST[0], last[0] + 1) for m in range(1, 13) if FIRST <= (y, m) <= last]
days = [str(today.replace(day=d)) for d in range(1, today.day)]     # this month, up to yesterday


def get(sym, period):
    """Rows of one monthly (or daily) file, or [] if the pair had no data then."""
    for attempt in range(6):                         # the archive occasionally times out: retry with back-off
        try:
            with urllib.request.urlopen(URL.format(s=sym, p=period), timeout=60) as r:
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
        if not line[0].isdigit():                    # some files have a header line
            continue
        t = int(line[0])
        t = t // 1000 if t > 10 ** 14 else t         # spot files switched from milliseconds to microseconds in 2025
        rows.append([dt.datetime.fromtimestamp(t / 1000, dt.UTC).strftime("%Y-%m-%d %H:%M")] + line[1:6] + line[7:10])
    return rows


def available(coin):
    return bool(get(symbol(coin), days[0] if DAILY else f"{last[0]}-{last[1]:02d}"))


os.makedirs(OUT, exist_ok=True)
candidates = CANDIDATES if MARKET == "spot" and not DAILY else sorted(f[:-7] for f in os.listdir("data/crypto") if f.endswith(".csv.gz"))
with ThreadPoolExecutor(16) as pool:
    ok = list(pool.map(available, candidates))
coins = [c for c, a in zip(candidates, ok) if a][:N_COINS]
print(f"skipped (no {MARKET} archive):", [c for c, a in zip(candidates, ok) if not a])
print("coins:", coins)

with ThreadPoolExecutor(16) as pool:
    for coin in coins:
        f = f"{OUT}/{coin}.csv.gz"
        if DAILY:                                    # merge the new days into the existing file
            new = [r for c in pool.map(lambda p: get(symbol(coin), p), days) for r in c]
            with gzip.open(f, "rt", newline="") as fh:
                old = list(csv.reader(fh))[1:]
            rows = sorted({r[0]: r for r in old + new}.values(), key=lambda r: r[0])
        elif os.path.exists(f):                      # already downloaded (resume after an interruption)
            continue
        else:
            rows = [r for c in pool.map(lambda p: get(symbol(coin), p), months) for r in c]
        with gzip.open(f"{OUT}/{coin}.csv.gz", "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"])
            w.writerows(rows)
        print(f"{coin:5s} {len(rows):7d} candles  {rows[0][0]} .. {rows[-1][0]}", flush=True)

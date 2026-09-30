"""Daily image card for social posts: today's longs and shorts, the latest booked day, and the gain so far.

Usage: python social_card.py [--log=rank_paper] [--out=cards/<date>.jpg]
Writes a 1080x1350 JPEG (Instagram's 4:5 portrait size) from runs/<log>.json.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BRAND = "CRYPTO RANK RADAR"
BG, INK, MUTED, GREEN, RED, LINE = "#0f1419", "#e8eaed", "#8b949e", "#2ea043", "#f85149", "#30363d"


def card(log, out):
    days = log["daily"]
    today = days[-1]
    booked = [r for r in days if r["taker"] is not None]
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
    t = lambda x, y, s, **k: fig.text(x, y, s, color=k.pop("color", INK), family="DejaVu Sans", **k)

    t(0.07, 0.93, BRAND, size=22, weight="bold", color=MUTED)
    t(0.07, 0.875, pd.Timestamp(today["date"]).strftime("%a %d %b %Y"), size=42, weight="bold")
    t(0.07, 0.845, "Daily ranking of the 20 most-traded coins on Binance by a LightGBM model",
      size=17, color=MUTED)

    for x, title, coins, color in ((0.07, "LONG", today["long"], GREEN), (0.53, "SHORT", today["short"], RED)):
        fig.patches.append(plt.Rectangle((x, 0.575), 0.40, 0.235, transform=fig.transFigure, facecolor="#161b22",
                                         edgecolor=color, linewidth=3))
        t(x + 0.03, 0.765, title, size=26, weight="bold", color=color)
        for i, c in enumerate(coins):
            t(x + 0.03, 0.70 - i * 0.052, c, size=38, weight="bold")
    t(0.07, 0.545, "Each position 1/6 of capital · filled at the 01:00 UTC close · held 1 day", size=16, color=MUTED)

    if booked:
        last, total = booked[-1], sum(r["taker"] for r in booked)
        col = lambda v: GREEN if v > 0 else RED if v < 0 else INK
        t(0.07, 0.49, f"{pd.Timestamp(last['date']).strftime('%d %b')} result", size=20, color=MUTED)
        t(0.07, 0.435, f"{100 * last['taker']:+.2f}%", size=48, weight="bold", color=col(last["taker"]))
        t(0.53, 0.49, f"Since {pd.Timestamp(log['start']).strftime('%d %b %Y')} ({len(booked)} days)",
          size=20, color=MUTED)
        t(0.53, 0.435, f"{100 * total:+.2f}%", size=48, weight="bold", color=col(total))
        ax = fig.add_axes([0.1, 0.12, 0.83, 0.26], facecolor=BG)
        # Value as of each morning: a day's result is known the next day, and the test starts at 0.
        eq = pd.Series([r["taker"] for r in booked],
                       index=pd.to_datetime([r["date"] for r in booked]) + pd.Timedelta(days=1)).cumsum() * 100
        eq = pd.concat([pd.Series([0.0], index=[pd.Timestamp(log["start"])]), eq])
        ax.plot(eq.index, eq.values, color=GREEN if eq.iloc[-1] >= 0 else RED, linewidth=4, marker="o", markersize=7)
        ax.axhline(0, color=LINE, linewidth=1.5)
        ax.set_title("Gain so far (%, net of fees and funding)", color=MUTED, size=17, loc="left", pad=12)
        ax.tick_params(colors=MUTED, labelsize=14)
        for s in ax.spines.values():
            s.set_color(LINE)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=2, maxticks=6))
        ax.grid(axis="y", color=LINE, linewidth=0.8)
    else:
        t(0.07, 0.45, "First day: results are posted from tomorrow.", size=24, color=MUTED)
    t(0.07, 0.045, "Net of taker fees (0.05%) and funding · paper trading, no real money", size=16, color=MUTED)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, format="jpg", facecolor=BG, pil_kwargs={"quality": 92})
    plt.close(fig)
    return out


if __name__ == "__main__":
    name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--log=")), "rank_paper")
    with open(f"runs/{name}.json", encoding="utf-8") as fh:
        log = json.load(fh)
    out = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), f"cards/{log['daily'][-1]['date']}.jpg")
    print(card(log, out))

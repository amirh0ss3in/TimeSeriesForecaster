"""Posts the paper trader's latest decision and results to a Telegram channel (run by the paper-trader workflow).

Needs the environment variables TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID (@channelname, or the
channel's numeric id); without them it does nothing. Each decision day is posted once: the last posted day is kept in
runs/telegram_posted.txt.
Usage: python telegram_post.py [--log=rank_paper] [--dry-run]    (--dry-run: print the message, send nothing)
"""
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--log=")), "rank_paper")
token, chat, dry = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"), "--dry-run" in sys.argv
if not dry and (not token or not chat):
    print("telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, nothing posted")
    sys.exit()
with open(f"runs/{name}.json", encoding="utf-8") as fh:
    log = json.load(fh)
POSTED = "runs/telegram_posted.txt"
posted = open(POSTED).read().strip() if os.path.exists(POSTED) else ""
days = log.get("daily", [])
if not dry and (not days or days[-1]["date"] <= posted):
    print(f"telegram: nothing new since {posted or 'the start'}")
    sys.exit()

today, booked = days[-1], [r for r in days if r["taker"] is not None]
pc = lambda v: f"{100 * v:+.2f}%"
day = lambda s, n=0: (lambda d: f"{d.day} {d:%b}")(dt.date.fromisoformat(s) + dt.timedelta(days=n))
lines = [f"<b>Holding now: {day(today['date'])} 01:00 → {day(today['date'], 1)} 01:00 UTC</b>",
         f"LONG: {' · '.join(today['long'])}",
         f"SHORT: {' · '.join(today['short'])}",
         "<i>Each 1/6 of capital, opened at the 01:00 UTC price, rebalanced daily.</i>"]
if booked:
    last, month = booked[-1], today["date"][:7]
    mtd = [r["taker"] for r in booked if r["date"].startswith(month)]
    lines += ["",
              f"Result of {day(last['date'])} (long {' '.join(last['long'])}, short {' '.join(last['short'])}): "
              f"<b>{pc(last['taker'])}</b>",
              f"Total since {day(log['start'])}: {pc(sum(r['taker'] for r in booked))} over {len(booked)} days"
              + (f" · this month: {pc(sum(mtd))}" if mtd and len(mtd) < len(booked) else ""),
              "<i>Net of fees and funding. Paper trading, no real money.</i>"]
lines.append(f"<i>Posted {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC</i>")
if dry:
    print("\n".join(lines))
    sys.exit()

body = urllib.parse.urlencode({"chat_id": chat, "text": "\n".join(lines), "parse_mode": "HTML",
                               "disable_web_page_preview": "true"}).encode()
try:
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=body, timeout=30) as r:
        ok = json.loads(r.read()).get("ok")
except urllib.error.HTTPError as e:                  # never print the URL: it contains the token
    sys.exit(f"telegram: HTTP {e.code}: {json.loads(e.read() or b'{}').get('description', '')}")
if not ok:
    sys.exit("telegram: the message was not accepted")
with open(POSTED, "w") as fh:
    fh.write(today["date"])
print(f"telegram: posted {today['date']}")

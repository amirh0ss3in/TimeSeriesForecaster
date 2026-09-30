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
lines = [f"<b>Crypto ranking, {today['date']}</b>",
         f"Long: {' · '.join(today['long'])}",
         f"Short: {' · '.join(today['short'])}",
         f"<i>16.7% of capital each, from data up to 00:00 UTC, filled at the 01:00 UTC close, held 1 day</i>"]
if booked:
    month = today["date"][:7]
    mtd = [r["taker"] for r in booked if r["date"].startswith(month)]
    lines += ["",
              f"{booked[-1]['date']}: <b>{pc(booked[-1]['taker'])}</b>",
              f"Since {log['start']}: {pc(sum(r['taker'] for r in booked))} over {len(booked)} days"
              + (f" · {month}: {pc(sum(mtd))}" if mtd and len(mtd) < len(booked) else ""),
              f"<i>Net of taker fees (0.05%) and funding. Paper trading, no real money.</i>"]
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

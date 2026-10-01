"""Posts the paper trader's latest decision and results to a Telegram channel (run by the paper-trader workflow).

Needs the environment variables TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID (@channelname, or the
channel's numeric id); without them it does nothing. Each decision day is posted once; runs/telegram_posted.txt keeps
one line per post: date and Telegram message id.
Usage: python telegram_post.py [--log=rank_paper] [--dry-run] [--edit=ID@DATE,...]
  --dry-run: print the message, send nothing.  --edit: rewrite earlier posts (message ID, decision DATE).
"""
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

POSTED = "runs/telegram_posted.txt"


def message(log, date):
    """The post for one decision day: the positions held from that day's 01:00, and the results known by then."""
    days = log["daily"]
    today = next(r for r in days if r["date"] == date)
    booked = [r for r in days if r["date"] < date and r["taker"] is not None]
    pc = lambda v: f"{100 * v:+.2f}%"
    day = lambda s, n=0: (lambda d: f"{d.day} {d:%b}")(dt.date.fromisoformat(s) + dt.timedelta(days=n))
    lines = [f"<b>Holding now: {day(date)} 01:00 → {day(date, 1)} 01:00 UTC</b>",
             f"LONG: {' · '.join(today['long'])}",
             f"SHORT: {' · '.join(today['short'])}",
             "<i>Each 1/6 of capital, opened at the 01:00 UTC price, rebalanced daily.</i>"]
    if booked:
        last = booked[-1]
        mtd = [r["taker"] for r in booked if r["date"][:7] == date[:7]]
        lines += ["",
                  f"Result of {day(last['date'])} (long {' '.join(last['long'])}, short {' '.join(last['short'])}): "
                  f"<b>{pc(last['taker'])}</b>",
                  f"Total since {day(log['start'])}: {pc(sum(r['taker'] for r in booked))} over {len(booked)} days"
                  + (f" · this month: {pc(sum(mtd))}" if mtd and len(mtd) < len(booked) else ""),
                  "<i>Net of fees and funding. Paper trading, no real money.</i>"]
    return "\n".join(lines)


def call(method, token, **params):
    params.update(parse_mode="HTML", disable_web_page_preview="true")
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}",
                                    data=urllib.parse.urlencode(params).encode(), timeout=30) as r:
            return json.loads(r.read())["result"]
    except urllib.error.HTTPError as e:              # never print the URL: it contains the token
        sys.exit(f"telegram: {method}: HTTP {e.code}: {json.loads(e.read() or b'{}').get('description', '')}")


if __name__ == "__main__":
    opt = lambda k: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), None)
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    with open(f"runs/{opt('log') or 'rank_paper'}.json", encoding="utf-8") as fh:
        log = json.load(fh)
    latest = log["daily"][-1]["date"]
    if "--dry-run" in sys.argv:
        print(message(log, latest))
        sys.exit()
    if not token or not chat:
        print("telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, nothing posted")
        sys.exit()
    if opt("edit"):
        for item in opt("edit").split(","):
            mid, date = item.split("@")
            call("editMessageText", token, chat_id=chat, message_id=mid, text=message(log, date))
            print(f"telegram: edited message {mid} ({date})")
        sys.exit()
    lines = open(POSTED).read().split("\n") if os.path.exists(POSTED) else []
    posted = lines[-1].split()[0] if lines and lines[-1].strip() else ""
    if latest <= posted:
        print(f"telegram: nothing new since {posted}")
        sys.exit()
    sent = call("sendMessage", token, chat_id=chat, text=message(log, latest))
    with open(POSTED, "a") as fh:
        fh.write(f"\n{latest} {sent['message_id']}")
    print(f"telegram: posted {latest} (message {sent['message_id']})")

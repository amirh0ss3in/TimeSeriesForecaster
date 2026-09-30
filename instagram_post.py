"""Posts the day's image card (social_card.py) to Instagram; run by the paper-trader workflow.

Instagram only publishes images from a public web address, so the card is first committed to a public GitHub
repository (PUBLIC_REPO, in cards/<date>.jpg) and published from there. Environment:
  INSTAGRAM_TOKEN    long-lived Instagram user token (Instagram API with Instagram Login, permissions
                     instagram_business_basic and instagram_business_content_publish)
  PUBLISH_TOKEN      fine-grained GitHub token: Contents read/write on PUBLIC_REPO, Secrets read/write on this repo
  PUBLIC_REPO        owner/name of the public repository for the cards
  GITHUB_REPOSITORY  this repository (set by GitHub Actions)
Without the tokens it does nothing. Each decision day is posted once (runs/instagram_posted.txt). Instagram tokens
last 60 days, so on the 1st and 15th of each month the token is renewed and saved back as the INSTAGRAM_TOKEN secret.
Usage: python instagram_post.py [--log=rank_paper] [--dry-run]   (--dry-run: make the card and caption, post nothing)
"""
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from social_card import card

IG = "https://graph.instagram.com/" + os.environ.get("IG_API_VERSION", "v23.0")
GH = "https://api.github.com"
POSTED = "runs/instagram_posted.txt"


def call(url, data=None, method=None, headers=None):
    """JSON from an HTTP call; on failure, exit with the service's message (never the URL's query, which can hold
    a token)."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "GET":
            return None
        sys.exit(f"instagram: HTTP {e.code} from {url.split('?')[0]}: {e.read().decode(errors='replace')[:300]}")


form = lambda d: urllib.parse.urlencode(d).encode()


def github(path, payload=None, method="GET"):
    headers = {"Authorization": f"Bearer {os.environ['PUBLISH_TOKEN']}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"}
    return call(f"{GH}{path}", json.dumps(payload).encode() if payload is not None else None, method, headers)


def renew_token(token):
    """Renew the 60-day Instagram token and store the new one as this repository's INSTAGRAM_TOKEN secret."""
    from nacl import encoding, public
    new = call(f"https://graph.instagram.com/refresh_access_token?grant_type=ig_refresh_token&access_token={token}",
               method="GET")
    repo = os.environ["GITHUB_REPOSITORY"]
    key = github(f"/repos/{repo}/actions/secrets/public-key")
    sealed = public.SealedBox(public.PublicKey(key["key"].encode(), encoding.Base64Encoder())).encrypt(
        new["access_token"].encode())
    github(f"/repos/{repo}/actions/secrets/INSTAGRAM_TOKEN",
           {"encrypted_value": base64.b64encode(sealed).decode(), "key_id": key["key_id"]}, "PUT")
    print(f"instagram: token renewed, valid for {int(new.get('expires_in', 0)) // 86400} more days")
    return new["access_token"]


def caption(log):
    days = log["daily"]
    today, booked = days[-1], [r for r in days if r["taker"] is not None]
    pc = lambda v: f"{100 * v:+.2f}%"
    d = lambda s: pd.Timestamp(s).strftime("%d %b")
    lines = [f"Crypto ranking, {pd.Timestamp(today['date']).strftime('%a %d %b %Y')}",
             f"Long: {', '.join(today['long'])}",
             f"Short: {', '.join(today['short'])}",
             "(16.7% of capital each, filled at the 01:00 UTC close, held 1 day)", ""]
    if booked:
        lines += [f"{d(booked[-1]['date'])}: {pc(booked[-1]['taker'])} · since {d(log['start'])}: "
                  f"{pc(sum(r['taker'] for r in booked))} over {len(booked)} days",
                  "Net of taker fees and funding. Paper trading, no real money.", ""]
    lines += ["A LightGBM model ranks the 20 most-traded coins on Binance every day at 00:00 UTC: long the top 3, "
              "short the bottom 3.", "",
              "#crypto #cryptotrading #bitcoin #altcoins #machinelearning #quant #trading"]
    return "\n".join(lines)


if __name__ == "__main__":
    name = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--log=")), "rank_paper")
    dry = "--dry-run" in sys.argv
    token, public_repo = os.environ.get("INSTAGRAM_TOKEN"), os.environ.get("PUBLIC_REPO")
    if not dry and not (token and os.environ.get("PUBLISH_TOKEN") and public_repo):
        print("instagram: INSTAGRAM_TOKEN / PUBLISH_TOKEN / PUBLIC_REPO not set, nothing posted")
        sys.exit()
    with open(f"runs/{name}.json", encoding="utf-8") as fh:
        log = json.load(fh)
    date = log["daily"][-1]["date"]
    image, text = card(log, f"cards/{date}.jpg"), caption(log)
    if dry:
        print(f"{image}\n{text}")
        sys.exit()

    if dt.datetime.now(dt.UTC).day in (1, 15):
        try:
            token = renew_token(token)
        except SystemExit as e:                      # keep posting with the current token
            print(f"instagram: token renewal failed ({e})")
    posted = open(POSTED).read().strip() if os.path.exists(POSTED) else ""
    if date <= posted:
        print(f"instagram: nothing new since {posted}")
        sys.exit()

    # 1. The card at a public address.
    branch = github(f"/repos/{public_repo}")["default_branch"]
    path = f"/repos/{public_repo}/contents/cards/{date}.jpg"
    old = github(path)
    with open(image, "rb") as fh:
        payload = {"message": f"Card for {date}", "content": base64.b64encode(fh.read()).decode(), "branch": branch}
    if old:
        payload["sha"] = old["sha"]
    github(path, payload, "PUT")
    url = f"https://raw.githubusercontent.com/{public_repo}/{branch}/cards/{date}.jpg"
    for _ in range(12):                              # wait until the image is served
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                if r.status == 200:
                    break
        except urllib.error.URLError:
            pass
        time.sleep(5)

    # 2. Instagram: create the post, wait until Instagram has fetched the image, publish.
    me = call(f"{IG}/me?fields=user_id,username&access_token={token}", method="GET")
    uid = me.get("user_id") or me["id"]
    cid = call(f"{IG}/{uid}/media", form({"image_url": url, "caption": text, "access_token": token}), "POST")["id"]
    for _ in range(24):
        status = call(f"{IG}/{cid}?fields=status_code&access_token={token}", method="GET").get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            sys.exit("instagram: Instagram could not process the image")
        time.sleep(5)
    post = call(f"{IG}/{uid}/media_publish", form({"creation_id": cid, "access_token": token}), "POST")
    with open(POSTED, "w") as fh:
        fh.write(date)
    print(f"instagram: posted {date} to @{me.get('username', '?')} (media {post.get('id')})")

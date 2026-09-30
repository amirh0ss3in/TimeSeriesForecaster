# Daily crypto ranking: current setup

Every day at 00:00 UTC, a LightGBM model ranks the 20 most-traded coins by their expected return over the next day
relative to the coin average. The system buys the top 3 and shorts the bottom 3 on perpetual futures, half the capital
on each side, so the market's overall direction cancels out. The coin list is rebuilt every month from what was
known at the time, and every result below is out of sample, including the choice of settings.

## Results (Jan 2022 – Aug 2026, 56 months, net of fees and funding, no leverage)

| Trades filled | Maker fees (0.02%), mean month [95% range] | Taker fees (0.05%) [95% range] | Positive months | Worst drawdown |
|---|---|---|---|---|
| At the 00:00 close | +5.9% [+3.4, +8.4] | +5.0% [+2.5, +7.6] | 41/56 | −21% |
| **1 hour later** | **+5.4%** [+2.7, +8.3] | **+4.4%** [+1.5, +7.4] | 37/56 | −23% |
| 3 hours later | +5.0% [+2.1, +7.9] | +4.0% [+1.1, +7.1] | 34–37/56 | −27% |

- **Every year is positive** at every delay. With a 1-hour delay and taker fees: 2022 +6.7%, 2023 +5.5%,
  2024 +2.1%, 2025 +4.7%, 2026 (Jan–Aug) +2.7% a month. 2026 is the weakest year so far.
- **The goal is more than 2% net a month.** The average clears it at every delay and fee level; the lower end of the
  range with a 1-hour delay and taker fees (+1.5%) doesn't.
- **Profit is spread across many coins** (best: RUNE, NEAR, WAVES, ENA, BONK, ZIL; worst: WIF, FTM, OP, DOT, BOME).
- **With today's 20 largest coins instead** (chosen with hindsight), the result is similar: +5.6% a month (maker,
  no delay).
- **Not modelled:** slippage on the smaller coins in the list, and the gap between the spot prices used here and
  the futures prices actually traded. Paper trading measures both.

## How it works (`crypto_rank.py`)

- **Coins:** at the start of each month, the 20 coins with the highest USDT spot volume over the previous 30 days,
  among coins with 90+ days of history and a live perpetual (a funding payment in the last 30 days). 147 different
  coins pass through the list from 2021 to 2026, including ones that later collapsed or were delisted (LUNA, FTT,
  WAVES, ...).
- **Inputs (51):** from each coin's daily candles: moves over 1, 3, 7, 14, 30 and 60 days (also divided by 30-day
  volatility), 7- and 30-day volatility, volume and trade-count trend, aggressive-buy share over 1 and 7 days,
  where the close sits in the day's range and in the 30-day range; each also relative to that day's coin average and
  as a rank among the coins.
- **Target:** the next H days' log move relative to the coin average.
- **Model:** LightGBM regression (300 trees, learning rate 0.02, 15 leaves, 300 minimum rows per leaf, row sampling
  0.7, input sampling 0.5, L2 penalty 10), averaged over 5 random seeds. Retrained at the start of every month on
  all earlier rows, with an H-day gap.
- **Portfolio:** long the top K, short the bottom K, weight 0.5/K each; with H > 1, the average of the last H days'
  portfolios. Candidates H ∈ {1, 3, 7}, K ∈ {3, 5}; each month uses the pair with the best average net return
  over earlier months (maker fees). It chose **H = 1, K = 3** in every month since mid-2025.
- **Costs:** fees on every change in a coin's weight; each position pays (long) or receives (short) its coin's
  funding rate. A position earns weight × the coin's simple move.

## Data

```bash
python fetch_universe.py            # daily candles + funding for 268 coins ever in the top 40 by volume -> data/universe*/
python fetch_universe.py --hourly   # hourly closes for them, for the trade-delay test -> data/universe_1h/
python fetch_crypto.py              # hourly candles for today's 20 largest coins -> data/crypto/ (see below)
```

All data folders, `checkpoints/` and `external/` are git-ignored.

## How to run

```bash
python serve.py                                             # pages at http://localhost:8000 (no caching)
python crypto_rank.py --universe=20 --delay=1 --walk-from=2021-01-01 --nested-from=2022-01 --save=rank_pit20_d1
python crypto_rank.py --walk-from=2021-01-01 --nested-from=2022-01 --save=rank20_s5    # today's 20 coins instead
```

| Page | Shows |
|---|---|
| `backtest.html?run=rank_pit20_d1` | monthly returns against the goal, gain over time, the setting chosen each month |

## Paper trading (`rank_trader.py`, frozen)

The frozen system uses the setting the honest test chose every month since mid-2025: the point-in-time top 20, hold
1 day, long the top 3 and short the bottom 3 (16.7% of capital each), filled at the 01:00 UTC close. It imports the
same functions as `crypto_rank.py`, and each month's models are trained once and saved in `checkpoints/rank_trader/`.

```bash
python rank_trader.py update     # latest candles, hourly closes and funding from Binance's public API
python rank_trader.py step       # today's decision (after 00:00 UTC); books earlier days -> runs/rank_paper.json
python rank_trader.py check      # replays Jul 2026 and compares it with the backtest
python rank_trader.py replay --from=2026-09-01 --to=2026-09-26 --log=rank_replay_sep
```

**The paper test runs on GitHub Actions** (`.github/workflows/paper-trader.yml`), starting with the decision for
2026-09-28:
- **Schedule:** it runs every day at 01:05 UTC (GitHub may start it a little late).
- **Data:** it runs `update`, then `step`. `step` decides every day not logged yet, so a missed run is caught up
  the next day, and days already logged are never recomputed.
- **Commits:** it commits `runs/rank_paper.json` (the log), `runs/paper_run.log` (the last run's output) and each
  new month's frozen models to `main`. Run `git pull` before viewing them or changing anything.
- **Addresses:** GitHub's servers are in the US, where api.binance.com and fapi.binance.com refuse connections. The
  workflow therefore reads the same data from data-api.binance.vision and www.binance.com (`BINANCE_SPOT`,
  `BINANCE_FUTURES`).
- **Telegram:** each new decision, with the latest booked result and the running total, is posted to a Telegram
  channel by `telegram_post.py`, once the repository secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` exist.
  Preview the message with `python telegram_post.py --dry-run`.
- **Instagram:** each new decision is also posted as a 1080×1350 image card (`social_card.py`) with a caption, by
  `instagram_post.py`. Instagram only publishes images from a public address, so the card is first committed to the
  public repository `PUBLIC_REPO` (cards/<date>.jpg). Secrets:
  - `INSTAGRAM_TOKEN`: a long-lived Instagram token (Instagram API with Instagram Login).
  - `PUBLISH_TOKEN`: a fine-grained GitHub token with Contents write on the public repository and Secrets write on
    this one. The Instagram token lasts 60 days, so it is renewed on the 1st and 15th of each month and saved back
    into `INSTAGRAM_TOKEN`.

  Preview with `python instagram_post.py --dry-run`.
- **Cache:** market data is kept in the Actions cache. If the cache is gone (e.g. evicted after 7 days unused), the
  workflow rebuilds it from Binance's archive and first reruns `check`.

Each decision uses only data from before that day's 00:00 UTC, and the paper fill is the 01:00 close, so the same
log can also be produced on the laptop (`update`, then `step --from=2026-09-28`). Don't run both, or the two logs
will diverge. The log is at `backtest.html?run=rank_paper`: daily positions, net per day with maker and taker fees,
and gain over time.

Checks passed:
- `check`: the frozen code reproduces the backtest's July 2026 result exactly (−2.59% maker, −3.46% taker).
- **September 2026 (never used by the ranking work), Sept 1–26:** +14.9% with maker fees, +14.1% with taker fees,
  18 of 26 days positive (`backtest.html?run=rank_replay_sep`). One month is a small sample; the backtest's months
  range from about −10% to +20%.
- `update` keeps archive rows as they are and only appends new days. A coin not on disk is added once it is in the
  top 40 by 24-hour volume.

Judge the paper test after a few months against the 2% goal, with taker fees, without changing anything.

## The hourly direction pipeline (earlier, still in the root)

`crypto_features.py`, `crypto_probe.py`, `crypto_backtest.py`, `crypto_nested.py` and `leak_check.py` predict each
coin's next hour from 46 inputs (candle shape, order flow, the last minutes of the hour). The edge is real but thin
(52.9% of hours right, Jan 2024 – Aug 2026) and shrinking, and trading it hourly lost 0.7% a month once the
settings were also chosen out of sample (`backtest.html?run=nested`, `direction.html?run=lgbm_w24`).

## How we got here (details in `archive/README.md`)

1. **Autoregressive forecasting on sin, trend, amplitude, sunspots and CO₂.** Per-window normalization lets a model
   extrapolate a trend; an ensemble fixes a single model's overconfidence; CRPS measures forecast quality.
2. **One model for many series, and sequential fine-tuning with replay.**
3. **BTC hourly return distribution.** Matched but didn't beat the recent-volatility baseline.
4. **Next-hour direction (LightGBM and transformers).** About 53% on 2025, 51.8% on 2026; the edge lasts an hour.
5. **Market LLM.** 52.5% / 51.1%; lost money traded hourly.
6. **Hourly trading system.** Profitable on development data, −0.04% on the Sept 2026 holdout, and −0.7% a month
   in the honest month-by-month test.
7. **Zero-shot Kronos.** A coin flip on the next hour.
8. **Daily ranking.** Moves over a day are large next to fees, and long-short cancels the market. With today's 20
   coins it made +4.4% a month, but ZEC and TAO (in the list because they rallied) drove much of it; a coin list
   rebuilt monthly from past volume, seed averaging, and a fix to the return calculation (weight × simple move,
   not log move, which had inflated crash days) gave the results above.

# Hourly crypto direction trading: current setup

A LightGBM model predicts whether each of 20 large coins will go up or down in the next hour. A trading rule turns
those predictions into long or short positions on perpetual futures, held until the signal fades. Everything below
is leak-tested and runs from this folder.

## Results (net of fees, no leverage)

| Period | Role | Monthly net, maker fees | 95% range | Taker fees | Hit rate on trades |
|---|---|---|---|---|---|
| 2025 | development | **+5.9%** | [+0.1, +10.6] | +3.8% | 56% |
| 2026 Jan–Aug, retrained monthly | development | **+3.5%** | [+0.7, +7.1] | +2.4% | 59% |
| **2026 Sep 1–26** | **one-time holdout** | **−0.04%** | [−3.4, +2.7] | −0.5% | 68% |

The goal is more than 3% net per month. **It is met on the development months, not yet on the holdout.** Read
these results with the following in mind:

- **The setting was chosen as the best of 176 combinations** on the development months, so those numbers are
  optimistic.
- **The chosen setting is an isolated peak.** An entry threshold of 0.10 or 0.15 instead of 0.12 loses money on the
  development data (`python crypto_diagnose.py`). A stable edge would degrade gradually around its best setting.
- **Months swing widely,** from −17% (Oct 2025) to +18% (Mar 2025); 16 of 20 development months were positive.
  Capital is often concentrated on 1–2 coins, and the worst drawdown in 2025 was 30%.
- **September was quiet.** The system held positions for only 12 hours. Its calls were right 68% of the time, but
  the fees from re-splitting capital between coins that signalled at once used up the gross gain of +0.23%.
- **What is solid:** the next-hour direction edge itself. The model is right about 52% of the time over all hours
  and 55–59% on its confident calls, with ranges clear of 50% on every period tested. The per-trade result after
  fees is also positive. Whether that adds up to 3% a month is the open question, and only paper trading on new
  data can answer it.

## Data (`fetch_crypto.py`)

Binance's public archive (data.binance.vision), for 20 coins in market-cap order: BTC ETH BNB XRP SOL TRX ZEC DOGE
LINK ADA XLM BCH NEAR UNI LTC SUI AVAX HBAR TAO SHIB. HYPE, XMR and TON have no archive, so they are skipped.

```bash
python fetch_crypto.py                        # spot 1h candles     -> data/crypto/        (2017-08 ..)
python fetch_crypto.py --interval=5m          # spot 5m candles     -> data/crypto_5m/     (last minutes of each hour)
python fetch_crypto.py --market=perp          # perp futures 1h     -> data/crypto_perp/   (not used by the model)
python fetch_crypto.py --market=premium       # perp premium index  -> data/crypto_premium/ (not used by the model)
python fetch_crypto.py --market=funding       # perp funding rates  -> data/crypto_funding/ (backtest costs)
python fetch_crypto.py --daily                # merge this month's daily files (also with --interval=5m, --market=...)
```

All data folders and `checkpoints/` are git-ignored.

## Features (`crypto_features.py`)

One row per coin and hour t. Every input is known once candle t has closed, and targets are the next candle. Moves
are divided by a causal volatility estimate (an exponentially weighted average of past squared moves, λ = 0.94), so
coins are comparable. The 46 inputs (`INPUTS`) are:

- **Own moves:** the last 1, 2, 3 and 4 hours; sums over 6, 24 and 168 hours; move relative to the market.
- **Candle shape:** where the close sits in the high–low range (the strongest input), wicks, and the share of up
  hours over the last 6.
- **Range and activity:** high–low range, volume and trade count relative to the past week, and the volatility
  regime.
- **Order flow:** aggressive-buy share of volume, for the last hour, the hour before and the last 6 hours.
- **Last minutes of the hour (from 5m candles):** moves over the last 5 and 15 minutes, 5-minute volatility, and
  order flow and volume share of the last 15 minutes.
- **Context:** market average, BTC and ETH versions of the above, hour of day, weekday.

`python leak_check.py 2026-09-10T05:00` rebuilds the table from data cut off at that date and confirms that no input
of any earlier hour changes. It passes for all 55 inputs (46 used, plus 9 unused futures inputs).

## Model (`crypto_probe.py`, frozen in `trader.py`)

- **Model:** a LightGBM binary classifier for "next hour up". Settings: 278 trees (early stopping on Oct–Dec 2024,
  × 1.1), learning rate 0.02, 15 leaves, 2000 minimum rows per leaf, row sampling 0.5, input sampling 0.7, L2
  penalty 10. One model is shared by all 20 coins.
- **Retraining:** at the start of every month, on every row before it (with a 25-hour gap before the month).
- **Validation (2025):** 52.9% of hours right, 56.0% on the most confident 20%, 53.2% on big moves. Monthly
  retraining over 2026 Jan–Aug: 51.8% of hours right, 55.0% on the most confident 20%.

## Trading rule (`crypto_backtest.py`, `trader.py`)

- **Edge** = P(up) − 0.5. A coin goes **long or short when |edge| ≥ 0.12**, and **stays in while the edge keeps
  its sign and is at least 0.03** (0.25 × entry); otherwise it's flat.
- **Allocation, current rule:** capital is split equally among the coins that have a position. Paper trading logs
  three alternatives side by side: `active_cap2` (at most 50% per coin), `fixed5` (1/5 per position, at most 5
  open, never re-split) and `slices` (1/20 per coin).
- **Costs:** every change in a coin's capital weight pays fees (maker 0.02% or taker 0.05% per side, Binance USD-M
  futures). Positions open at a funding time pay or receive that coin's funding rate.
- **Accounting:** the decision at hour t uses only candles closed by t + 1h, and earns the move of the next candle.

## How to run

```bash
python serve.py                                             # pages at http://localhost:8000 (no caching)
python crypto_features.py                                   # build the feature table (cached)
python leak_check.py 2026-09-10T05:00                       # leak test
python crypto_probe.py --obj=binary --walk --save=lgbm_plain   # 2025 validation + monthly retraining over 2026
python crypto_backtest.py --dev=lgbm_plain,lgbm_plain_walk --alloc=active   # development backtest (no holdout)
python crypto_diagnose.py                                   # sizing and costs, profit by coin and month, neighbouring settings
python trader.py check                                      # frozen script reproduces the backtest's predictions
python trader.py replay --from=2026-09-01 --to=2026-09-27 --log=paper_replay_sep   # replay through the live code path
python trader.py step                                       # one live hourly decision -> runs/paper_log.json
```

| Page | Shows |
|---|---|
| `direction.html?run=lgbm_plain` | direction accuracy: summary, per coin, by confidence, calibration |
| `backtest.html?run=bt_final_holdout` | backtest: monthly returns against the 3% goal, equity, threshold grid |
| `paper.html?run=paper_log` | paper trading: equity per variant, monthly returns, hourly log (demo: `?run=paper_replay_sep`) |

Checks passed so far:
- `trader.py check` reproduces the backtest's September predictions exactly.
- A September replay through `trader.py` reproduces the backtest's holdout result exactly.
- A live step that rebuilds features from data cut at the decision time gives identical probabilities.

## Before paper trading

- **Live data feed (not done yet).** The archive publishes each day's files the following day, so hourly decisions
  need the latest 1h and 5m candles from Binance's REST API, with the same columns. After that, schedule
  `python trader.py step` a few minutes after every full hour.
- **Rules for the paper test.** Run it frozen: no changes to the model, threshold or features. Judge it after a few
  months against the 3% goal, per allocation variant, with both fee levels.

## How we got here (details in `archive/README.md`)

1. **Autoregressive forecasting on sin, trend, amplitude, sunspots and CO₂.** Per-window normalization lets a model
   extrapolate a trend; a single model's σ is overconfident; an ensemble fixes that; CRPS measures forecast
   quality.
2. **One model for many series, and sequential fine-tuning with replay.** Training on many series had trouble
   learning each one well; sequential fine-tuning worked but forgot earlier series without replay.
3. **BTC hourly return distribution.** It matched but didn't beat the recent-volatility baseline, and showed no
   direction edge from price alone.
4. **Direction models (LightGBM and transformer).** Candle shape, order flow and the last minutes of each hour gave
   about 53% on 2025 and about 51.8% on 2026. The edge lasts about one hour.
5. **Market LLM (one step per hour for all coins).** 52.5% on 2025 and 51.1% on 2026. Its distributions beat the
   volatility baseline slightly, but it lost money when traded hourly.
6. **Trading.** Hourly flipping loses to fees. Holding a position until the signal fades, with capital
   concentrated on active signals, is what turned the edge into development profits.

# Archive

Experiments that led to the current setup but aren't part of it. They're kept so they can be rerun and compared.
The findings are summarized in `../SETUP.md` under *How we got here*.

| File | What it was |
|---|---|
| `train.py` | Single-series forecaster: autoregressive transformer ensemble with sampled paths, on sin / trend / amp / sunspots / CO₂ |
| `universal.py`, `sequential.py`, `data.py` | One model for many series, and sequential fine-tuning with replay |
| `model.py` | Stacked-weight transformer ensemble used by the scripts above |
| `crypto_train.py` | BTC hourly-return distribution model (tied the volatility baseline) |
| `crypto_seq.py`, `crypto_combine.py` | Next-hour direction transformer (about 53% on 2025, 51.8% on 2026), and averaging models |
| `market_llm.py` | "Market LLM": one step per hour for all 20 coins, Student-t moves (52.5% on 2025, 51.1% on 2026) |
| `monitor.html`, `crypto.html`, `dummy_train.py` | Monitor pages for the runs above, and a monitor test |
| `trader.py`, `paper.html` | Frozen hourly trading system (LightGBM, entry 0.12, exit 0.25 × entry) and its paper-trading page. Development months +5.9% / +3.5% a month, Sept 2026 holdout −0.04%, and the honest month-by-month test lost money, so it was retired |
| `crypto_diagnose.py` | Diagnostics for that system: sizing and costs, profit by coin and month, neighbouring thresholds |
| `kronos_test.py` | Zero-shot Kronos-small (80-hour context, 10 samples) on next-hour direction: 50.2% hit on Jul 2025 – Aug 2026 (after its weights were uploaded) against 53.3% for LightGBM on the same hours. Needs the repo cloned to `../external/Kronos` |
| `chronos_test.py` | Same test for Chronos-Bolt; never run (replaced by Kronos) |
| `fetch_crypto.py`, `crypto_features.py`, `leak_check.py` | Hourly candles (spot, 5m, perp, premium, funding) for today's 20 largest coins, the 55-input hourly feature table, and its truncation leak test |
| `crypto_probe.py`, `crypto_eval.py` | LightGBM next-hour direction model (52.9% of hours right, Jan 2024 – Aug 2026) and its scoring |
| `crypto_backtest.py`, `crypto_nested.py` | Hourly trading backtest (entry/exit thresholds, allocations, fees, funding) and its honest month-by-month test (−0.7% a month) |
| `direction.html` | Direction-accuracy page (`?run=lgbm_w24`) |
| `monitor.py` | Shared helpers for the scripts above (`Monitor`, `crps`, `write_json`) |
| `runs/rank20_s5.json` | Daily ranking with today's 20 largest coins (chosen with hindsight): +5.6% a month (maker, no delay) |
| `data/` | Sunspot and CO₂ series used by `train.py` |
| `runs/` | Their result files |

**Running an archived script.** Run it from the repository root, so the data folders are found. The archived modules
it imports sit next to it:

```bash
python archive/train.py amp
```

New results are written to the root `runs/` folder. The archived pages read `archive/runs/`, so move a new result
file there to view it at `http://localhost:8000/archive/monitor.html?run=NAME` (or `archive/direction.html`). An
archived backtest result can be opened with the root page as `backtest.html?run=../archive/runs/nested`.

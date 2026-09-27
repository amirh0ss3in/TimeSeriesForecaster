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
| `runs/` | Their result files |

**Running an archived script.** Run it from the repository root so that the root modules (`crypto_features.py`,
`crypto_eval.py`, `monitor.py`) and `data/` are found:

```bash
PYTHONPATH=. python archive/train.py amp
```

New results are written to the root `runs/` folder. The archived pages read `archive/runs/`, so move a new result
file there to view it at `http://localhost:8000/archive/monitor.html?run=NAME`.

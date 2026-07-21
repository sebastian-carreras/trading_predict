# Hosted demo — trading_predict

Showcase of the ML trading MLOps system, deployed on **Hugging Face Spaces (Static SDK)**.
It's a **thin reader**: the page loads only the pre-computed `assets/` — no backend, no torch,
no MLflow, no secrets.

**▶️ Live:** https://huggingface.co/spaces/chatoxz/trading-predict-demo

## Architecture

- **`bundle_assets.py`** — runs against the full repo. Reads `models/registry.json`, the
  walk-forward predictions and backtests of every champion, the generated leaderboard and a set
  of curated figures → produces a self-contained `assets/` bundle (~17 MB). Re-run it after each
  retrain to refresh the demo.
- **`index.html`** — **the deployed app**: a static page (HTML + JS + Plotly.js from CDN) that
  fetches `assets/` and builds the five tabs (Overview, Performance with interactive drilldown,
  Champion vs Baseline, Drift & Stability, Methodology).
- **`app.py`** (Gradio) and **`app_streamlit.py`** (Streamlit) — equivalent alternate frontends,
  in case this is deployed elsewhere. Neither is used by the static Space.
- **`assets/`** — bundled data and figures. Generated, never edited by hand.

## Running locally

```bash
python -m demo.bundle_assets           # (re)generate assets/ from the full repo
cd demo && python -m http.server 8899  # serve statically → http://127.0.0.1:8899/index.html
```

## Why Static and not Gradio

Hugging Face changed its policy: on the free tier, **Gradio Spaces run on ZeroGPU** (gated,
requires PRO for CPU-basic) and **Docker is paid**. **Static is free without PRO** — and since
this demo is a read-only dashboard, plain HTML + JS fits it perfectly.

## Deploying to Hugging Face Spaces (Static SDK, free)

1. **huggingface.co/new-space** → SDK **Static**, template **Blank**, `apache-2.0`, public.
2. The Space repo needs `index.html`, `assets/`, and a `README.md` carrying `sdk: static` in its
   YAML front-matter — see `SPACE_README.md`.
3. PNG figures must be versioned through **git-LFS** (HF rejects binaries in plain git):
   run `git lfs install && git lfs track "*.png"` before committing, or
   `git lfs migrate import --include="*.png"` if they're already committed.
4. `git push`, authenticating with your username and a **Write token** from
   huggingface.co/settings/tokens.

Static builds in seconds since it only serves files — no secrets, no ZeroGPU.

## Honesty of the numbers

- The Overview medians are **recomputed live** from the registry; they are never hardcoded from
  the README.
- **E3 (intraday)** is displayed as a **negative result** (Sharpe < 0, doesn't beat costs). It's
  reported, not hidden.
- The synthetic test ticker (`AAA`) is filtered out of the bundle.

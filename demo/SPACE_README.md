---
title: trading_predict — ML Trading MLOps Demo
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: MLOps trading system demo (GRU/LSTM) — not financial advice
---

# trading_predict — ML Trading MLOps Demo

Interactive showcase of an end-to-end **MLOps trading system** built with
deep learning (GRU/LSTM) across three strategies (E1 conservative, E2 moderate,
E3 intraday).

**What it demonstrates**
- **45 champion models** managed through a model registry with champion/challenger
  promotion and permissive guardrails.
- **Walk-forward cross-validation with embargo** and train-only z-scoring — an
  anti-leakage methodology reported honestly.
- **Champion-vs-baseline** comparison, **drift & stability** monitoring (KS/PSI),
  and a per-ticker interactive drilldown (equity curves net of costs, predicted
  vs actual returns).
- An **honest negative result**: the E3 intraday strategy does not beat transaction
  costs, and is shown as such rather than hidden.

All figures are **out-of-sample** walk-forward results read from pre-computed
artifacts. This is a **portfolio project** (CEIA-FIUBA final work) — **not a trading
tool and not financial advice.**

Source code: https://github.com/sebastian-carreras/trading_predict

---

## Deploy notes (why Static and not Gradio)

This Space runs as **`sdk: static`** — HTML + JS + Plotly.js served from
`demo/index.html`, with assets pre-computed by `demo/bundle_assets.py`.

It was originally built as a **Gradio** app (`demo/app.py`), but Hugging Face moved
the Gradio free tier behind PRO (ZeroGPU) and the Docker SDK is paid, so it was
pivoted to Static while reusing 100% of the assets. `app.py` (Gradio) and
`app_streamlit.py` remain in the repo as alternate frontends — they are **not**
what gets deployed.

That's why the front-matter above carries **no** `sdk_version` and no `app_file`:
static Spaces don't use them. If this ever moves back to Gradio, both need to be
restored, using the `sdk_version` Hugging Face offers when creating the Space
rather than one pinned by hand.

Two other things that cost time: binaries have to go through **git-LFS**, and
`short_description` has a **60-character limit** — hence the terse one above.

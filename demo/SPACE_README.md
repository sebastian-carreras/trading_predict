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

## Nota de deploy (por qué Static y no Gradio)

Este Space corre como **`sdk: static`** — HTML + JS + Plotly.js servidos desde
`demo/index.html`, con los assets pre-computados por `demo/bundle_assets.py`.

Originalmente se construyó como app **Gradio** (`demo/app.py`), pero HF dejó el
free tier de Gradio detrás de PRO (ZeroGPU) y el SDK Docker es pago, así que se
pivoteó a Static reutilizando el 100% de los assets. `app.py` (Gradio) y
`app_streamlit.py` quedan en el repo como frontends alternativos, **no** como lo
que se despliega.

Por eso el front-matter de arriba **no** lleva `sdk_version` ni `app_file`: los
Spaces estáticos no los usan. Si alguna vez se vuelve a Gradio, hay que
reponerlos y usar el `sdk_version` que HF ofrezca al crear el Space (no uno
fijado a mano).

Otros detalles que costaron en su momento: los binarios van por **git-LFS**, y
`short_description` tiene un límite de **60 caracteres** (por eso es corta).

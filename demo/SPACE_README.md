---
title: trading_predict — ML Trading MLOps Demo
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
license: apache-2.0
pinned: false
short_description: End-to-end MLOps trading system (GRU/LSTM) with walk-forward validation, model registry, and drift monitoring. Portfolio demo — not financial advice.
---

# trading_predict — ML Trading MLOps Demo

Interactive Gradio showcase of an end-to-end **MLOps trading system** built with
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

> **Nota:** si al crear el Space HF puso un `sdk_version` distinto en su README
> auto-generado, conservá ESE valor (es el que su plataforma soporta) en vez del
> `5.49.1` de arriba.

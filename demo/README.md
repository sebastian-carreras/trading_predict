# Demo hosteada — trading_predict

Showcase del sistema MLOps de trading con ML, desplegado en **Hugging Face Spaces
(SDK Static)**. **Lector fino**: la página lee solo `assets/` (pre-calculado), sin
backend, sin torch/mlflow/secretos.

**▶️ En vivo:** https://huggingface.co/spaces/chatoxz/trading-predict-demo

## Arquitectura

- **`bundle_assets.py`** — corre en el repo completo. Lee `models/registry.json`,
  las predicciones y backtests walk-forward de cada champion, el leaderboard ya
  generado y figuras curadas → produce `assets/` self-contained (~17 MB).
  Re-correlo después de cada retrain para refrescar la demo.
- **`index.html`** — **la app desplegada**: página estática (HTML + JS + Plotly.js
  desde CDN) que lee `assets/` por fetch y arma las 5 tabs (Overview, Performance
  con drilldown interactivo, Champion vs Baseline, Drift & Estabilidad, Metodología).
- **`app.py`** (Gradio) y **`app_streamlit.py`** (Streamlit) — front-ends alternativos
  equivalentes, por si se despliega en otro entorno. No se usan en el Space Static.
- **`assets/`** — datos + figuras del bundle (regenerados, no editar a mano).

## Correr local

```bash
python -m demo.bundle_assets          # (re)genera assets/ desde el repo completo
cd demo && python -m http.server 8899 # servir estático → http://127.0.0.1:8899/index.html
```

## Por qué Static (y no Gradio)

HF cambió su política: en el free tier los Spaces **Gradio corren en ZeroGPU**
(gateado, requiere PRO para CPU-basic) y **Docker es pago**. **Static es gratis sin
PRO** — y como la demo es un dashboard de solo-lectura, encaja perfecto en HTML+JS.

## Deploy a Hugging Face Spaces (SDK: Static, gratis)

1. **huggingface.co/new-space** → SDK **Static**, template **Blank**, `apache-2.0`, público.
2. El repo del Space necesita: `index.html`, `assets/` y un `README.md` con
   `sdk: static` en la metadata YAML (ver `SPACE_README.md`).
3. Las figuras PNG se versionan por **git-LFS** (HF rechaza binarios en git normal):
   `git lfs install && git lfs track "*.png"` antes de commitear, o
   `git lfs migrate import --include="*.png"` si ya están commiteadas.
4. `git push` (auth: usuario + **token Write** de huggingface.co/settings/tokens).

Static buildea en segundos (solo sirve archivos), sin secretos ni ZeroGPU.

## Honestidad de las cifras

- Las medianas del Overview se **recalculan en vivo** desde el registry (no se
  hardcodean de `README.md`).
- **E3 (intradía)** se muestra como **resultado negativo** (Sharpe < 0, no supera
  costos): se reporta, no se esconde.
- El ticker sintético de test (`AAA`) se filtra del bundle.

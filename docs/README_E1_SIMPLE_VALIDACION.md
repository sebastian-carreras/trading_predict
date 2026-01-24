# E1 Simple - Validacion basica (1 ticker)

Guia rapida para verificar que el pipeline E1 Simple funciona de forma reproducible con un ticker fijo (ejemplo: AAPL).

## Objetivo
- Usar un dataset limpio fijo.
- Ejecutar el pipeline sin descarga ni limpieza.
- Revisar metricas clave y confirmar reproducibilidad con dos corridas iguales.

## Prerrequisitos
- Dataset limpio presente: `data/clean/AAPL_daily.csv`.
- Config base: `src/config/base.yaml`.
- Semilla del proyecto: `project.seed = 42` (ya en el config).

## Paso 1: Snapshot opcional del dataset
Para fijar insumo:
```bash
mkdir -p data/snapshots
cp data/clean/AAPL_daily.csv data/snapshots/AAPL_daily.csv
```

## Paso 2: Check rapido de integridad del CSV
```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('data/clean/AAPL_daily.csv')
df = pd.read_csv(p, parse_dates=['timestamp']).sort_values('timestamp')
assert df['timestamp'].is_monotonic_increasing, 'Fechas no ordenadas'
assert df['close'].isna().sum() == 0, 'NaNs en close'
gaps = df['timestamp'].diff().dt.days.gt(7).sum()
print('Filas:', len(df), '| Gaps >7d:', gaps)
print('Min:', df['timestamp'].min(), '| Max:', df['timestamp'].max())
PY
```

## Paso 3: Correr el pipeline (sin descarga/limpieza)
```bash
PYTHONHASHSEED=42 /opt/anaconda3/bin/conda run -p /opt/anaconda3 --no-capture-output \
  python -m src.train_e1_simple_pipeline \
  --tickers AAPL \
  --config src/config/base.yaml \
  --skip-download \
  --skip-cleaning
```
Outputs en `runs/e1_simple/<timestamp>/`.

### Ejemplo de resultados (run: 20260124_175222)
- ML: MAE=0.1632, RMSE=0.1866, IC=0.270, DirAcc=50.8%.
- Backtest: total_return=15.19%, CAGR=12.31%, Sharpe=0.68, Sortino=0.68, max_drawdown=18.96%, Calmar=0.65, profit_factor=1.18, hit_rate=53.99%, trades=3, avg_turnover=0.0098, costs=0.0015, time_in_market=53.4%.
- Decision score=0.818 (threshold 0.70) -> BUY.

## Paso 4: Repetir para reproducibilidad
Ejecuta el mismo comando del paso 3 una segunda vez. Debes obtener metricas identicas o casi identicas. Para comparar los dos `summary_all.csv`:
```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
runs = [
    Path('runs/e1_simple/20260124_175222/summary_all.csv'),
    Path('runs/e1_simple/20260124_175337/summary_all.csv'),
]
for p in runs:
    df = pd.read_csv(p)
    print(p)
    print(df.to_string(index=False))
    print('-'*60)
PY
```

## Paso 5: Revisar artefactos
- `summary_all.csv`: metricas agregadas.
- `AAPL/AAPL_predictions.csv`: y_true vs y_pred en test.
- `AAPL/AAPL_backtest.csv`: equity, signals, turnover, costs.
- `AAPL/AAPL_model.pth`: modelo GRU guardado.
- `AAPL/AAPL_scaler.csv`: medias y desvios de features.

## Paso 6: Checks rapidos de calidad
- Sharpe y CAGR positivos y consistentes con los targets del decision score.
- max_drawdown razonable (<~20% en este ejemplo).
- hit_rate y profit_factor > 1 en el backtest.
- Sin NaNs en predicciones ni en backtest.

## Troubleshooting
- Si las metricas difieren entre corridas, fija determinismo adicional en PyTorch (ej. `torch.use_deterministic_algorithms(True)` y `CUBLAS_WORKSPACE_CONFIG=:16:8` antes de correr).
- Si faltan datos: vuelve a correr limpieza o usa el snapshot en `data/snapshots/`.

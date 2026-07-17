# E4 Pairs Trading - Descubrimiento Automático de Pares

## 🎯 Nuevo Enfoque

En lugar de definir pares manualmente, el sistema ahora **descubre automáticamente** qué activos están cointegrados analizando todas las combinaciones posibles.

## 🔄 Flujo del Pipeline

```
1. Download Data (universo completo)
   ↓
2. Clean Data
   ↓
3. Discover Cointegrated Pairs
   ├─ Probar todas las combinaciones (N choose 2)
   ├─ Test de cointegración (Engle-Granger)
   ├─ Calcular parámetros OU (half-life)
   ├─ Evaluar correlación y estabilidad
   ├─ Scoring de calidad
   └─ Guardar en MLflow
   ↓
4. Process Pairs (solo cointegrados)
   ↓
5. Generate Report
```

## ⚙️ Configuración

En `src/config/base.yaml`:

```yaml
e4_pairs:
  discovery:
    enabled: true  # Activar descubrimiento automático
    pvalue_max: 0.05  # P-value máximo para cointegración
    half_life_min: 5.0  # Half-life mínimo en días
    half_life_max: 60.0  # Half-life máximo en días
    correlation_min: 0.5  # Correlación mínima
    max_pairs: 20  # Máximo de pares a procesar
    min_quality_score: 0.5  # Score mínimo de calidad

  universe:  # Lista de tickers a analizar
    - "GGAL.BA"
    - "BMA.BA"
    - "PAMP.BA"
    - "CEPU.BA"
    # ... más tickers
```

## 📊 Scoring de Calidad

Cada par descubierto recibe un **quality_score** (0-1) calculado como:

```python
quality_score =
    (1 - pvalue) * 0.4 +                    # 40% - Fuerza de cointegración
    (1 - correlation_stability) * 0.3 +      # 30% - Estabilidad
    min(30 / half_life, 1.0) * 0.3          # 30% - Half-life (ideal ~30 días)
```

**Score más alto = Mejor par para trading**

## 🚀 Uso

### Opción 1: Airflow DAG

```bash
# Trigger con descubrimiento automático (default)
airflow dags trigger e4_pairs_trading_pipeline

# O con parámetros custom
airflow dags trigger e4_pairs_trading_pipeline --conf '{
  "use_discovery": "True"
}'
```

### Opción 2: Script de prueba

```bash
# Ejecutar descubrimiento rápido
python scripts/test_pair_discovery.py
```

### Opción 3: Python directo

```python
from pathlib import Path
from src.pairs.discover_pairs import discover_cointegrated_pairs

tickers = ["GGAL.BA", "BMA.BA", "PAMP.BA", "CEPU.BA", ...]
data_dir = Path("data/clean")

pairs_df = discover_cointegrated_pairs(
    tickers,
    data_dir,
    pvalue_max=0.05,
    half_life_max=60.0,
    verbose=True
)

print(f"Descubiertos {len(pairs_df)} pares cointegrados")
print(pairs_df[['ticker_a', 'ticker_b', 'pvalue', 'ou_half_life', 'quality_score']])
```

## 📈 Resultados en MLflow

### Experiment: `E4_Pairs_Discovery`

**Run: Pair_Discovery**
- Parámetros:
  - `num_tickers`: Cantidad de tickers analizados
  - `pvalue_max`: Umbral de cointegración
  - `half_life_max`: Half-life máximo
  - `correlation_min`: Correlación mínima

- Métricas:
  - `pairs_discovered`: Total de pares encontrados
  - `pairs_selected`: Pares seleccionados para trading
  - `avg_pvalue`: P-value promedio
  - `avg_half_life`: Half-life promedio
  - `best_quality_score`: Mejor score de calidad

- Artifacts:
  - `discovered_pairs_YYYYMMDD_HHMMSS.csv`: Todos los pares encontrados
  - `selected_pairs_YYYYMMDD_HHMMSS.csv`: Pares seleccionados

### Experiment: `E4_Pairs_Trading_Strategy`

Luego cada par seleccionado se procesa con su propio run registrando:
- Métricas de backtest (sharpe, return, drawdown, etc.)
- Parámetros OU
- Artifacts (trades, spreads, etc.)

## 📁 Archivos Generados

```
runs/e4_pairs/
├── discovery/
│   ├── discovered_pairs_20260109_120000.csv  # Todos los pares
│   └── selected_pairs_20260109_120000.csv    # Pares seleccionados
└── airflow_20260109_120000/
    ├── PAMP.BA_CEPU.BA/
    │   ├── spread_timeseries.csv
    │   ├── trades.csv
    │   └── backtest_summary.json
    └── TGSU2.BA_TGNO4.BA/
        └── ...
```

## 🎯 Ventajas vs Enfoque Manual

| Aspecto | Manual | Automático |
|---------|--------|------------|
| Definición de pares | Requiere conocimiento previo | Descubre automáticamente |
| Cobertura | Solo pares conocidos | Todas las combinaciones |
| Actualización | Manual mensual | Automática cada run |
| False positives | Ninguno (pre-validados) | Filtrados por quality_score |
| Escalabilidad | Limitada | N choose 2 combinaciones |

## ⚠️ Consideraciones

### Complejidad Computacional

Para N tickers:
- Combinaciones posibles: **N × (N-1) / 2**
- Ejemplo: 30 tickers → **435 pares** a probar

**Tiempo estimado**: ~1-2 segundos por par
- 435 pares → ~7-15 minutos

### Modo Manual (fallback)

Si `discovery.enabled = false`, usa pares fijos:

```yaml
e4_pairs:
  discovery:
    enabled: false

  pairs:  # Se usan estos
    - ["PAMP.BA", "CEPU.BA"]
    - ["TGSU2.BA", "TGNO4.BA"]
```

## 🔧 Troubleshooting

### No se encuentran pares

Causa común: Filtros muy restrictivos

**Solución**: Relajar parámetros
```yaml
discovery:
  pvalue_max: 0.10  # Aumentar de 0.05
  half_life_max: 90.0  # Aumentar de 60
  correlation_min: 0.3  # Reducir de 0.5
```

### Demasiados pares

Causa: Universo muy grande

**Solución**: Reducir universo o usar filtros más estrictos
```yaml
discovery:
  pvalue_max: 0.01  # Más estricto
  min_quality_score: 0.7  # Solo mejores
  max_pairs: 10  # Limitar cantidad
```

## 📚 Referencias

- **Cointegración**: Engle-Granger (1987)
- **Ornstein-Uhlenbeck**: Proceso estocástico de reversión a la media
- **Pairs Trading**: Gatev et al. (2006)

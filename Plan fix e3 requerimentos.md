# Plan — Cierre de los WARN de E3 en la matriz de requerimientos

## Contexto

La matriz [reports/conclusiones/tablas/requirements_validation.csv](../../proyecto_final/trading_predict/reports/conclusiones/tablas/requirements_validation.csv), generada por [scripts/evaluation/requirements_validation.py](../../proyecto_final/trading_predict/scripts/evaluation/requirements_validation.py), reporta tres WARN para la estrategia intradiaria:

- **REQ-07** Métricas ML reportadas: `PARTIAL (0/4)`. Se exige que el registry contenga las cuatro métricas `ml_mae`, `ml_rmse`, `ml_ic`, `ml_directional_accuracy` para cada ticker, ya sea en el slot `champion` o en el slot `baseline`.
- **REQ-11** Baseline comparativo por estrategia: `PARTIAL (0/4)`. Se exige que cada ticker tenga un slot `baseline` poblado en el registry. Los 4 tickers de E3 (AAPL, NVDA, QQQ, SPY) no lo tienen.
- **REQ-13** Reproducibilidad: `WARN`. Se exige `seed=42` (cumplido a nivel global) y la presencia de `config_used.yaml` dentro del último directorio de run de cada estrategia. El último run de E3 (`runs/e3_intraday/20260503_133411/`) no contiene ese archivo.

Estos WARN se mencionan explícitamente en la subsección 4.3.3 y en la matriz de 4.5 del capítulo 4 de la memoria. Cerrarlos produce una matriz de cumplimiento limpia sin reescribir nada del capítulo y refuerza la trazabilidad del trabajo.

Los tres WARN tienen el mismo origen: el pipeline `e3_intraday` persiste menos artefactos que sus pares diarios. Las tres correcciones son cambios pequeños y locales en `src/e3/`, sin impacto sobre el modelo entrenado.

## Diagnóstico técnico

### REQ-07 — `ml_rmse` ausente del summary
- El validador ([scripts/evaluation/requirements_validation.py:217](../../proyecto_final/trading_predict/scripts/evaluation/requirements_validation.py#L217)) requiere las cuatro claves `["ml_mae", "ml_rmse", "ml_ic", "ml_directional_accuracy"]` en el slot `champion` o `baseline` del registry.
- El summary de E3 ([src/e3/train_pipeline.py:411-413](../../proyecto_final/trading_predict/src/e3/train_pipeline.py#L411-L413)) registra `ml_mae`, `ml_ic` y `ml_directional_accuracy`, pero falta `ml_rmse`. La función `rmse()` ya existe e importa en [src/e3/intraday_metrics.py:10](../../proyecto_final/trading_predict/src/e3/intraday_metrics.py#L10) y se calcula internamente para diagnóstico ([train_pipeline.py:747](../../proyecto_final/trading_predict/src/e3/train_pipeline.py#L747)), pero ese valor no se propaga al summary final.
- Comparación con E1 ([src/e1/train_pipeline.py:877-880](../../proyecto_final/trading_predict/src/e1/train_pipeline.py#L877-L880)) y E2 ([src/e2/train_pipeline.py:714-717](../../proyecto_final/trading_predict/src/e2/train_pipeline.py#L714-L717)): ambos sí incluyen `ml_rmse` en su summary global. El baseline de E3 ([src/e3/train_baseline.py:290](../../proyecto_final/trading_predict/src/e3/train_baseline.py#L290)) también lo incluye, pero al no estar registrado el baseline slot (REQ-11), no compensa la falta en el champion.

### REQ-11 — Slot `baseline` ausente en el registry para los 4 tickers de E3
- El validador ([requirements_validation.py:300](../../proyecto_final/trading_predict/scripts/evaluation/requirements_validation.py#L300)) cuenta tickers con `td.get("baseline")` no vacío.
- El script `src/e3/train_baseline.py` sí registra el baseline en el registry ([línea 325](../../proyecto_final/trading_predict/src/e3/train_baseline.py#L325)) cuando se ejecuta. La causa más probable es operativa: tras el último reentrenamiento del champion no se volvió a correr el baseline, por lo que el slot `baseline` quedó vacío para los 4 tickers (el promotion_log o limpiezas previas pueden haberlo eliminado). El código está; falta la corrida.

### REQ-13 — `config_used.yaml` no se escribe en el run del champion E3
- El validador ([requirements_validation.py:334](../../proyecto_final/trading_predict/scripts/evaluation/requirements_validation.py#L334)) chequea la existencia del archivo en el directorio del último run.
- E1 lo escribe en [train_pipeline.py:1493](../../proyecto_final/trading_predict/src/e1/train_pipeline.py#L1493): `shutil.copy(cfg_path, out_base / "config_used.yaml")`.
- E2 lo escribe en [train_pipeline.py:1232](../../proyecto_final/trading_predict/src/e2/train_pipeline.py#L1232) con la misma línea.
- E3 baseline sí lo escribe ([train_baseline.py:508](../../proyecto_final/trading_predict/src/e3/train_baseline.py#L508)), pero el `train_pipeline.py` del champion E3 no lo emite. Falta la línea análoga.
- El último run de E3 (`runs/e3_intraday/20260503_133411/`) confirma la ausencia del archivo.

## Cambios propuestos

### 1. Agregar `ml_rmse` al summary de E3 (REQ-07)
Archivo: [src/e3/train_pipeline.py](../../proyecto_final/trading_predict/src/e3/train_pipeline.py).
- Calcular `rmse_all` análogo a `mae_all`, `ic_all`, `dir_acc_all` (ya existe `rmse()` en `intraday_metrics`).
- Calcular también `rmse_fold` para mantener simetría con la sección por fold (línea ~333 ya tiene `ml_mae`, `ml_ic`, `ml_directional_accuracy` por fold; sumar `ml_rmse`).
- Añadir la clave `"ml_rmse": rmse_all` al diccionario `summary` (línea ~411-413, junto a las otras tres `ml_*`).
- Verificar que el import en línea 44 ya incluye `rmse` (lo incluye).

Riesgo: bajo. Cambio aditivo, no rompe consumidores existentes (`compare_e3_models.py` ya calcula rmse al vuelo desde los archivos por ticker).

### 2. Emitir `config_used.yaml` en el pipeline de E3 (REQ-13)
Archivo: [src/e3/train_pipeline.py](../../proyecto_final/trading_predict/src/e3/train_pipeline.py).
- Localizar el bloque donde se construye `out_base` para el run del champion (similar a E1/E2 cerca de línea 1493 y 1232 respectivamente). En E3 está en la función `run_e3_walk_forward()` o equivalente del pipeline; identificar el punto exacto.
- Asegurar que el `cfg_path` (ruta al `base.yaml` o YAML utilizado) está disponible en ese ámbito. Si no, propagarlo desde el caller o reutilizar el approach de E1.
- Añadir `shutil.copy(cfg_path, out_base / "config_used.yaml")` al final del pipeline (después de escribir `summary_all.csv`).

Riesgo: bajo. Cambio aditivo y estándar (E1 y E2 ya lo hacen igual).

### 3. Re-ejecutar el baseline de E3 para los 4 tickers (REQ-11)
No requiere cambio de código; es operativo.
- Activar el entorno: `conda activate ia_ceia_18co`.
- Correr `python -m src.e3.train_baseline` con la configuración por defecto (cubre los 4 tickers según `base.yaml`). Validar que cada ticker quede registrado en el slot `baseline` del registry.
- Verificación post-corrida: inspeccionar `models/registry.json` para confirmar que cada uno de los 4 tickers de E3 tiene `baseline` no vacío.

Riesgo: bajo. El script ya está validado y los entrenamientos del baseline son rápidos en intradía (modelos Ridge, no LSTM).

### 4. Reentrenar el champion de E3 una sola vez (consolidar REQ-07 y REQ-13)
Necesario para que los cambios 1 y 2 produzcan un run con el summary y el `config_used.yaml` actualizados.
- Correr `python -m src.e3.train_pipeline` con la configuración estándar.
- Inspeccionar el nuevo directorio bajo `runs/e3_intraday/` para confirmar que existen `config_used.yaml` y que `summary_all.csv` incluye la columna `ml_rmse`.
- Validar que la corrida produzca un nuevo candidate registrado, con las cuatro métricas ML completas. La promoción a champion se decide automáticamente por puntaje compuesto y no es objetivo de este plan; si no promueve, el champion previo (sin `ml_rmse` ni `config_used.yaml`) seguirá vigente y los WARN persistirán hasta la próxima promoción exitosa. En ese caso, opcionalmente forzar la promoción manual o aceptar el estado y dejar nota.

Riesgo: medio. El reentrenamiento de E3 toma tiempo (ya conocido por el usuario). Mitigación: ejecutar en paralelo con otras tareas, sin bloquear la redacción del capítulo 4. Si no se desea esperar, los cambios 1 y 2 quedan en código pero los WARN persisten hasta el próximo reentrenamiento natural.

### 5. Regenerar la matriz de validación
- Correr `python -m scripts.evaluation.requirements_validation`.
- Verificar que `reports/conclusiones/tablas/requirements_validation.csv` muestre los tres WARN convertidos en `OK`.
- Refrescar `consolidated_metrics.csv`/`consolidated_aggregates.csv` con `python -m scripts.evaluation.consolidate_strategy_metrics` para que las nuevas métricas E3 aparezcan también en los agregados.

## Archivos críticos

A modificar:
- [src/e3/train_pipeline.py](../../proyecto_final/trading_predict/src/e3/train_pipeline.py) — sumar `ml_rmse` al summary y la copia de `config_used.yaml`.

A re-ejecutar (sin modificación):
- `src.e3.train_baseline` — re-corre baseline E3 para los 4 tickers.
- `src.e3.train_pipeline` — re-corre champion E3 una vez aplicados los cambios.
- `scripts.evaluation.requirements_validation` y `scripts.evaluation.consolidate_strategy_metrics` — regeneran las tablas afectadas.

A consultar como referencia (no se modifican):
- [src/e1/train_pipeline.py:877-880, 1493](../../proyecto_final/trading_predict/src/e1/train_pipeline.py) — patrón canónico de `ml_rmse` y `config_used.yaml`.
- [src/e2/train_pipeline.py:714-717, 1232](../../proyecto_final/trading_predict/src/e2/train_pipeline.py) — mismo patrón aplicado a E2.
- [src/e3/intraday_metrics.py:10](../../proyecto_final/trading_predict/src/e3/intraday_metrics.py#L10) — definición de `rmse()`.

## Verificación end-to-end

1. Aplicar los cambios 1 y 2 en `src/e3/train_pipeline.py`.
2. Re-ejecutar baseline E3 (cambio 3); verificar registry actualizado.
3. Re-ejecutar champion E3 (cambio 4); confirmar presencia de `config_used.yaml` y de `ml_rmse` en `summary_all.csv`.
4. Regenerar `requirements_validation.csv` (cambio 5) y comprobar que REQ-07, REQ-11 y REQ-13 figuran como `OK` para E3.
5. Refrescar la matriz citada en la sección 4.5 del capítulo 4 de la memoria; quitar las observaciones específicas sobre los WARN en 4.3.3 si ya no aplican.
6. Opcional: correr la suite `pytest tests/ -q` desde la raíz del proyecto (52 tests) para asegurar que los cambios no rompen invariantes del lifecycle.

## Notas

- El plan no toca el modelo en sí ni los hiperparámetros, solo la persistencia de artefactos. La métrica `ml_rmse` es informativa para el registry; el composite score de promoción no la usa, así que no afecta decisiones históricas de promoción.
- Si en el futuro se incorpora otra estrategia con un patrón análogo, conviene factorizar la copia de `config_used.yaml` en una utilidad común dentro de `src/utils.py` para evitar la divergencia entre pipelines.

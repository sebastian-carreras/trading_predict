"""
Pipeline completo E1 (Estrategia Conservadora).

Flujo:
1.  Cargar config base y aplicar overrides de Optuna por ticker (si existen)
2.  Cargar datos OHLCV (prioriza datos limpios en data/clean/, fallback a data/raw/)
3.  Calcular features técnicas E1
4.  Crear target (retorno futuro a N días) y secuencias temporales
5.  Walk-forward validation (si splits.method == "walk_forward"):
      Por cada fold:
      a. Split temporal train/val/test con embargo
      b. Estandarizar features y targets (Z-score, stats solo de train)
      c. Entrenar GRU con early stopping
      d. Predecir en test y calcular métricas ML (MAE, RMSE, IC, dir. accuracy)
      e. Backtest de señales de trading
    O bien split temporal simple (70/15/15) con el mismo flujo a-e en una sola pasada.
6.  Guardar modelo (.pth), predicciones, scalers, métricas y gráficos
7.  Registrar candidato en lifecycle registry + auto-promoción opcional
8.  Tracking en MLflow (intenta servidor remoto, fallback SQLite local)
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
from functools import lru_cache
import socket
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    from dotenv import load_dotenv
    load_dotenv()  # Cargar .env (MLflow, MinIO, etc.)
except ImportError:
    pass

from .build_features import compute_e1_features, make_target_e1
from ..features.build_sequences_e1e2 import make_sequences, time_split, temporal_train_val_split
from .gru import GRURegressor
from ..backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from ..utils import (
    apply_training_window,
    ensure_dir,
    load_yaml,
    log_timing_event,
    project_root,
    resolve_lifecycle_paths,
)

# Memoriza hasta 8 lecturas de archivos de parámetros tuneados (clave = path_str);
# si se consulta el mismo path otra vez, evita re-leer/parsing del YAML.
# esto acelera cargas repetidas del mismo archivo para múltiples tickers en un mismo proceso.
@lru_cache(maxsize=8)
def _load_tuned_params(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        # Si no hay archivo tuneado, devolvemos vacío y se usa config base.
        return {}
    data = load_yaml(path)
    if not isinstance(data, dict):
        # Esperamos un mapping {ticker: {...}}; cualquier otro formato se ignora.
        return {}
    return data


_SHARED_PARAMS_WARNED: set = set()


def _warn_shared_params(all_tuned: dict, source_path: Path) -> None:
    """Detecta tickers con parámetros idénticos (posibles YAML anchors residuales)."""
    cache_key = str(source_path)
    if cache_key in _SHARED_PARAMS_WARNED:
        return
    _SHARED_PARAMS_WARNED.add(cache_key)

    # Agrupar tickers por repr de sus params
    by_repr: dict[str, list[str]] = {}
    for ticker, params in all_tuned.items():
        if not isinstance(params, dict):
            continue
        key = repr(sorted(params.items()))
        by_repr.setdefault(key, []).append(ticker)

    for group in by_repr.values():
        if len(group) > 1:
            import warnings
            warnings.warn(
                f"Optuna: tickers con params idénticos (posible YAML anchor): "
                f"{', '.join(group)} en {source_path.name}. "
                f"Considerar re-correr --per_ticker para optimización individual.",
                UserWarning,
                stacklevel=3,
            )
            break  # Un solo warning es suficiente


def _apply_tuned_overrides(*, config: dict, strategy_key: str, ticker: str) -> tuple[dict, dict]:
    """Aplica overrides de Optuna (por ticker) sobre el config base.

    Resolución de path con prioridad:
      1. E1_TUNED_PARAMS_PATH (env var, override de emergencia)
      2. TUNED_PARAMS_PATH (env var, fallback genérico)
      3. config["optuna"][strategy_key]["tuned_params_path"] (base.yaml, fuente principal)

    Returns:
        (config, hyperparams_info) donde hyperparams_info es un dict con:
          - source: "optuna" | "base_yaml"
          - tuned_params_file: path del archivo de overrides (si aplica)
          - overrides_applied: lista de secciones que se sobreescribieron
    """
    hp_info: dict = {"source": "base_yaml", "tuned_params_file": None, "overrides_applied": []}

    # --- Resolver path: ENV var (deprecado) > config ---
    env_path = (
        os.getenv("E1_TUNED_PARAMS_PATH", "").strip()
        or os.getenv("TUNED_PARAMS_PATH", "").strip()
    )

    if env_path:
        import warnings
        warnings.warn(
            "E1_TUNED_PARAMS_PATH / TUNED_PARAMS_PATH están deprecados. "
            "Usar base.yaml > optuna > e1_conservative > tuned_params_path en su lugar.",
            DeprecationWarning,
            stacklevel=2,
        )
        tuned_path = env_path
    else:
        # Buscar en config (base.yaml > optuna > strategy_key)
        optuna_cfg = config.get("optuna", {}).get(strategy_key, {})
        tuned_path = optuna_cfg.get("tuned_params_path", "")
        if not tuned_path:
            return config, hp_info

    # Resolver path relativo al root del proyecto
    resolved = Path(tuned_path)
    if not resolved.is_absolute():
        try:
            resolved = project_root() / resolved
        except Exception:
            pass

    all_tuned = _load_tuned_params(str(resolved))
    if not all_tuned:
        print(f"  ⚠️  Optuna: archivo no encontrado o vacío: {resolved}")
        return config, hp_info

    # Validar params compartidos entre tickers (detecta YAML anchors residuales)
    _warn_shared_params(all_tuned, resolved)

    per_ticker = all_tuned.get(ticker)
    if not isinstance(per_ticker, dict):
        # Archivo tuneado existe pero no contiene este ticker.
        fallback = config.get("optuna", {}).get(strategy_key, {}).get("fallback_to_base", True)
        if fallback:
            print(f"  ℹ️  Optuna: sin overrides para {ticker} → usando base.yaml")
        return config, hp_info

    # --- Aplica overrides y loguea cuales se aplicaron ---
    hp_info["source"] = "optuna"
    hp_info["tuned_params_file"] = str(resolved)

    strat = config.setdefault("strategies", {}).setdefault(strategy_key, {})
    overrides_applied: list[str] = []

    thresholds = per_ticker.get("thresholds")
    if isinstance(thresholds, dict):
        strat.setdefault("thresholds", {}).update(thresholds)
        parts = [f"{k}={v}" for k, v in thresholds.items()]
        overrides_applied.append(f"thresholds({', '.join(parts)})")

    model = per_ticker.get("model")
    if isinstance(model, dict):
        strat.setdefault("model", {}).update(model)
        key_params = []
        if "gru_units" in model:
            key_params.append(f"GRU {model['gru_units']}")
        if "dropout" in model:
            key_params.append(f"dropout={model['dropout']}")
        if "learning_rate" in model:
            key_params.append(f"lr={model['learning_rate']:.6f}")
        if "weight_decay" in model:
            key_params.append(f"wd={model['weight_decay']}")
        if "batch_size" in model:
            key_params.append(f"batch={model['batch_size']}")
        if key_params:
            overrides_applied.append(f"model({', '.join(key_params)})")

    hp_info["overrides_applied"] = overrides_applied
    if overrides_applied:
        print(f"  ✓ Optuna overrides para {ticker}: {' | '.join(overrides_applied)}")

    return config, hp_info


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    # Normalizamos a Path por si llega como string u otro tipo path-like.
    path = Path(path)

    # Validaciones tempranas de existencia/tipo para dar errores claros.
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected CSV file but got directory: {path}")

    try:
        # Lectura inicial del CSV completo en memoria.
        # EmptyDataError: archivo existe pero sin columnas/filas parseables
        # (por ejemplo, 0 bytes o solo saltos de línea).
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        # Incluir tamaño del archivo ayuda a diagnosticar corrupciones o descargas incompletas.
        size = path.stat().st_size
        raise ValueError(
            f"Empty/invalid CSV (no columns to parse): {path} (size={size} bytes)"
        ) from exc

    # Columna temporal obligatoria para indexar series en orden cronológico.
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    # Parseo robusto a datetime con zona horaria UTC para evitar ambigüedades.
    # Esto homogeniza fuentes con distintos formatos horarios.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)

    # Orden cronológico ascendente y uso de timestamp como índice principal.
    # El pipeline asume este orden para splits temporales y cálculo de features.
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Esquema mínimo OHLCV requerido por features/backtesting.
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    # Devuelve DataFrame indexado por timestamp y listo para etapas siguientes.
    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # IC mide ranking predictivo (correlación monotónica entre retorno real y predicho).
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        # Si una serie es constante no existe correlación informativa.
        return float("nan")

    try:
        from scipy.stats import spearmanr

        # Spearman es más robusto que Pearson frente a outliers/no-linealidad.
        ic, _ = spearmanr(y_true, y_pred)
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        # Fallback defensivo cuando scipy no está disponible.
        return float(np.corrcoef(y_true, y_pred)[0, 1])


_PLOT_RCPARAMS = {
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
}


def save_walkforward_plot(fold_df: pd.DataFrame, ticker: str, out_path: Path) -> None:
    """Guarda grafico de IC y Sharpe por fold."""
    if fold_df.empty:  # Sin datos
        return  # No hay datos para graficar

    try:  # Import matplotlib
        import matplotlib  # Import matplotlib

        matplotlib.use("Agg")  # Backend no interactivo
        import matplotlib.pyplot as plt  # Pyplot
    except Exception as exc:  # pragma: no cover - fallback si matplotlib no está
        print(f"⚠️  No se pudo generar el gráfico walk-forward para {ticker}: {exc}")  # Warning
        return  # Salir si no hay matplotlib

    plt.rcParams.update(_PLOT_RCPARAMS)  # Estilo tipográfico unificado
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)  # Crear figura con 2 subplots

    axes[0].plot(fold_df["fold"], fold_df["ml_ic"], marker="o")  # Serie IC
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)  # Línea base
    axes[0].set_ylabel("IC")  # Label eje Y
    axes[0].set_title(f"{ticker} - Walk-Forward Validation")  # Título

    axes[1].plot(fold_df["fold"], fold_df["bt_sharpe"], marker="o", color="#2ca02c")  # Serie Sharpe
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)  # Línea base
    axes[1].set_ylabel("Sharpe")  # Label eje Y
    axes[1].set_xlabel("Fold")  # Label eje X

    if "window" in fold_df.columns:  # Si hay etiquetas de ventana
        axes[1].set_xticks(fold_df["fold"].to_numpy())  # Ticks por fold
        axes[1].set_xticklabels(fold_df["window"].to_list(), rotation=35, ha="right")  # Labels ventanas

    for ax in axes:  # Iterar ejes
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)  # Grilla suave

    fig.tight_layout()  # Ajustar layout
    fig.savefig(str(out_path), dpi=150)  # Guardar PNG
    plt.close(fig)  # Cerrar figura


def _import_matplotlib():
    """Importa matplotlib con backend no interactivo. Retorna (matplotlib, plt) o None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update(_PLOT_RCPARAMS)
        return matplotlib, plt
    except Exception:
        return None


def save_equity_curve_plot(
    bt_all: pd.DataFrame, ohlcv: pd.DataFrame, ticker: str, out_path: Path
) -> None:
    """Equity curve de la estrategia vs buy-and-hold del activo."""
    result = _import_matplotlib()
    if result is None:
        return
    _, plt = result

    fig, ax = plt.subplots(figsize=(10, 5))

    # Equity de la estrategia (ya viene en bt_all)
    equity = bt_all["equity"]
    ax.plot(equity.index, equity.values, label="Estrategia", linewidth=1.2)

    # Buy-and-hold: normalizado al mismo capital inicial
    close = ohlcv.loc[equity.index, "close"]
    bh_equity = equity.iloc[0] * (close / close.iloc[0])
    ax.plot(bh_equity.index, bh_equity.values, label="Buy & Hold", linewidth=1.0, alpha=0.7)

    ax.set_title(f"{ticker} - Equity Curve")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Fecha")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_pred_vs_actual_plot(
    y_true: np.ndarray, y_pred: np.ndarray, ticker: str, out_path: Path
) -> None:
    """Scatter de retornos predichos vs reales con línea identidad."""
    result = _import_matplotlib()
    if result is None:
        return
    _, plt = result

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none")

    # Línea identidad
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.6, label="y = x")

    # Correlación de Pearson como anotación
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    ax.text(
        0.05, 0.95, f"Pearson r = {corr:.3f}",
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Retorno real")
    ax.set_ylabel("Retorno predicho")
    ax.set_title(f"{ticker} - Predicho vs Real")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_residuals_plot(
    y_true: np.ndarray, y_pred: np.ndarray, ticker: str, out_path: Path
) -> None:
    """Histograma de residuos (y_true - y_pred) con estadísticas."""
    result = _import_matplotlib()
    if result is None:
        return
    _, plt = result

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(residuals, bins=50, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

    mean_r = float(np.mean(residuals))
    std_r = float(np.std(residuals))
    ax.text(
        0.95, 0.95,
        f"media = {mean_r:.4f}\nstd = {std_r:.4f}",
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
    )

    ax.set_xlabel("Residuo (real - predicho)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"{ticker} - Distribución de Residuos")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def save_fold_metrics_panel(fold_df: pd.DataFrame, ticker: str, out_path: Path) -> None:
    """Panel de 4 métricas por fold: IC, Sharpe, MAE y Directional Accuracy."""
    if fold_df.empty:
        return
    result = _import_matplotlib()
    if result is None:
        return
    _, plt = result

    metrics = [
        ("ml_ic", "IC (Spearman)", "#1f77b4"),
        ("bt_sharpe", "Sharpe", "#2ca02c"),
        ("ml_mae", "MAE", "#d62728"),
        ("ml_directional_accuracy", "Dir. Accuracy", "#9467bd"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    folds = fold_df["fold"].to_numpy()

    for ax, (col, label, color) in zip(axes, metrics):
        if col not in fold_df.columns:
            ax.set_visible(False)
            continue
        vals = fold_df[col].to_numpy()
        ax.bar(folds, vals, color=color, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axhline(float(np.mean(vals)), color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8, axis="y")
        ax.set_title(label)

    # Labels en eje X del último row
    for ax in axes[2:]:
        ax.set_xlabel("Fold")
        if "window" in fold_df.columns:
            ax.set_xticks(folds)
            ax.set_xticklabels(fold_df["window"].to_list(), rotation=35, ha="right", fontsize=10)

    fig.suptitle(f"{ticker} - Métricas por Fold", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_e1_walk_forward(  # Walk-forward completo
    *,  # Solo kwargs
    X: np.ndarray,  # Matriz de features (n_samples, n_features)
    y: np.ndarray,  # Target de retornos futuros alineado a X
    ts: pd.DatetimeIndex,  # Timestamps utilizados para indexar las muestras
    feat_names: list[str] | pd.Index,  # Nombres de columnas de features
    ticker: str,  # Símbolo del activo en entrenamiento
    out_dir: Path,  # Carpeta de salida para predicciones/backtests
    ohlcv: pd.DataFrame,  # Historial OHLCV completo para cálculos de backtest
    splits_cfg: dict,  # Configuración de folds, embargo y tamaños
    gru_units: list[int],  # Tamaños de capas ocultas del GRU
    dropout: float,  # Dropout entre capas recurrentes
    dense_units: int,  # Dimensión de la capa densa final
    learning_rate: float,  # Learning rate del optimizador
    weight_decay: float,  # Weight decay del optimizador
    batch_size: int,  # Tamaño del batch para cada step
    max_epochs: int,  # Máximo epochs permitidos
    patience: int,  # Paciencia para early stopping
    loss: str,  # Tipo de loss («huber» o «mse»)
    huber_delta: float,  # Delta para Huber loss
    tau_buy: float,  # Umbral de activación para comprar
    tau_sell: float,  # Umbral para cerrar compras o abrir cortos
    round_trip_bps: float,  # Costos de ida y vuelta en bps
    holding_period: int,  # Días a mantener abiertas las posiciones
    allow_short: bool,  # Permitir señales de venta en corto
    max_position: float,  # Tamaño máximo de exposición por operación
    seed: int,  # Semilla para inicialización aleatoria
    lookback_days: int,  # Ventana histórica usada para features
    horizon_days: int,  # Horizonte de predicción siguiente
) -> dict:  # Retorna resumen
    """Ejecuta el walk-forward completo y resume resultados"""

    # Esta flag permite abortar si el script no llegó a guardar el modelo final
    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }  # Fin flags REQUIRE_MODEL_SAVE

    # Garantizamos que exista el directorio de salida donde se almacenarán los artefactos
    ensure_dir(out_dir)

    # Validamos que tengamos muestras suficientes para entrenar
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward")

    # Número de folds de validación (folds = cantidad de reentrenamientos temporales)
    folds = int(splits_cfg.get("folds", 5))
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1 para walk-forward")

    # Embargo entre train y test para evitar look-ahead bias.
    # En series temporales financieras, los datos cercanos al límite train/test
    # están correlacionados (autocorrelación, ventanas móviles solapadas).
    # El embargo descarta N días entre el fin de train y el inicio de test,
    # rompiendo esa correlación y evitando que el modelo "vea" información futura.
    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        embargo_days = int(embargo_cfg.get("e1", 0))
    else:
        embargo_days = int(embargo_cfg or 0)

    # Convertimos los días de embargo a muestras (approx. 1 muestra ≈ 1 día hábil)
    gap_samples = max(0, int(embargo_days))

    # Tamaño del test set por fold.
    # Se reparten muestras entre (folds + 1) bloques: los primeros folds son de test
    # y el +1 reserva el bloque inicial mínimo para train.
    # Ej: 1000 muestras, 5 folds → default_test_size = 1000 // 6 ≈ 166 muestras/fold.
    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    # Guardia: si test_size <= embargo, el test quedaría vacío tras descartar
    # las muestras embargadas. Se fuerza mínimo gap+1 para tener al menos 1 muestra útil.
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    # Inicializamos splitter que preserva orden temporal y respeta gap
    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    fold_summaries: list[dict] = []  # Lista de resúmenes por fold
    pred_frames: list[pd.DataFrame] = []  # Lista de predicciones por fold

    # El Payload, no es solo el modelo entrenado. Incluye qué arquitectura usar, cómo
    # normalizar los datos de entrada y con qué horizonte se entrenó. El payload te da todo eso junto.
    last_model_payload: dict | None = None  # Payload del último fold (para guardar)
    last_torch = None  # Referencia a torch para serializar el modelo
    last_mean_X = None  # Media de X del último fold
    last_std_X = None  # Std de X del último fold
    last_mean_y = None  # Media de y del último fold
    last_std_y = None  # Std de y del último fold
    total_train_seconds = 0.0  # Acumular timing de train
    total_predict_seconds = 0.0  # Acumular timing de predict
    total_train_samples = 0  # Acumular n_train
    total_val_samples = 0  # Acumular n_val
    val_losses: list[float] = []  # Val loss por fold

    root = project_root()  # Directorio raíz del proyecto (para paths relativos)

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))  # Convertir a path relativo si es posible
        except ValueError:
            return str(path)  # Fallback: devolver path absoluto si no está dentro del root

    # Determinar fracción de validación interna dentro del bloque de entrenamiento de cada fold.
    # Este valor NO define cuántos folds hay (eso lo define splits.folds / n_folds).
    raw_val_fraction = splits_cfg.get(
        "internal_val_fraction", splits_cfg.get("val_fraction", 0.15)
    )
    try:
        val_fraction_cfg = float(raw_val_fraction)  # Normalizar a float
    except (TypeError, ValueError):
        val_fraction_cfg = 0.15  # Valor por defecto si no se puede convertir

    if not 0 < val_fraction_cfg < 1:
        val_fraction_cfg = 0.15  # Enforce rango válido (0,1)


    # Bucle principal walk-forward:
    # 1) TimeSeriesSplit define train/test por fold (usando folds, test_size, gap/embargo)
    # 2) Dentro de train_full_idx se hace otro split train/val con internal_val_fraction
    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1  # Generar splits temporales de índices
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue  # Si algún bloque queda vacío, se salta este fold

        # Split interno train/val dentro del bloque de entrenamiento del fold actual.
        # Este split interno es temporal (respeta orden) para evitar leakage.
        try:
            train_idx, val_idx = temporal_train_val_split(
                train_full_idx,  # Índices del bloque de entrenamiento completo
                val_fraction=val_fraction_cfg,  # Porcentaje destinado a validación
            )
        except ValueError:
            split_point = max(1, int(len(train_full_idx) * 0.8))  # Fallback 80/20
            train_idx = train_full_idx[:split_point]  # Índices train
            val_idx = train_full_idx[split_point:]  # Índices val

        X_train = X[train_idx]  # Features de entrenamiento
        X_val = X[val_idx]  # Features de validación
        X_test = X[test_idx]  # Features de test

        X_tr_2d = X_train.reshape(-1, X_train.shape[-1])  # Flatten para calcular stats
        mean_X = X_tr_2d.mean(axis=0)  # Media por feature (solo train)
        std_X = X_tr_2d.std(axis=0) + 1e-12  # Desvío por feature (evita div 0)

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)  # Z-score en X

        y_train = y[train_idx]  # Targets train
        y_val = y[val_idx]  # Targets val
        y_test = y[test_idx]  # Targets test

        mean_y = float(y_train.mean())  # Media del target (train)
        std_y = float(y_train.std()) + 1e-12  # Desvío del target (train)

        def scale_y(data: np.ndarray) -> np.ndarray:
            return ((data - mean_y) / std_y).astype(np.float32)  # Z-score en y

        def unscale_y(data: np.ndarray) -> np.ndarray:
            return (data * std_y + mean_y).astype(np.float32)  # Deshacer Z-score

        X_train_s = scale_X(X_train)  # X train escalado
        X_val_s = scale_X(X_val)  # X val escalado
        X_test_s = scale_X(X_test)  # X test escalado

        y_train_s = scale_y(y_train)  # y train escalado
        y_val_s = scale_y(y_val)  # y val escalado

        model = GRURegressor(
            input_size=X_train.shape[-1],  # Número de features
            hidden_sizes=list(gru_units),  # Arquitectura GRU
            dropout=dropout,  # Dropout entre capas
            dense_units=dense_units,  # Tamaño de la capa densa
            seed=seed,  # Seed para reproducibilidad
        )

        train_started_at = datetime.now(timezone.utc).isoformat()
        train_start = time.perf_counter()
        res = model.fit(
            X_train_s,  # X train
            y_train_s,  # y train
            X_val_s,  # X val
            y_val_s,  # y val
            learning_rate=learning_rate,  # LR
            weight_decay=weight_decay,  # Weight decay
            batch_size=batch_size,  # Batch size
            max_epochs=max_epochs,  # Máximo de epochs
            early_stopping_patience=patience,  # Early stopping
            loss=loss,  # Tipo de loss
            huber_delta=huber_delta,  # Delta Huber
        )
        train_end = time.perf_counter()
        train_ended_at = datetime.now(timezone.utc).isoformat()
        total_train_seconds += train_end - train_start
        total_train_samples += int(len(train_idx))
        total_val_samples += int(len(val_idx))
        val_losses.append(float(res.best_val_loss))
        log_timing_event(
            strategy="e1_conservative",
            phase="train",
            duration_seconds=train_end - train_start,
            started_at=train_started_at,
            ended_at=train_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "split": "walk_forward",
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            },
        )

        pred_started_at = datetime.now(timezone.utc).isoformat()
        pred_start = time.perf_counter()
        y_pred_s = model.predict(X_test_s)  # Predicciones escaladas
        pred_end = time.perf_counter()
        pred_ended_at = datetime.now(timezone.utc).isoformat()
        total_predict_seconds += pred_end - pred_start
        log_timing_event(
            strategy="e1_conservative",
            phase="predict",
            duration_seconds=pred_end - pred_start,
            started_at=pred_started_at,
            ended_at=pred_ended_at,
            ticker=ticker,
            run_dir=out_dir,
            extra={
                "split": "walk_forward",
                "fold": int(fold_idx),
                "n_test": int(len(test_idx)),
            },
        )
        y_pred = unscale_y(y_pred_s)  # Predicciones en escala original

        last_torch = model.torch  # Guardamos referencia para serializar
        last_mean_X = mean_X
        last_std_X = std_X
        last_mean_y = mean_y
        last_std_y = std_y

        ts_test = ts[test_idx]  # Timestamps de test
        ts_train = ts[train_idx]  # Timestamps de train (para log de ventana completa)
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"  # Etiqueta de ventana (test; usada por plots/CSV)

        # Guardar payload del último fold (modelo + scalers) para inferencia.
        # Elegimos el último fold porque es el más reciente (más relevante operativamente).
        last_model_payload = {
            "ticker": ticker,  # Ticker del activo
            "strategy": "e1_conservative",  # Nombre de estrategia
            "created_at": datetime.now(timezone.utc).isoformat(),  # Timestamp UTC de entrenamiento
            "model_class": "GRURegressor",  # Clase del modelo
            "model_kwargs": {
                "input_size": int(X_train.shape[-1]),  # Número de features
                "hidden_sizes": list(gru_units),  # Arquitectura GRU
                "dropout": float(dropout),  # Dropout
                "dense_units": int(dense_units),  # Capa densa final
                "seed": int(seed),  # Semilla
            },
            "lookback_days": int(lookback_days),  # Lookback usado
            "horizon_days": int(horizon_days),  # Horizonte objetivo
            "feature_names": list(feat_names),  # Lista de features
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Stats de X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Stats de y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Training info
            "walkforward": {
                "fold": int(fold_idx),  # Fold actual
                "window": window_label,  # Ventana temporal
                "test_size": int(test_size),  # Tamaño test
                "gap_samples": int(gap_samples),  # Gap por embargo
            },
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos del modelo
        }

        mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE del fold
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE del fold
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy direccional
        ic = compute_information_coefficient(y_test, y_pred)  # IC (Spearman)

        close_prices = ohlcv.loc[ts_test, "close"].to_numpy()  # Precios de cierre test
        bt = backtest_daily_signals(
            timestamps=ts_test,  # Índices temporales
            close_prices=close_prices,  # Precio de cierre
            pred_returns=y_pred,  # Predicción de retornos
            tau_buy=tau_buy,  # Umbral compra
            tau_sell=tau_sell,  # Umbral venta
            round_trip_bps=round_trip_bps,  # Costos
            holding_period_days=holding_period,  # Holding period
            allow_short=allow_short,  # Permitir cortos
            max_position=max_position,  # Máx. exposición
        )
        trading_metrics = summarize_backtest(bt)  # Métricas del backtest

        sharpe = trading_metrics.get("sharpe", float("nan"))  # Sharpe del fold

        ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"  # IC formateado
        sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"  # Sharpe formateado
        # Log de fold: mostramos train Y test para dejar explícito que la ventana
        # es CRECIENTE (expanding) — el train siempre arranca en la misma fecha y
        # solo crece su fin; sin esto, ver únicamente el test da la falsa impresión
        # de ventana deslizante.
        fold_pct = 100.0 * fold_idx / folds
        print(
            f"    Fold {fold_idx}/{folds} ({fold_pct:5.1f}%): "
            f"train[{ts_train[0].date()} -> {ts_train[-1].date()}] n={len(train_idx)} (expanding) | "
            f"test[{window_label}] n={len(test_idx)} | "
            f"MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"
        )

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")  # Guardar backtest del fold

        fold_summaries.append(
            {
                "fold": fold_idx,  # ID de fold
                "window": window_label,  # Ventana temporal
                "test_start": ts_test[0],  # Inicio de test
                "test_end": ts_test[-1],  # Fin de test
                "n_train": int(len(train_idx)),  # Tamaño train
                "n_val": int(len(val_idx)),  # Tamaño val
                "n_test": int(len(test_idx)),  # Tamaño test
                "epochs_ran": int(res.epochs_ran),  # Epochs efectivas
                "val_loss": float(res.best_val_loss),  # Mejor loss de val
                "ml_mae": mae,  # MAE ML
                "ml_rmse": rmse,  # RMSE ML
                "ml_directional_accuracy": dir_acc,  # Directional accuracy
                "ml_ic": ic,  # IC
                **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},  # Métricas de backtest
            }
        )

        preds_df = pd.DataFrame(
            {
                "fold": fold_idx,  # Fold al que pertenece
                "y_true": y_test,  # Target real
                "y_pred": y_pred,  # Predicción
            },
            index=ts_test,  # Index temporal de test
        )
        preds_df.index.name = "timestamp"  # Etiqueta del índice
        pred_frames.append(preds_df)  # Guardar para concatenar luego

    if not fold_summaries:
        raise ValueError("No se generaron folds válidos para walk-forward")  # Validación final

    fold_df = pd.DataFrame(fold_summaries)  # DataFrame con métricas por fold
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)  # Guardar CSV

    combined_preds = pd.concat(pred_frames).sort_index()  # Unir predicciones de todos los folds
    combined_preds.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")  # Guardar predicciones

    y_true_all = combined_preds["y_true"].to_numpy()  # Targets reales combinados
    y_pred_all = combined_preds["y_pred"].to_numpy()  # Predicciones combinadas

    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))  # MAE global
    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))  # RMSE global
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))  # Direccional global
    ic_all = compute_information_coefficient(y_true_all, y_pred_all)  # IC global

    bt_all = backtest_daily_signals(
        timestamps=pd.DatetimeIndex(combined_preds.index),  # Timestamps globales
        close_prices=ohlcv.loc[combined_preds.index, "close"].to_numpy(),  # Precios de cierre
        pred_returns=y_pred_all,  # Predicciones globales
        tau_buy=tau_buy,  # Umbral compra
        tau_sell=tau_sell,  # Umbral venta
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding period
        allow_short=allow_short,  # Permitir cortos
        max_position=max_position,  # Máximo tamaño de posición
    )
    trading_metrics_all = summarize_backtest(bt_all)  # Métricas globales
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")  # Guardar backtest

    plot_path = out_dir / f"{ticker}_walkforward_metrics.png"  # Path gráfico
    save_walkforward_plot(fold_df, ticker, plot_path)  # Generar gráfico IC/Sharpe

    # Gráficos adicionales de diagnóstico
    save_equity_curve_plot(bt_all, ohlcv, ticker, out_dir / f"{ticker}_equity_curve.png")
    save_pred_vs_actual_plot(y_true_all, y_pred_all, ticker, out_dir / f"{ticker}_pred_vs_actual.png")
    save_residuals_plot(y_true_all, y_pred_all, ticker, out_dir / f"{ticker}_residuals.png")
    save_fold_metrics_panel(fold_df, ticker, out_dir / f"{ticker}_fold_metrics_panel.png")

    # Guardar modelo entrenado del último fold como artifact por ticker
    if last_model_payload is not None and last_torch is not None:
        model_path = out_dir / f"{ticker}_model.pth"  # Path del modelo
        try:
            last_torch.save(last_model_payload, model_path)  # Serializar payload
        except Exception as exc:
            print(f"⚠️  No se pudo guardar el modelo walk-forward para {ticker}: {exc}")  # Warning

        if require_model_save and not model_path.exists():
            raise RuntimeError(
                f"El pipeline terminó pero NO se guardó el modelo walk-forward en {model_path}. "
                "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
            )  # Error si se exige guardado

    scaler_path = None
    if last_mean_X is not None and last_std_X is not None:
        scaler_X_df = pd.DataFrame({"mean": last_mean_X, "std": last_std_X}, index=feat_names)
        scaler_path = out_dir / f"{ticker}_scaler.csv"
        scaler_X_df.to_csv(scaler_path)

    if last_mean_y is not None and last_std_y is not None:
        scaler_y_df = pd.DataFrame({"mean_y": [last_mean_y], "std_y": [last_std_y]})
        scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    summary = {
        "ticker": ticker,  # Ticker
        "n_samples": int(n_samples),  # Total de muestras
        "n_train": int(total_train_samples),  # Total de muestras train
        "n_val": int(total_val_samples),  # Total de muestras val
        "n_test": int(len(combined_preds)),  # Total de samples evaluadas
        "folds": int(len(fold_df)),  # Cantidad de folds
        "lookback_days": int(lookback_days),  # Lookback
        "horizon_days": int(horizon_days),  # Horizonte
        "split_method": "walk_forward",  # Método de split
        "walkforward_test_size": int(test_size),  # Tamaño test por fold
        "walkforward_gap": int(gap_samples),  # Gap entre train/test
        "val_loss": float(np.mean(val_losses)) if val_losses else None,  # Val loss promedio
        "ml_mae": mae_all,  # MAE global
        "ml_rmse": rmse_all,  # RMSE global
        "ml_directional_accuracy": dir_acc_all,  # Accuracy direccional
        "ml_ic": ic_all,  # IC global
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},  # Métricas backtest
        "timing_train_seconds": round(total_train_seconds, 2),  # Timing train total
        "timing_predict_seconds": round(total_predict_seconds, 4),  # Timing predict total
        "predictions_file": as_relative(out_dir / f"{ticker}_walkforward_predictions.csv"),  # CSV preds
        "backtest_file": as_relative(out_dir / f"{ticker}_walkforward_backtest.csv"),  # CSV backtest
        "scaler_file": as_relative(scaler_path) if scaler_path else "",  # CSV scaler
        "model_file": as_relative(model_path) if last_model_payload is not None and last_torch is not None else "",  # Modelo
    }  # Fin set REQUIRE_MODEL_SAVE

    return summary  # Retornar resumen final del walk-forward


def run_e1_for_ticker(
    config: dict,  # Configuración global (YAML ya cargado)
    ticker: str,  # Símbolo del activo a entrenar
    raw_dir: Path,  # Ruta de datos raw diarios
    out_dir: Path,  # Directorio base de salida
    register_lifecycle: bool = True,  # Registrar en lifecycle (False para Optuna trials)
    auto_promote: bool = False,  # Auto-promover a champion si supera al actual
    use_latest_data: bool = False,  # Si True, extiende training_window.end a hoy
 ) -> dict:  # Retorna resumen
    """Entrena y evalúa la estrategia E1 para un ticker específico."""

    # Copia defensiva para no contaminar el config compartido (Airflow / loops)
    config = copy.deepcopy(config)  # Evita side-effects cuando hay múltiples tickers
    config, hp_info = _apply_tuned_overrides(config=config, strategy_key="e1_conservative", ticker=ticker)  # Overrides por ticker

    # Flag para exigir guardado del modelo (por defecto sí)
    require_model_save = os.getenv("REQUIRE_MODEL_SAVE", "1").strip().lower() not in {  # Flag guardado
        "0",  # Deshabilitar
        "false",  # Deshabilitar
        "no",  # Deshabilitar
    }  # Fin set REQUIRE_MODEL_SAVE

    # Compatibilidad: si out_dir es base, crear subfolder por ticker
    out_dir = Path(out_dir)  # Normalizar a Path
    if out_dir.name != ticker:  # Asegurar subdir por ticker
        out_dir = out_dir / ticker  # Asegurar carpeta por ticker
    ensure_dir(out_dir)  # Crear carpeta si no existe

    root = project_root()

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    # Parámetros E1 del config
    e1 = config.get("strategies", {}).get("e1_conservative", {})  # Sección E1
    lookback_days = int(e1.get("lookback_days", 360))  # Lookback por default 360
    horizon_days = int(e1.get("horizon_days", 90))  # Horizonte por default 90

    model_cfg = e1.get("model", {})  # Sección de modelo
    gru_units = model_cfg.get("gru_units", [96, 32])  # Arquitectura GRU
    dropout = float(model_cfg.get("dropout", 0.2))  # Dropout
    dense_units = int(model_cfg.get("dense_units", 32))  # Dense units
    lr = float(model_cfg.get("learning_rate", 1e-3))  # Learning rate
    weight_decay = float(model_cfg.get("weight_decay", 0.0))  # Weight decay
    batch_size = int(model_cfg.get("batch_size", 64))  # Batch size
    max_epochs = int(model_cfg.get("max_epochs", 200))  # Max epochs
    patience = int(model_cfg.get("early_stopping_patience", 15))  # Patience
    loss = str(model_cfg.get("loss", "huber"))  # Loss
    huber_delta = float(model_cfg.get("huber_delta", 1.0))  # Huber delta

    seed = int(config.get("project", {}).get("seed", 42))  # Semilla global

    # Cargar datos (priorizar datos limpios si existen)
    # raw_dir es data/raw/daily, entonces data/clean está en raw_dir.parent.parent / "clean"
    clean_dir = raw_dir.parent.parent / "clean"  # Carpeta de datos limpios
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"  # Path del CSV limpio

    if clean_csv_path.exists():  # Si hay datos limpios
        csv_path = clean_csv_path  # Usar datos limpios
        print(f"  ✓ Usando datos limpios: {csv_path.name}")  # Log
    else:  # Fallback a datos raw
        csv_path = raw_dir / f"{ticker}_daily.csv"  # Fallback a raw
        if not csv_path.exists():  # Validar existencia
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")  # Error si no hay data
        print(f"  ⚠️  Usando datos raw (limpieza no ejecutada): {csv_path.name}")  # Warning

    ohlcv = load_ohlcv_csv(csv_path)  # Cargar OHLCV desde CSV
    ohlcv = apply_training_window(
        ohlcv, config, granularity="daily", use_latest=use_latest_data, ticker=ticker,
    )
    # Último día de la ventana de entrenamiento: cutoff para la comparación justa
    # champion-vs-candidate (ventana OOS común). Ver lifecycle.reevaluation.
    train_data_end = str(ohlcv.index.max().date()) if len(ohlcv) else None

    # Features
    features = compute_e1_features(ohlcv)  # Features E1
    target = make_target_e1(ohlcv, horizon_days=horizon_days)  # Target futuro
    # make_sequences alinea X/y/ts y descarta filas iniciales sin contexto suficiente.

    # Secuencias
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)  # Construir secuencias

    thresholds = e1.get("thresholds", {})  # Config de umbrales
    tau_buy = float(thresholds.get("tau_buy", 0.02))  # Umbral compra (default base.yaml)
    tau_sell = float(thresholds.get("tau_sell", 0.00))  # Umbral venta (default base.yaml)

    costs_cfg = config.get("costs", {})  # Config de costos
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))  # Costos diarios

    bt_cfg = e1.get("backtest", {})  # Config de backtest
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))  # Holding period
    allow_short = bool(bt_cfg.get("allow_short", False))  # Permitir cortos
    max_position = float(bt_cfg.get("max_position", 1.0))  # Máx posición

    splits_cfg = config.get("splits", {})  # Config de splits
    if splits_cfg.get("method") == "walk_forward":  # Si está habilitado walk-forward
        print(  # Log inicio walk-forward
            f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})"  # Log
        )  # Mensaje de inicio
        summary = run_e1_walk_forward(  # Ejecutar walk-forward
            X=X,  # Features
            y=y,  # Targets
            ts=ts,  # Timestamps
            feat_names=feat_names,  # Features
            ticker=ticker,  # Ticker
            out_dir=out_dir,  # Output
            ohlcv=ohlcv,  # OHLCV
            splits_cfg=splits_cfg,  # Config splits
            gru_units=list(gru_units),  # GRU units
            dropout=dropout,  # Dropout
            dense_units=dense_units,  # Dense units
            learning_rate=lr,  # LR
            weight_decay=weight_decay,  # Weight decay
            batch_size=batch_size,  # Batch
            max_epochs=max_epochs,  # Epochs
            patience=patience,  # Patience
            loss=loss,  # Loss
            huber_delta=huber_delta,  # Huber delta
            tau_buy=tau_buy,  # Threshold buy
            tau_sell=tau_sell,  # Threshold sell
            round_trip_bps=round_trip_bps,  # Costs
            holding_period=holding_period,  # Holding
            allow_short=allow_short,  # Shorts
            max_position=max_position,  # Max position
            seed=seed,  # Seed
            lookback_days=lookback_days,  # Lookback
            horizon_days=horizon_days,  # Horizon
        )  # Ejecutar walk-forward

        # Agregar hiperparámetros al summary para tracking en JSONL
        summary["hp_gru_units_1"] = gru_units[0]
        summary["hp_gru_units_2"] = gru_units[1] if len(gru_units) > 1 else None
        summary["hp_dropout"] = dropout
        summary["hp_dense_units"] = dense_units
        summary["hp_learning_rate"] = lr
        summary["hp_weight_decay"] = weight_decay
        summary["hp_batch_size"] = batch_size
        summary["hp_tau_buy"] = tau_buy
        summary["hp_tau_sell"] = tau_sell

        # Register walk-forward candidate in model lifecycle registry
        if register_lifecycle:
            try:
                from ..lifecycle.registry import ModelRegistry
                from ..lifecycle.guardrails import validate_candidate, log_candidate_metrics

                registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)
                log_candidate_metrics(
                    metrics=summary, strategy="e1", ticker=ticker,
                    run_dir=out_dir, variant="e1_conservative",
                    log_path=metrics_log_path,
                    feature_names=list(feat_names),
                    hyperparams_info=hp_info,
                )
                _guardrail_cfg = config.get("lifecycle", {}).get("guardrails", {})
                _min_sharpe = float(
                    _guardrail_cfg.get("min_sharpe_by_strategy", {}).get("e1_conservative", 0.0)
                )
                passed, errors = validate_candidate(
                    run_dir=out_dir, ticker=ticker, metrics=summary, min_sharpe=_min_sharpe
                )
                if passed:
                    # Solo registramos candidatos que superan guardrails mínimos de calidad.
                    registry = ModelRegistry(registry_path)
                    registry.register_candidate(
                        strategy="e1", ticker=ticker,
                        run_dir=str(out_dir.relative_to(root)),
                        metrics=summary, variant="e1_conservative",
                        feature_names=list(feat_names),
                        hyperparams=hp_info,
                        train_data_end=train_data_end,
                    )
                    print(f"  ✓ {ticker} registrado como candidato en el registro de ciclo de vida")

                    # Auto-promotion: compare candidate vs champion
                    if auto_promote:
                        try:
                            from ..lifecycle.promotion import evaluate_and_promote
                            promo_cfg = config.get("lifecycle", {}).get("promotion", {})
                            decision = evaluate_and_promote(
                                registry, "e1", ticker, promo_cfg,
                            )
                            if decision.promoted:
                                print(f"  ★ PROMOTED {ticker} to champion ({decision.reason})")
                            else:
                                print(f"  ↳ Kept current champion for {ticker} ({decision.reason})")
                        except Exception as promo_exc:
                            print(f"  ⚠️  Auto-promotion failed for {ticker}: {promo_exc}")
                else:
                    print(f"  ⚠️  Guardrails failed for {ticker}: {errors}")
            except Exception as exc:
                print(f"  Registracion en el ciclo de vida salteado: {exc}")

        pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")  # Guardar summary per-ticker
        return summary  # Retornar resumen walk-forward

    # Split temporal: dividir datos en 70% train, 15% val, 15% test
    # Se mantiene el orden temporal para evitar data leakage
    idx_train, idx_val, idx_test = time_split(len(X))  # Índices de split
    X_train, y_train = X[idx_train], y[idx_train]  # Datos train
    X_val, y_val = X[idx_val], y[idx_val]  # Datos val
    X_test, y_test = X[idx_test], y[idx_test]  # Datos test
    ts_test = ts[idx_test]  # Timestamps del test

    # ESTANDARIZAR FEATURES X (Z-score normalization)
    # - Usar solo datos de train para calcular media y std (prevenir data leakage)
    # - Aplicar la misma transformación a val y test
    # - Esto centra y escala los datos a media=0 y std=1
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])  # Flatten para stats
    mean_X = Xtr2d.mean(axis=0)  # Media por feature
    std_X = Xtr2d.std(axis=0) + 1e-12  # Std por feature (+1e-12 evita /0)

    def scale_X(Xa: np.ndarray) -> np.ndarray:  # Escalado Z-score de X
        """Aplica estandarización Z-score a las features."""
        return ((Xa - mean_X) / std_X).astype(np.float32)  # Z-score X

    X_train_s = scale_X(X_train)  # X train escalado
    X_val_s = scale_X(X_val)  # X val escalado
    X_test_s = scale_X(X_test)  # X test escalado

    # ESTANDARIZAR TARGETS y (Z-score normalization)
    # - Normalizamos los targets para estabilizar el entrenamiento
    # - Mejora convergencia durante el backprop
    mean_y = float(y_train.mean())  # Media del target en train
    std_y = float(y_train.std()) + 1e-12  # Std del target en train

    def scale_y(ya: np.ndarray) -> np.ndarray:
        """Aplica estandarización Z-score a los targets."""
        return ((ya - mean_y) / std_y).astype(np.float32)  # Z-score y

    def unscale_y(ya_scaled: np.ndarray) -> np.ndarray:  # Des-escalado de y
        """Deshace la estandarización (vuelve a escala original)."""
        return (ya_scaled * std_y + mean_y).astype(np.float32)  # Des-escalar y

    y_train_s = scale_y(y_train)  # y train escalado
    y_val_s = scale_y(y_val)  # y val escalado
    y_test_s = scale_y(y_test)  # y test escalado

    # Entrenar modelo GRU (Gated Recurrent Unit)
    # - Red recurrente para modelar secuencias temporales
    # - Capaz de capturar dependencias a largo plazo
    model = GRURegressor(
        input_size=X_train_s.shape[-1],  # Número de features (columnas)
        hidden_sizes=gru_units,  # Unidades en capas recurrentes
        dropout=dropout,  # Regularización: desactivar unidades al azar
        dense_units=dense_units,  # Unidades en capa densa final
        seed=seed,  # Para reproducibilidad
    )

    print(f"Entrenando GRU para {ticker}...")  # Log inicio entrenamiento
    train_started_at = datetime.now(timezone.utc).isoformat()
    train_start = time.perf_counter()
    res = model.fit(
        X_train_s,  # Features de entrenamiento (normalizados)
        y_train_s,  # Targets de entrenamiento (normalizados)
        X_val_s,  # Features de validación (para early stopping)
        y_val_s,  # Targets de validación
        learning_rate=lr,  # Velocidad de aprendizaje del optimizador
        weight_decay=weight_decay,  # Weight decay del optimizador
        batch_size=batch_size,  # Muestras por iteración
        max_epochs=max_epochs,  # Épocas máximas de entrenamiento
        early_stopping_patience=patience,  # Parar si no mejora en N épocas
        loss=loss,  # Función de pérdida: "huber" o "mse"
        huber_delta=huber_delta,  # Parámetro delta para loss Huber
    )
    train_end = time.perf_counter()
    train_ended_at = datetime.now(timezone.utc).isoformat()
    train_seconds = train_end - train_start
    log_timing_event(
        strategy="e1_conservative",
        phase="train",
        duration_seconds=train_seconds,
        started_at=train_started_at,
        ended_at=train_ended_at,
        ticker=ticker,
        run_dir=out_dir,
        extra={
            "split": splits_cfg.get("method", "time_split"),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        },
    )

    print(f"  Epochs: {res.epochs_ran}, Val Loss: {res.best_val_loss:.6f}")  # Log entrenamiento

    # Guardar modelo entrenado (por ticker)
    model_path = out_dir / f"{ticker}_model.pth"  # Path del modelo
    try:
        torch = model.torch  # Referencia a torch
        payload = {
            "ticker": ticker,  # Ticker
            "strategy": "e1_conservative",  # Estrategia
            "created_at": datetime.utcnow().isoformat(),  # Timestamp
            "model_class": "GRURegressor",  # Clase del modelo
            "model_kwargs": {
                "input_size": int(X_train_s.shape[-1]),  # Input size
                "hidden_sizes": list(gru_units),  # GRU units
                "dropout": float(dropout),  # Dropout
                "dense_units": int(dense_units),  # Dense units
                "seed": int(seed),  # Seed
            },
            "lookback_days": int(lookback_days),  # Lookback
            "horizon_days": int(horizon_days),  # Horizon
            "feature_names": list(feat_names),  # Feature names
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},  # Escalado X
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},  # Escalado y
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},  # Resultados
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},  # Pesos
        }
        torch.save(payload, model_path)  # Guardar modelo
        print(f"  ✓ Modelo guardado: {model_path}")  # Log guardado
    except Exception as exc:
        print(f"  ⚠️  No se pudo guardar el modelo para {ticker}: {exc}")  # Warning si falla

    if require_model_save and not model_path.exists():
        raise RuntimeError(
            f"El pipeline terminó pero NO se guardó el modelo en {model_path}. "
            "Si querés permitir continuar sin guardar, setear REQUIRE_MODEL_SAVE=0."
        )  # Error si se exige guardado

    # Predicciones en test (normalizadas)
    pred_started_at = datetime.now(timezone.utc).isoformat()
    pred_start = time.perf_counter()
    y_pred_s = model.predict(X_test_s)  # Predicciones escaladas
    pred_end = time.perf_counter()
    pred_ended_at = datetime.now(timezone.utc).isoformat()
    predict_seconds = pred_end - pred_start
    log_timing_event(
        strategy="e1_conservative",
        phase="predict",
        duration_seconds=predict_seconds,
        started_at=pred_started_at,
        ended_at=pred_ended_at,
        ticker=ticker,
        run_dir=out_dir,
        extra={
            "split": splits_cfg.get("method", "time_split"),
            "n_test": int(len(X_test)),
        },
    )

    # Desnormalizar predicciones para métricas y backtesting
    y_pred = unscale_y(y_pred_s)  # Predicciones en escala original

    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))  # MAE
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))  # RMSE
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))  # Accuracy direccional
    ic = compute_information_coefficient(y_test, y_pred)  # IC

    ml = {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "ic": ic}  # Métricas ML

    # Backtesting
    # Obtener precios de cierre del período de test
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()  # Close prices

    # Ejecutar backtest
    bt = backtest_daily_signals(
        timestamps=ts_test,  # Timestamps
        close_prices=close_prices_test,  # Close prices
        pred_returns=y_pred,  # Predicciones
        tau_buy=tau_buy,  # Umbral compra
        tau_sell=tau_sell,  # Umbral venta
        round_trip_bps=round_trip_bps,  # Costos
        holding_period_days=holding_period,  # Holding
        allow_short=allow_short,  # Shorts
        max_position=max_position,  # Max posición
    )

    # Métricas de trading
    trading_metrics = summarize_backtest(bt)  # Métricas de backtest



    # Guardar outputs
    ensure_dir(out_dir)  # Asegurar dir

    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)  # DF preds
    preds_df.index.name = "timestamp"  # Nombrar índice
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")  # Guardar preds

    # Guardar backtest
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")  # Guardar backtest

    # Guardar scaler de features X
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)  # DF scaler X
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")  # Guardar scaler X

    # Guardar scaler de target y (para inferencia futura)
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],  # Media y
        "std_y": [std_y]  # Std y
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)  # Guardar scaler y

    meta = {
        "ticker": ticker,  # Ticker
        "n_samples": int(len(X)),  # Muestras totales
        "n_train": int(len(X_train)),  # Muestras train
        "n_val": int(len(X_val)),  # Muestras val
        "n_test": int(len(X_test)),  # Muestras test
        "lookback_days": lookback_days,  # Lookback
        "horizon_days": horizon_days,  # Horizon
        "split_method": splits_cfg.get("method", "time_split"),  # Método split
        "epochs_ran": res.epochs_ran,  # Epochs ejecutadas
        "val_loss": res.best_val_loss,  # Best val loss
    }

    summary = {
        **meta,  # Metadata
        **{f"ml_{k}": v for k, v in ml.items()},  # Métricas ML
        **{f"bt_{k}": v for k, v in trading_metrics.items()},  # Métricas backtest
        "timing_train_seconds": round(train_seconds, 2),  # Timing train
        "timing_predict_seconds": round(predict_seconds, 4),  # Timing predict
        "predictions_file": as_relative(out_dir / f"{ticker}_predictions.csv"),  # CSV preds
        "backtest_file": as_relative(out_dir / f"{ticker}_backtest.csv"),  # CSV backtest
        "scaler_file": as_relative(out_dir / f"{ticker}_scaler.csv"),  # CSV scaler
        "model_file": as_relative(model_path),  # Modelo
        # Hiperparámetros para tracking en JSONL
        "hp_gru_units_1": gru_units[0],
        "hp_gru_units_2": gru_units[1] if len(gru_units) > 1 else None,
        "hp_dropout": dropout,
        "hp_dense_units": dense_units,
        "hp_learning_rate": lr,
        "hp_weight_decay": weight_decay,
        "hp_batch_size": batch_size,
        "hp_tau_buy": tau_buy,
        "hp_tau_sell": tau_sell,
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")  # Guardar summary

    # Register as candidate in model lifecycle registry
    if register_lifecycle:
        try:
            from ..lifecycle.registry import ModelRegistry
            from ..lifecycle.guardrails import validate_candidate, log_candidate_metrics

            registry_path, metrics_log_path = resolve_lifecycle_paths(config, root=root)
            log_candidate_metrics(
                metrics=summary, strategy="e1", ticker=ticker,
                run_dir=out_dir, variant="e1_conservative",
                log_path=metrics_log_path,
                feature_names=list(feat_names),
                hyperparams_info=hp_info,
            )
            _guardrail_cfg = config.get("lifecycle", {}).get("guardrails", {})
            _min_sharpe = float(
                _guardrail_cfg.get("min_sharpe_by_strategy", {}).get("e1_conservative", 0.0)
            )
            passed, errors = validate_candidate(
                run_dir=out_dir, ticker=ticker, metrics=summary, min_sharpe=_min_sharpe
            )
            if passed:
                registry = ModelRegistry(registry_path)
                registry.register_candidate(
                    strategy="e1", ticker=ticker,
                    run_dir=str(out_dir.relative_to(root)),
                    metrics=summary, variant="e1_conservative",
                    feature_names=list(feat_names),
                    hyperparams=hp_info,
                    train_data_end=train_data_end,
                )
                print(f"  ✓ {ticker} registrado como candidato en el registro de ciclo de vida")

                # Auto-promotion: compare candidate vs champion
                if auto_promote:
                    try:
                        from ..lifecycle.promotion import evaluate_and_promote
                        promo_cfg = config.get("lifecycle", {}).get("promotion", {})
                        decision = evaluate_and_promote(
                            registry, "e1", ticker, promo_cfg,
                        )
                        if decision.promoted:
                            print(f"  ★ PROMOTED {ticker} to champion ({decision.reason})")
                        else:
                            print(f"  ↳ Kept current champion for {ticker} ({decision.reason})")
                    except Exception as promo_exc:
                        print(f"  ⚠️  Auto-promotion failed for {ticker}: {promo_exc}")
            else:
                print(f"  ⚠️  Guardrails failed for {ticker}: {errors}")
        except Exception as exc:
            print(f"  Registracion en el ciclo de vida salteado: {exc}")

    return summary  # Retornar resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E1 (GRU conservador)")  # Parser CLI
    parser.add_argument(
        "--config",  # Flag de config
        type=str,  # Tipo str
        default="src/config/base.yaml",  # Config por defecto
        help="Path to config YAML",  # Ayuda CLI
    )
    parser.add_argument(
        "--tickers",  # Flag de tickers
        type=str,  # Tipo str
        default="",  # Por defecto vacío (usa config)
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e1_conservative)",  # Ayuda CLI
    )
    parser.add_argument(
        "--auto-promote",  # Flag de auto-promoción
        action="store_true",  # Booleano
        help="Automatically promote candidate to champion if it beats the current champion",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="No descargar datos; usar los existentes (por defecto se descargan datos nuevos, ver data.download en base.yaml)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Descargar datos y salir sin entrenar",
    )
    parser.add_argument(
        "--use-latest-data",
        action="store_true",
        help=(
            "Entrenar con el rango extendido hasta hoy "
            "(ignora data.training_window.daily.end del config; start se preserva)."
        ),
    )
    args = parser.parse_args()  # Parseo de argumentos

    if args.skip_download and args.use_latest_data:
        print(
            "⚠️  --skip-download + --use-latest-data: se extiende la ventana hasta hoy "
            "pero no se descargan datos frescos; puede no haber datos recientes."
        )

    root = project_root()  # Directorio raíz del proyecto
    cfg_path = Path(args.config)  # Path del config
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path  # Resolver a path absoluto

    config = load_yaml(cfg_path)  # Cargar YAML de config

    # Tickers E1
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]  # Tickers por CLI
    if not tickers:
        tickers = list(
            config.get("universe", {})  # Config de universo
            .get("tickers_by_strategy", {})  # Mapa de estrategia a tickers
            .get("e1_conservative", [])  # Tickers E1
        )

    if not tickers:
        raise ValueError("No tickers for E1")  # Error si no hay tickers

    # Informar si hay Optuna tuned params disponibles
    # Prioridad: ENV var > config (optuna.e1_conservative.tuned_params_path)
    tuned_path = (
        os.getenv("E1_TUNED_PARAMS_PATH", "").strip()
        or os.getenv("TUNED_PARAMS_PATH", "").strip()
        or config.get("optuna", {}).get("e1_conservative", {}).get("tuned_params_path", "")
    )
    if tuned_path:
        resolved = Path(tuned_path)
        if not resolved.is_absolute():
            resolved = root / resolved
        if resolved.exists():
            tuned_data = load_yaml(resolved)
            tuned_tickers = list(tuned_data.keys()) if isinstance(tuned_data, dict) else []
            matching = [t for t in tickers if t in tuned_tickers]
            print(f"✓ Optuna tuned params: {resolved.name}")
            if matching:
                print(f"  Tickers con overrides: {', '.join(matching)}")
            no_match = [t for t in tickers if t not in tuned_tickers]
            if no_match:
                print(f"  Tickers sin overrides (usarán base.yaml): {', '.join(no_match)}")
        else:
            print(f"⚠️  Optuna tuned params configurado pero archivo no existe: {resolved}")

    raw_dir = root / "data" / "raw" / "daily"  # Directorio raw diario
    clean_dir = root / "data" / "clean"  # Directorio de datos limpios

    # ------------------------------------------------------------------
    # Descarga y limpieza (config-driven: data.download; opt-out con --skip-download)
    # ------------------------------------------------------------------
    from ..data.ingest import refresh_data_for_training
    refresh_data_for_training(
        config, tickers, granularity="daily",
        skip_download=args.skip_download, root=root,
        raw_dir=raw_dir, clean_dir=clean_dir,
    )

    if args.download_only:
        print("\n--download-only: datos descargados, sin entrenar.")
        return

    out_base = root / "runs" / "e1_conservative" / datetime.now().strftime("%Y%m%d_%H%M%S")  # Output run
    ensure_dir(out_base)  # Crear folder de salida

    # Guardar config usado
    import shutil  # Import local para copia
    shutil.copy(cfg_path, out_base / "config_used.yaml")  # Copiar YAML usado

    # MLflow setup: intenta servidor remoto; fallback local robusto
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    remote_check_timeout_seconds = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E1_Conservative_Strategy")

    local_sqlite_dir = root / "runs" / "mlflow_local"
    ensure_dir(local_sqlite_dir)
    # DB de fallback SEPARADA de la del server. Si el server MLflow está caído y
    # caemos a SQLite local, NO debe escribir en mlflow.db (la DB compartida que
    # usan server + Airflow): hacerlo contamina la DB con artifact_location del
    # host (file:///Users/...) y rompe los runs en contenedor.
    local_sqlite_db = local_sqlite_dir / "mlflow_fallback.db"
    local_artifacts_dir = local_sqlite_dir / "artifacts"
    ensure_dir(local_artifacts_dir)
    local_sqlite_uri = f"sqlite:///{local_sqlite_db}"

    fallback_local_tracking_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
    if fallback_local_tracking_uri:
        local_sqlite_uri = fallback_local_tracking_uri

    local_mlruns = str(root / "mlruns")
    isolated_mlruns = str(root / "runs" / "e1_conservative" / "mlflow_store")
    ensure_dir(Path(isolated_mlruns))

    base_mlruns_path = Path(local_mlruns)
    has_malformed_mlruns = False
    if base_mlruns_path.exists():
        for exp_dir in base_mlruns_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.isdigit():
                continue
            if not (exp_dir / "meta.yaml").exists():
                has_malformed_mlruns = True
                break

    def _activate_mlflow(
        _mlflow_module,
        uri: str,
        experiment_artifact_dir: Path | None = None,
    ) -> tuple[bool, str | None]:
        # Encapsulamos activación para reutilizar misma lógica entre remoto/local.
        try:
            _mlflow_module.set_tracking_uri(uri)
            if experiment_artifact_dir is not None:
                # En backend SQLite creamos experiment con artifact_location explícito.
                exp = _mlflow_module.get_experiment_by_name(experiment_name)
                if exp is None:
                    _mlflow_module.create_experiment(
                        experiment_name,
                        artifact_location=experiment_artifact_dir.resolve().as_uri(),
                    )
                _mlflow_module.set_experiment(experiment_name)
            else:
                _mlflow_module.set_experiment(experiment_name)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _is_tracking_uri_reachable(uri: str, timeout_seconds: float) -> tuple[bool, str | None]:
        # Chequeo rápido de conectividad para no bloquear entrenamiento si el remoto cayó.
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            # URIs locales (file/sqlite) se consideran alcanzables por definición.
            return True, None

        host = parsed.hostname
        if not host:
            return True, None

        if parsed.port is not None:
            port = parsed.port
        elif parsed.scheme == "https":
            port = 443
        else:
            port = 80

        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True, None
        except OSError as exc:
            return False, str(exc)

    local_uris = [
        {
            "uri": local_sqlite_uri,
            "store_path": str(local_sqlite_db),
            "artifact_dir": local_artifacts_dir,
        },
        {
            "uri": f"file://{isolated_mlruns}",
            "store_path": isolated_mlruns,
            "artifact_dir": None,
        },
        {
            "uri": f"file://{local_mlruns}",
            "store_path": local_mlruns,
            "artifact_dir": None,
        },
    ] if has_malformed_mlruns else [
        {
            "uri": local_sqlite_uri,
            "store_path": str(local_sqlite_db),
            "artifact_dir": local_artifacts_dir,
        },
        {
            "uri": f"file://{local_mlruns}",
            "store_path": local_mlruns,
            "artifact_dir": None,
        },
        {
            "uri": f"file://{isolated_mlruns}",
            "store_path": isolated_mlruns,
            "artifact_dir": None,
        },
    ]

    try:
        import mlflow as _mlflow  # type: ignore

        if tracking_uri:
            remote_reachable, remote_reachability_err = _is_tracking_uri_reachable(
                tracking_uri,
                remote_check_timeout_seconds,
            )
            if remote_reachable:
                ok_remote, remote_err = _activate_mlflow(_mlflow, tracking_uri)
                if ok_remote:
                    mlflow = _mlflow
                    mlflow_enabled = True
                    print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
                else:
                    print(f"⚠️  MLflow servidor no disponible ({remote_err})")
            else:
                print(
                    "⚠️  MLflow remoto no accesible "
                    f"({tracking_uri}, timeout={remote_check_timeout_seconds}s): {remote_reachability_err}"
                )

        if not mlflow_enabled:
            # Intentos en cascada: sqlite local -> file store; evita perder trazabilidad.
            for local_cfg in local_uris:
                uri = local_cfg["uri"]
                store_path = local_cfg["store_path"]
                artifact_dir = local_cfg["artifact_dir"]
                ok_local, local_err = _activate_mlflow(
                    _mlflow,
                    uri,
                    experiment_artifact_dir=artifact_dir,
                )
                if ok_local:
                    mlflow = _mlflow
                    mlflow_enabled = True
                    mode = "local fallback" if tracking_uri else "tracking local"
                    print(f"✓ MLflow habilitado ({mode}, experiment={experiment_name})")
                    print(f"  → Store URI: {store_path}")
                    print(f"  → Para visualizar: mlflow ui --backend-store-uri {store_path}")
                    break
                print(f"⚠️  Falló MLflow local en {store_path}: {local_err}")
    except Exception as exc:
        print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
        mlflow_enabled = False

    summaries: list[dict] = []  # Resúmenes por ticker
    for i, ticker in enumerate(tickers, 1):
        ticker_out = out_base / ticker  # Output por ticker
        print("-" * 60)
        print(f"[{i}/{len(tickers)}] Entrenando {ticker}")
        print("-" * 60)
        try:
            if mlflow_enabled and mlflow is not None:
                timestamp = out_base.name  # Timestamp de run
                with mlflow.start_run(run_name=f"E1_{ticker}_{timestamp}"):
                    summary = run_e1_for_ticker(
                        config,  # Config
                        ticker=ticker,  # Ticker
                        raw_dir=raw_dir,  # Raw dir
                        out_dir=ticker_out,  # Output dir
                        auto_promote=args.auto_promote,  # Auto-promoción CLI
                        use_latest_data=args.use_latest_data,  # Override de training_window
                    )

                    # Parámetros: usar hp_ del summary (post-override de Optuna)
                    mlflow.log_param("strategy", "e1_conservative")  # Param estrategia
                    mlflow.log_param("ticker", ticker)  # Param ticker
                    mlflow.log_param("model_type", "GRU")  # Param tipo modelo
                    mlflow.log_param("timestamp", timestamp)  # Param timestamp
                    strategy_cfg = config.get("strategies", {}).get("e1_conservative", {})
                    model_cfg = strategy_cfg.get("model", {})
                    mlflow.log_params(
                        {
                            "lookback_days": int(summary.get("lookback_days", 360)),
                            "horizon_days": int(summary.get("horizon_days", 90)),
                            "gru_units": str([int(summary.get("hp_gru_units_1", 64)), int(summary.get("hp_gru_units_2", 32))]),
                            "dropout": summary.get("hp_dropout", 0.2),
                            "dense_units": int(summary.get("hp_dense_units", 32)),
                            "learning_rate": summary.get("hp_learning_rate", 1e-3),
                            "weight_decay": summary.get("hp_weight_decay", 0.0),
                            "batch_size": int(summary.get("hp_batch_size", 64)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 10)),
                            "max_epochs": int(summary.get("epochs_ran", 0)),
                            "tau_buy": summary.get("hp_tau_buy", 0.02),
                            "tau_sell": summary.get("hp_tau_sell", 0.00),
                            "split_method": summary.get("split_method", "time_split"),
                            "seed": int(config.get("project", {}).get("seed", 42)),
                        }
                    )

                    # Métricas
                    metrics: dict[str, float] = {}  # Dict de métricas
                    for k, v in summary.items():
                        if not (k.startswith("ml_") or k.startswith("bt_") or k.startswith("timing_")):
                            continue  # Solo métricas ML/BT
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)  # Convertir a float
                    if isinstance(summary.get("val_loss"), (int, float)):
                        metrics["val_loss"] = float(summary["val_loss"])  # Loss de validación agregada
                    if metrics:
                        mlflow.log_metrics(metrics)  # Log métricas

                    # Artifacts por ticker
                    try:
                        config_used = out_base / "config_used.yaml"  # Config usado
                        if config_used.exists():
                            mlflow.log_artifact(str(config_used), artifact_path="config")  # Log config

                        for fname in [
                            f"{ticker}_model.pth",
                            f"{ticker}_predictions.csv",
                            f"{ticker}_summary.csv",
                            f"{ticker}_backtest.csv",
                            f"{ticker}_scaler.csv",
                            f"{ticker}_target_scaler.csv",
                            f"{ticker}_walkforward_folds.csv",
                            f"{ticker}_walkforward_predictions.csv",
                            f"{ticker}_walkforward_backtest.csv",
                            f"{ticker}_walkforward_metrics.png",
                            f"{ticker}_equity_curve.png",
                            f"{ticker}_pred_vs_actual.png",
                            f"{ticker}_residuals.png",
                            f"{ticker}_fold_metrics_panel.png",
                        ]:
                            p = ticker_out / fname  # Path del artefacto
                            if p.exists():
                                # agrupamos por tipo para que sea navegable en la UI
                                if fname.endswith(".pth"):
                                    artifact_path = "models"  # Carpeta modelos
                                elif fname.endswith(".png"):
                                    artifact_path = "plots"  # Carpeta gráficos
                                elif "pred" in fname:
                                    artifact_path = "predictions"  # Carpeta preds
                                elif "backtest" in fname:
                                    artifact_path = "backtests"  # Carpeta backtests
                                elif "scaler" in fname:
                                    artifact_path = "scalers"  # Carpeta scalers
                                elif "walkforward" in fname:
                                    artifact_path = "walkforward"  # Carpeta WF
                                else:
                                    artifact_path = "artifacts"  # Carpeta default
                                mlflow.log_artifact(str(p), artifact_path=artifact_path)  # Log artifact
                    except Exception as art_exc:
                        print(f"  ⚠️  MLflow artifacts no guardados: {art_exc}")  # Warning artifacts
            else:
                # Camino sin tracking: se entrena igual y se guardan artefactos locales.
                summary = run_e1_for_ticker(
                    config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out,
                    auto_promote=args.auto_promote,  # Run sin MLflow
                    use_latest_data=args.use_latest_data,  # Override de training_window
                )
            summaries.append(summary)  # Agregar summary
            print(f"✓ {ticker}: MAE={summary['ml_mae']:.4f} IC={summary['ml_ic']:.3f}\n")  # Log ticker OK
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")  # Log error
        finally:
            print(f"Fin entrenamiento {ticker}")
            print()

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)  # Guardar resumen global

    # Run agregado (opcional) para el summary de todos los tickers.
    if mlflow_enabled and mlflow is not None:
        timestamp = out_base.name  # Timestamp del run
        try:
            with mlflow.start_run(run_name=f"E1_Summary_{timestamp}"):
                summary_path = out_base / "summary_all.csv"  # Path summary
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path), artifact_path="reports")  # Log summary
                mlflow.log_metric("total_tickers", float(len(tickers)))  # Total tickers
                mlflow.log_metric("successful_tickers", float(len(summaries)))  # Tickers ok

                # Loguear todas las variables numéricas de summary_all.csv
                summary_df = pd.DataFrame(summaries)
                if not summary_df.empty:
                    numeric_cols = list(summary_df.select_dtypes(include=[np.number]).columns)

                    # Métricas agregadas (media) para cada variable numérica
                    aggregate_metrics: dict[str, float] = {}
                    for col in numeric_cols:
                        series = pd.to_numeric(summary_df[col], errors="coerce")
                        series = series.replace([np.inf, -np.inf], np.nan).dropna()
                        if series.empty:
                            continue
                        aggregate_metrics[f"summary_{col}_mean"] = float(series.mean())

                    if aggregate_metrics:
                        mlflow.log_metrics(aggregate_metrics)

                    # Métricas por ticker y variable (valor exacto de summary_all.csv)
                    if "ticker" in summary_df.columns:
                        for _, row in summary_df.iterrows():
                            ticker_raw = str(row.get("ticker", "unknown"))
                            ticker_key = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in ticker_raw)
                            ticker_metrics: dict[str, float] = {}
                            for col in numeric_cols:
                                value = pd.to_numeric(row.get(col), errors="coerce")
                                if pd.isna(value) or np.isinf(float(value)):
                                    continue
                                ticker_metrics[f"ticker_{ticker_key}_{col}"] = float(value)
                            if ticker_metrics:
                                mlflow.log_metrics(ticker_metrics)

        except Exception as exc:
            print(f"⚠️  No se pudo loguear el summary en MLflow: {exc}")  # Warning MLflow

    print(f"\n✓ Resultados guardados en {out_base}")  # Log final


if __name__ == "__main__":  # Entry-point script
    main()  # Ejecutar main

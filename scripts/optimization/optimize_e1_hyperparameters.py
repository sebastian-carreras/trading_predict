#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Estrategia E1 usando Optuna + MLflow

Este script busca automáticamente los mejores hiperparámetros para la
estrategia E1 Conservative. Entrena el modelo GRU con split temporal de tuning
(time_split) para cada combinación de parámetros y optimiza una métrica objetivo
que combina
IC (Information Coefficient) y Sharpe ratio.

Nota metodológica: la validación walk-forward se reserva para la fase de
entrenamiento/evaluación final, fuera del loop de Optuna.

Parámetros optimizados y rangos de búsqueda:
  Trading thresholds:
    - tau_buy:  [0.02, 0.10] step=0.01  — Umbral de predicción para señal de compra
    - tau_sell: [-0.02, 0.02] step=0.01 — Umbral de predicción para señal de venta

  Arquitectura GRU (2 capas):
    - gru_units_1: [32, 128] step=16   — Unidades primera capa GRU
    - gru_units_2: [16, 64]  step=16   — Unidades segunda capa GRU
    - dropout:     [0.20, 0.50] step=0.05

  Entrenamiento:
    - learning_rate: [1e-4, 1e-2] log scale
        - weight_decay:  [1e-6, 1e-2] log scale
    - batch_size:    {32, 64, 128} categórico

Métrica objetivo:
  0.5 * IC_mean + 0.5 * Sharpe_mean + trade_penalty
  (trade_penalty = -5 si <50% de tickers generan trades)

Uso:
    # Optimización con defaults (MLflow local SQLite)
    python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 10

    # Optimizar un ticker específico
    python scripts/optimization/optimize_e1_hyperparameters.py --ticker GGAL.BA --n_trials 30

    # Optimización per-ticker (genera YAML de overrides por ticker)
    python scripts/optimization/optimize_e1_hyperparameters.py --per_ticker --n_trials 30

    # Usar servidor MLflow remoto (Docker)
    python scripts/optimization/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050

    # Continuar estudio existente
    python scripts/optimization/optimize_e1_hyperparameters.py --study_name e1_optimization --n_trials 20

Outputs:
- MLflow tracking: MLFLOW_TRACKING_URI (.env) o runs/mlflow_local/mlflow.db (SQLite local)
- Optuna database: runs/optuna_trials/optuna_studies.db (persistencia entre corridas)
- Mejores parámetros: reports/hyperparameter_optimization/best_params_e1.yaml
- Per-ticker overrides: reports/hyperparameter_optimization/e1_tuned_params_by_ticker.yaml
- Visualizaciones: reports/hyperparameter_optimization/by_ticker/<TICKER>/figures/
"""

import argparse
import io
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import yaml

# Carga opcional de variables de entorno desde .env (si python-dotenv está instalado).
# Esto permite setear, por ejemplo, URIs de MLflow sin hardcodear en el código.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optuna
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

# MLflow
import mlflow

# Agregar src/ al path
# Se inserta al inicio para priorizar módulos locales del proyecto por sobre paquetes globales.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.e1.train_pipeline import run_e1_for_ticker, load_ohlcv_csv
from src.utils import load_yaml, project_root


DISPLAY_PARAM_KEYS = [
    # Orden de impresión "humano" para el resumen final de mejores parámetros.
    # Esto evita mostrar params legacy o desordenados de studies viejos.
    "tau_buy",
    "tau_sell",
    "gru_units_1",
    "gru_units_2",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
]

OPTIMIZED_PARAM_KEYS = [
    "tau_buy",
    "tau_sell",
    "gru_units_1",
    "gru_units_2",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
]


class E1HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para estrategia E1.
    """
    
    def __init__(
        self,
        config_path: Path,
        tickers: Optional[List[str]] = None,
        mlflow_tracking_uri: str = "local",
        optuna_db_path: str = "sqlite:///runs/optuna_trials/optuna_studies.db",
        search_space_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size_choices: Optional[List[int]] = None,
    ):
        """
        Args:
            config_path: Path al archivo de configuración base
            tickers: Lista de tickers a usar (None = usar todos de E1)
            mlflow_tracking_uri: URI de MLflow tracking server (use 'local' para almacenamiento local)
            optuna_db_path: Path a base de datos de Optuna
            search_space_overrides: Overrides opcionales para rangos de búsqueda
            batch_size_choices: Opciones de batch size a evaluar (default: [32, 64, 128])
        """
        self.config = load_yaml(config_path)
        self.root = project_root()

        # Reducir ruido de logs de librerías de infraestructura
        logging.getLogger("alembic").setLevel(logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        
        # Tickers
        if tickers:
            # Si el usuario pasa tickers por CLI, usar exactamente esos.
            self.tickers = tickers
        else:
            # Fallback: usar universo E1 definido en config base.
            self.tickers = list(
                self.config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e1_conservative", [])
            )
        
        # MLflow - Configurar tracking URI
        self.mlflow_enabled = True
        mlflow_dir = self.root / "runs/mlflow_local"
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        mlflow_db = mlflow_dir / "mlflow.db"
        local_tracking_uri = f"sqlite:///{mlflow_db}"

        if mlflow_tracking_uri == "local":
            # Modo local: usar SQLite backend
            tracking_uri = local_tracking_uri
            print(f"✓ Usando MLflow local (SQLite): {mlflow_db}")
        else:
            # Modo remoto: usar servidor MLflow (ej: Docker)
            tracking_uri = mlflow_tracking_uri
            print(f"✓ Conectando a MLflow server: {tracking_uri}")

        mlflow.set_tracking_uri(tracking_uri)

        # Configurar experimento
        experiment_name = "E1_Hyperparameter_Optimization"

        # Artifact location compartido en modo local, para poder levantar MLflow UI luego
        local_artifacts_dir = self.root / "runs" / "mlflow_local" / "artifacts"
        local_artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_location = local_artifacts_dir.resolve().as_uri()

        try:
            # set_experiment crea o reutiliza experimento por nombre.
            mlflow.set_experiment(experiment_name)

            # Verificar si el artifact_location del experimento existente es accesible.
            # Si fue creado desde Docker (ej: /opt/airflow/...), log_artifact fallará localmente.
            exp = mlflow.get_experiment_by_name(experiment_name)
            if exp and exp.artifact_location:
                _art_loc = exp.artifact_location
                if _art_loc.startswith("/opt/") or _art_loc.startswith("file:///opt/"):
                    print(f"⚠️  Experimento '{experiment_name}' tiene artifact_location de Docker: {_art_loc}")
                    print(f"   Recreando experimento con artifact_location local...")
                    mlflow.delete_experiment(exp.experiment_id)
                    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                    mlflow.set_experiment(experiment_name)
                    print(f"✓ Experimento recreado con artifacts locales: {artifact_location}")

            # Validar conexión con un test run
            _test_run = mlflow.start_run(run_name="_init_test")
            mlflow.end_run()
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Error configurando MLflow: {e}")

            # Fallback automático: si falla remoto, intentar local SQLite
            if mlflow_tracking_uri != "local":
                try:
                    mlflow.set_tracking_uri(local_tracking_uri)
                    mlflow.set_experiment(experiment_name)
                    _test_run = mlflow.start_run(run_name="_init_test_local_fallback")
                    mlflow.end_run()
                    tracking_uri = local_tracking_uri
                    print(f"✓ Fallback MLflow local habilitado: {tracking_uri} (experiment={experiment_name})")
                    error_msg = ""  # Evitar lógica de error posterior
                except Exception as fallback_exc:
                    print(f"⚠️  Falló fallback local de MLflow: {fallback_exc}")

            # Si es un error de revisión de Alembic, recrear la base de datos
            if error_msg and ("Can't locate revision" in error_msg or "alembic" in error_msg.lower()):
                if mlflow_tracking_uri == "local":
                    print(f"⚠️  Base de datos MLflow corrupta. Recreando...")
                    if mlflow_db.exists():
                        # Si la DB quedó en estado inconsistente, se borra para regenerar limpia.
                        mlflow_db.unlink()
                    mlflow.set_tracking_uri(tracking_uri)

            # Crear experimento manualmente
            if error_msg:
                try:
                    # Intento manual con artifact_location explícito.
                    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                    mlflow.set_experiment(experiment_name)
                    print(f"✓ Experimento '{experiment_name}' creado")
                except Exception:
                    try:
                        mlflow.set_experiment(experiment_name)
                        print(f"✓ Experimento '{experiment_name}' configurado")
                    except Exception as final_exc:
                        print(f"⚠️  MLflow no disponible, continuando sin tracking: {final_exc}")
                        self.mlflow_enabled = False
        
        # Optuna
        self.optuna_db_path = optuna_db_path
        self.current_run_min_trial_number: int | None = None

        # Search space defaults (se pueden overridear vía CLI / DAG)
        self.tau_buy_bounds = {"min": 0.02, "max": 0.10, "step": 0.01}
        self.tau_sell_bounds = {"min": -0.02, "max": 0.02, "step": 0.01}
        self.gru_units_1_bounds = {"min": 32, "max": 128, "step": 16}
        self.gru_units_2_bounds = {"min": 16, "max": 64, "step": 16}
        self.dropout_bounds = {"min": 0.2, "max": 0.5, "step": 0.05}
        self.learning_rate_bounds = {"min": 1e-4, "max": 1e-2, "log": True}
        self.weight_decay_bounds = {"min": 1e-6, "max": 1e-2, "log": True}

        # Opciones de batch size
        default_batch_sizes = batch_size_choices or [32, 64, 128]
        # Se limpia/normaliza: enteros positivos, únicos y ordenados.
        cleaned_batch_sizes = sorted({int(b) for b in default_batch_sizes if int(b) > 0})
        if not cleaned_batch_sizes:
            raise ValueError("Se requiere al menos un batch_size válido para la optimización")
        self.batch_size_choices = cleaned_batch_sizes
        
        # Aplicar overrides si se proporcionaron
        overrides = search_space_overrides or {}

        def update_bounds(bounds: Dict[str, float], key: str) -> None:
            # Toma overrides de CLI y pisa min/max/step del rango default.
            override = overrides.get(key)
            if not override:
                return
            if "min" in override and override["min"] is not None:
                bounds["min"] = float(override["min"])
            if "max" in override and override["max"] is not None:
                bounds["max"] = float(override["max"])
            if "step" in override and override["step"] is not None and "step" in bounds:
                bounds["step"] = float(override["step"])

        update_bounds(self.tau_buy_bounds, "tau_buy")
        update_bounds(self.tau_sell_bounds, "tau_sell")
        update_bounds(self.gru_units_1_bounds, "gru_units_1")
        update_bounds(self.gru_units_2_bounds, "gru_units_2")
        update_bounds(self.dropout_bounds, "dropout")
        update_bounds(self.learning_rate_bounds, "learning_rate")
        update_bounds(self.weight_decay_bounds, "weight_decay")

        def validate_bounds(bounds: Dict[str, float], name: str) -> None:
            # Guardrails básicos del espacio de búsqueda.
            if bounds["min"] >= bounds["max"]:
                raise ValueError(
                    f"Rango inválido para {name}: min ({bounds['min']}) debe ser < max ({bounds['max']})"
                )
            if "step" in bounds and bounds.get("step", 0) <= 0:
                raise ValueError(f"El step debe ser > 0 para {name}")

        validate_bounds(self.tau_buy_bounds, "tau_buy")
        validate_bounds(self.tau_sell_bounds, "tau_sell")
        validate_bounds(self.gru_units_1_bounds, "gru_units_1")
        validate_bounds(self.gru_units_2_bounds, "gru_units_2")
        validate_bounds(self.dropout_bounds, "dropout")
        validate_bounds(self.learning_rate_bounds, "learning_rate")
        validate_bounds(self.weight_decay_bounds, "weight_decay")
        
        print(f"✓ Inicializado optimizador para {len(self.tickers)} tickers")
        print(f"  Tickers: {', '.join(self.tickers[:5])}{'...' if len(self.tickers) > 5 else ''}")
        print(f"  MLflow: {mlflow_tracking_uri}")
        print(f"  Optuna DB: {optuna_db_path}")
    
    @staticmethod
    def _suggest_float(trial: optuna.Trial, name: str, bounds: Dict[str, float]) -> float:
        # Normaliza tipos y decide la familia de distribución (lineal vs log).
        low = float(bounds["min"])
        high = float(bounds["max"])
        step = bounds.get("step")
        log_flag = bool(bounds.get("log"))

        if log_flag:
            if step:
                # Optuna no permite step + log simultáneamente.
                raise ValueError(f"No se puede usar step con distribución log para {name}")
            return trial.suggest_float(name, low, high, log=True)

        if step:
            return trial.suggest_float(name, low, high, step=float(step))

        return trial.suggest_float(name, low, high)

    @staticmethod
    def _suggest_int(trial: optuna.Trial, name: str, bounds: Dict[str, float]) -> int:
        # Versión para variables discretas enteras.
        low = int(bounds["min"])
        high = int(bounds["max"])
        step = bounds.get("step")

        if step:
            return trial.suggest_int(name, low, high, step=int(step))

        return trial.suggest_int(name, low, high)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para Optuna.
        
        Entrena modelo con hiperparámetros sugeridos y retorna métrica a optimizar.
        
        Args:
            trial: Trial de Optuna con suggestions
            
        Returns:
            Métrica objetivo (Sharpe ratio promedio o IC promedio)
        """
        
        # Iniciar run de MLflow para este trial (si está habilitado).
        # Se usa nullcontext para evitar if/else duplicando toda la lógica.
        from contextlib import nullcontext
        # Timer de trial completo (incluye entrenamiento en todos los tickers).
        trial_start = time.perf_counter()
        ctx = mlflow.start_run(run_name=f"trial_{trial.number}") if self.mlflow_enabled else nullcontext()
        with ctx:
            
            # 1. SUGERIR HIPERPARÁMETROS
            
            # Trading thresholds (críticos para backtesting)
            tau_buy = self._suggest_float(trial, "tau_buy", self.tau_buy_bounds)
            tau_sell = self._suggest_float(trial, "tau_sell", self.tau_sell_bounds)
            
            # Arquitectura GRU
            gru_units_1 = self._suggest_int(trial, "gru_units_1", self.gru_units_1_bounds)
            gru_units_2 = self._suggest_int(trial, "gru_units_2", self.gru_units_2_bounds)
            
            # Regularización
            dropout = self._suggest_float(trial, "dropout", self.dropout_bounds)
            
            # Entrenamiento
            learning_rate = self._suggest_float(trial, "learning_rate", self.learning_rate_bounds)
            weight_decay = self._suggest_float(trial, "weight_decay", self.weight_decay_bounds)
            batch_size = trial.suggest_categorical("batch_size", self.batch_size_choices)
            # Early stopping (fijar patience para consistencia)
            early_stopping_patience = 10
            
            # 2. ACTUALIZAR CONFIG CON PARÁMETROS SUGERIDOS
            
            # Copia superficial del dict principal. Ojo: objetos anidados siguen siendo compartidos.
            # En este caso se sobrescriben claves relevantes, por eso alcanza para el flujo actual.
            config_trial = self.config.copy()

            # Política del proyecto: Optuna solo para selección de hiperparámetros.
            # Forzamos split temporal liviano durante el tuning para evitar
            # duplicar costo con walk-forward en cada trial.
            splits_cfg = dict(config_trial.get("splits", {}))
            splits_cfg["method"] = "time_split"
            config_trial["splits"] = splits_cfg
            
            # Thresholds
            config_trial["strategies"]["e1_conservative"]["thresholds"] = {
                "tau_buy": tau_buy,
                "tau_sell": tau_sell,
            }
            
            # Modelo
            config_trial["strategies"]["e1_conservative"]["model"] = {
                **config_trial["strategies"]["e1_conservative"].get("model", {}),
                "gru_units": [gru_units_1, gru_units_2],
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "early_stopping_patience": early_stopping_patience,
            }
            
            # 3. ENTRENAR PARA TODOS LOS TICKERS
            
            results = []
            raw_dir = self.root / "data/raw/daily"
            
            # Output temporal para este trial
            out_dir = self.root / "runs/optuna_trials" / f"trial_{trial.number}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for ticker in self.tickers:
                train_log_buffer = io.StringIO()
                prev_e1_tuned = None
                prev_tuned = None
                try:
                    # Capturar logs internos del pipeline para mantener salida de Optuna concisa.
                    # Si el trial falla, mostramos un extracto para diagnóstico.
                    # Evitar contaminación de trials por YAMLs de parámetros tuneados
                    # inyectados vía variables de entorno.
                    prev_e1_tuned = os.environ.pop("E1_TUNED_PARAMS_PATH", None)
                    prev_tuned = os.environ.pop("TUNED_PARAMS_PATH", None)
                    try:
                        with redirect_stdout(train_log_buffer), redirect_stderr(train_log_buffer):
                            result = run_e1_for_ticker(
                                config=config_trial,
                                ticker=ticker,
                                raw_dir=raw_dir,
                                out_dir=out_dir,
                                register_lifecycle=False,
                            )
                    finally:
                        if prev_e1_tuned is not None:
                            os.environ["E1_TUNED_PARAMS_PATH"] = prev_e1_tuned
                        if prev_tuned is not None:
                            os.environ["TUNED_PARAMS_PATH"] = prev_tuned
                    
                    results.append({
                        # Se guardan métricas mínimas para score agregado y debugging.
                        "ticker": ticker,
                        "ic": result.get("ml_ic", 0),
                        "sharpe": result.get("bt_sharpe", 0),
                        "num_trades": result.get("bt_num_trades", 0),
                        "win_rate": result.get("bt_win_rate", 0),
                        "max_dd": result.get("bt_max_drawdown", 0),
                    })
                    
                except Exception as e:
                    debug_excerpt = ""
                    try:
                        # Mostrar solo las últimas líneas para no ensuciar consola.
                        lines = train_log_buffer.getvalue().splitlines()
                        if lines:
                            debug_excerpt = " | logs: " + " || ".join(lines[-3:])
                    except Exception:
                        pass
                    print(f"  ⚠️  Error en {ticker}: {e}{debug_excerpt}")
                    # Penalizar trials que fallan
                    results.append({
                        "ticker": ticker,
                        "ic": -1.0,
                        "sharpe": -10.0,
                        "num_trades": 0,
                        "win_rate": 0,
                        "max_dd": 1.0,
                    })
            
            # 4. CALCULAR MÉTRICAS AGREGADAS
            
            df_results = pd.DataFrame(results)
            
            # Métricas promedio
            ic_mean = df_results["ic"].mean()
            ic_median = df_results["ic"].median()
            sharpe_mean = df_results["sharpe"].mean()
            sharpe_median = df_results["sharpe"].median()
            
            # Métricas de consistencia
            ic_std = df_results["ic"].std()
            sharpe_std = df_results["sharpe"].std()
            
            # Tickers con performance positiva
            ic_positive_pct = (df_results["ic"] > 0.05).sum() / len(df_results) * 100
            sharpe_positive_pct = (df_results["sharpe"] > 0).sum() / len(df_results) * 100
            
            # Tickers con trades (evitar thresholds muy altos que no generan señales)
            tickers_with_trades = (df_results["num_trades"] > 0).sum()
            avg_num_trades = df_results["num_trades"].mean()
            
            # 5. REGISTRAR EN MLFLOW (dentro del trial)

            if self.mlflow_enabled:
                mlflow.log_params({
                    "tau_buy": tau_buy,
                    "tau_sell": tau_sell,
                    "gru_units_1": gru_units_1,
                    "gru_units_2": gru_units_2,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "trial_number": trial.number,
                })

                mlflow.log_metrics({
                    "ic_mean": ic_mean,
                    "ic_median": ic_median,
                    "ic_std": ic_std,
                    "sharpe_mean": sharpe_mean,
                    "sharpe_median": sharpe_median,
                    "sharpe_std": sharpe_std,
                    "ic_positive_pct": ic_positive_pct,
                    "sharpe_positive_pct": sharpe_positive_pct,
                    "tickers_with_trades": tickers_with_trades,
                    "avg_num_trades": avg_num_trades,
                })

            # Guardar tabla de resultados como artifact
            results_path = out_dir / "trial_results.csv"
            # Escritura robusta para evitar caídas del trial por timeouts del filesystem
            # (común en carpetas sincronizadas como iCloud/OneDrive).
            try:
                self._safe_write_file(results_path, lambda f: df_results.to_csv(f, index=False))
            except Exception as write_exc:
                # No abortar el trial por falla de IO auxiliar: métricas/objective ya están calculadas.
                print(f"  ⚠️  No se pudo escribir trial_results.csv: {write_exc}")
            if self.mlflow_enabled:
                try:
                    mlflow.log_artifact(str(results_path))
                except Exception as art_exc:
                    print(f"  ⚠️  MLflow artifact no guardado: {art_exc}")
            
            # 6. DEFINIR MÉTRICA OBJETIVO
            
            # Opción 1: Sharpe ratio promedio (preferido para trading)
            # objective_value = sharpe_mean
            
            # Opción 2: IC promedio (preferido para predicción)
            # objective_value = ic_mean
            
            # Opción 3: Combinación (balance predicción + trading)
            # Penalizar si pocos tickers generan trades
            # trade_penalty: castigo fuerte para evitar soluciones “bonitas” en métricas pero que casi no operan.
            trade_penalty = 0 if tickers_with_trades >= len(self.tickers) * 0.5 else -5
            # ic_mean: calidad predictiva (si el modelo ordena bien retornos futuros).
            # sharpe_mean: calidad de trading ajustada por riesgo.
            objective_raw = 0.5 * ic_mean + 0.5 * sharpe_mean
            objective_value = objective_raw + trade_penalty

            # Se guarda en user_attrs para poder graficar la evolución sin la penalización.
            trial.set_user_attr("objective_raw", float(objective_raw))
            trial.set_user_attr("trade_penalty", float(trade_penalty))
           
            
            if self.mlflow_enabled:
                mlflow.log_metric("objective_value", objective_value)
                mlflow.log_metric("objective_raw", objective_raw)
                mlflow.log_metric("trade_penalty", trade_penalty)
                mlflow.log_metric("trial_duration_seconds", time.perf_counter() - trial_start)
            
            trial_seconds = time.perf_counter() - trial_start
            print(
                # Línea compacta por trial: permite escanear rápido rendimiento y configuración.
                f"[Trial {trial.number:04d}] obj={objective_value:+.4f} "
                f"ic={ic_mean:+.4f} sharpe={sharpe_mean:+.4f} "
                f"trades={tickers_with_trades}/{len(self.tickers)} "
                f"time={trial_seconds:.1f}s | "
                f"tau=({tau_buy:.3f},{tau_sell:.3f}) "
                f"gru=[{gru_units_1},{gru_units_2}] do={dropout:.2f} "
                f"lr={learning_rate:.6f} wd={weight_decay:.6f} bs={batch_size}"
            )
            
            return objective_value
    
    def optimize(
        self,
        n_trials: int = 50,
        study_name: str = "e1_hyperparameter_optimization",
        timeout: Optional[int] = None,
    ) -> optuna.Study:
        """
        Ejecuta optimización de hiperparámetros.
        
        Args:
            n_trials: Número de trials a ejecutar
            study_name: Nombre del estudio (para continuar optimización)
            timeout: Timeout en segundos (None = sin límite)
            
        Returns:
            Estudio de Optuna con resultados
        """
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZACIÓN DE HIPERPARÁMETROS E1")
        print(f"{'='*80}")
        print(f"Trials: {n_trials}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Study: {study_name}")
        print(f"{'='*80}\n")
        
        # Reducir ruido de logs INFO de Optuna (mantenemos warnings/errores).
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Crear o cargar estudio
        study = optuna.create_study(
            study_name=study_name,
            storage=self.optuna_db_path,
            direction="maximize",  # Maximizar Sharpe/IC
            load_if_exists=True,  # Continuar si existe
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducibilidad
        )

        existing_numbers = [trial.number for trial in study.trials]
        self.current_run_min_trial_number = (max(existing_numbers) + 1) if existing_numbers else 0
        
        # Ejecutar optimización
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        return study
    
    @staticmethod
    def _safe_write_file(path: Path, write_fn, *, retries: int = 3, delay: float = 2.0) -> None:
        """Write a file with retry logic to handle iCloud/file-provider timeouts.

        Writes to a temp file first, then renames to avoid partial writes.
        """
        for attempt in range(1, retries + 1):
            try:
                # Escritura atómica aproximada: tmp + move.
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=path.suffix,
                )
                with os.fdopen(tmp_fd, "w") as f:
                    write_fn(f)
                shutil.move(tmp_path, str(path))
                return
            except (TimeoutError, OSError) as exc:
                # Clean up temp file on failure
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                if attempt < retries:
                    # Reintento útil para iCloud/FS remotos con latencia.
                    print(f"  ⚠️  Timeout escribiendo {path.name}, reintentando ({attempt}/{retries})...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"No se pudo escribir {path} después de {retries} intentos: {exc}"
                    ) from exc

    def save_results(
        self,
        study: optuna.Study,
        output_dir: Path,
    ) -> None:
        """
        Guarda resultados de optimización.

        Args:
            study: Estudio de Optuna
            output_dir: Directorio de salida
        """
        
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("GUARDANDO RESULTADOS")
        print(f"{'='*80}\n")

        # 1. Mejores parámetros en YAML

        best_params = study.best_params
        best_value = study.best_value

        best_params_yaml = {
            # Bloque de metadata del estudio para trazabilidad.
            "optimization": {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "best_value": float(best_value),
                "best_trial": study.best_trial.number,
            },
            # Serialización segura de tipos numéricos (evita objetos raros de numpy/optuna).
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in best_params.items()},
        }

        yaml_path = output_dir / "best_params_e1.yaml"
        self._safe_write_file(yaml_path, lambda f: yaml.dump(best_params_yaml, f, default_flow_style=False, sort_keys=False))

        print(f"✓ Mejores parámetros guardados: {yaml_path}")

        # 2. Todos los trials en CSV

        df_trials = study.trials_dataframe()
        csv_path = output_dir / "e1_all_trials.csv"
        self._safe_write_file(csv_path, lambda f: df_trials.to_csv(f, index=False))

        print(f"✓ Todos los trials guardados: {csv_path}")

        # 3. Visualizaciones

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Trials de la corrida actual (evita mezclar historia vieja del estudio).
        if self.current_run_min_trial_number is not None:
            df_run = df_trials[df_trials["number"] >= self.current_run_min_trial_number].copy()
        else:
            df_run = df_trials.copy()
        df_run_complete = df_run[df_run["state"] == "COMPLETE"].copy()

        run_csv_path = output_dir / "e1_run_trials.csv"
        self._safe_write_file(run_csv_path, lambda f: df_run.to_csv(f, index=False))

        df_all_complete = df_trials[df_trials["state"] == "COMPLETE"].copy()

        # Optimization history (historial completo del estudio, objetivo penalizado)
        try:
            import plotly.graph_objects as go  # type: ignore

            hist_df = df_all_complete.sort_values("number")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["number"],
                    y=hist_df["value"],
                    mode="lines+markers",
                    name="objective_value",
                )
            )
            fig.update_layout(
                title="Optimization History (Full Study) - Penalized Objective",
                xaxis_title="Trial",
                yaxis_title="Objective Value",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "optimization_history.png"), width=1200, height=600)
            print(f"✓ Gráfico: optimization_history.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history: {e}")

        # Optimization history (solo corrida actual, objetivo penalizado)
        try:
            import plotly.graph_objects as go  # type: ignore

            hist_df = df_run_complete.sort_values("number")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["number"],
                    y=hist_df["value"],
                    mode="lines+markers",
                    name="objective_value",
                )
            )
            fig.update_layout(
                title="Optimization History (Current Run) - Penalized Objective",
                xaxis_title="Trial",
                yaxis_title="Objective Value",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "optimization_history_current_run.png"), width=1200, height=600)
            print(f"✓ Gráfico: optimization_history_current_run.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_current_run: {e}")

        # Optimization history sin penalización (historial completo del estudio)
        try:
            import plotly.graph_objects as go  # type: ignore

            hist_df = df_all_complete.sort_values("number")
            raw_col = "user_attrs_objective_raw"
            raw_values = hist_df[raw_col] if raw_col in hist_df.columns else hist_df["value"]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["number"],
                    y=raw_values,
                    mode="lines+markers",
                    name="objective_raw",
                )
            )
            fig.update_layout(
                title="Optimization History (Full Study) - Raw Objective",
                xaxis_title="Trial",
                yaxis_title="Objective Raw (0.5*IC + 0.5*Sharpe)",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "optimization_history_raw.png"), width=1200, height=600)
            print(f"✓ Gráfico: optimization_history_raw.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_raw: {e}")

        # Optimization history sin penalización (solo corrida actual)
        try:
            import plotly.graph_objects as go  # type: ignore

            hist_df = df_run_complete.sort_values("number")
            raw_col = "user_attrs_objective_raw"
            raw_values = hist_df[raw_col] if raw_col in hist_df.columns else hist_df["value"]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["number"],
                    y=raw_values,
                    mode="lines+markers",
                    name="objective_raw",
                )
            )
            fig.update_layout(
                title="Optimization History (Current Run) - Raw Objective",
                xaxis_title="Trial",
                yaxis_title="Objective Raw (0.5*IC + 0.5*Sharpe)",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "optimization_history_raw_current_run.png"), width=1200, height=600)
            print(f"✓ Gráfico: optimization_history_raw_current_run.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_raw_current_run: {e}")

        # Parameter importances (mostrar siempre todos los hiperparámetros del espacio actual)
        try:
            import plotly.graph_objects as go  # type: ignore

            try:
                importances = optuna.importance.get_param_importances(
                    study,
                    params=OPTIMIZED_PARAM_KEYS,
                )
            except Exception:
                importances = {}

            full_importances = {k: float(importances.get(k, 0.0)) for k in OPTIMIZED_PARAM_KEYS}

            # Fallback: si todo quedó en cero, estimar importancia por correlación en la corrida actual.
            if sum(full_importances.values()) == 0.0 and not df_run_complete.empty:
                target_series = pd.to_numeric(df_run_complete["value"], errors="coerce")
                for key in OPTIMIZED_PARAM_KEYS:
                    col = f"params_{key}"
                    if col not in df_run_complete.columns:
                        continue
                    param_series = df_run_complete[col]
                    if param_series.dtype == object:
                        codes, _ = pd.factorize(param_series)
                        param_values = pd.Series(codes, index=param_series.index, dtype=float)
                    else:
                        param_values = pd.to_numeric(param_series, errors="coerce")

                    valid = (~param_values.isna()) & (~target_series.isna())
                    if valid.sum() < 3 or param_values[valid].nunique() <= 1:
                        full_importances[key] = 0.0
                        continue
                    corr = param_values[valid].corr(target_series[valid], method="spearman")
                    full_importances[key] = float(abs(corr)) if pd.notna(corr) else 0.0

            fig = go.Figure(
                go.Bar(
                    x=list(full_importances.values()),
                    y=list(full_importances.keys()),
                    orientation="h",
                )
            )
            fig.update_layout(
                title="Parameter Importances (Current Search Space)",
                xaxis_title="Importance",
                yaxis_title="Hyperparameter",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "param_importances.png"), width=1200, height=700)
            print(f"✓ Gráfico: param_importances.png")
        except Exception as e:
            print(f"⚠️  Error generando param_importances: {e}")

        # Parallel coordinate
        try:
            fig = plot_parallel_coordinate(study)
            fig.write_image(str(figures_dir / "parallel_coordinate.png"), width=1400, height=800)
            print(f"✓ Gráfico: parallel_coordinate.png")
        except Exception as e:
            print(f"⚠️  Error generando parallel_coordinate: {e}")

        # 4. Resumen en texto

        summary_path = output_dir / "e1_optimization_summary.txt"

        def _write_summary(f):
            # Resumen textual simple para lectura rápida sin abrir notebook/dashboard.
            f.write("="*80 + "\n")
            f.write("OPTIMIZACIÓN DE HIPERPARÁMETROS E1 - RESUMEN\n")
            f.write("="*80 + "\n\n")
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Best trial: #{study.best_trial.number}\n")
            f.write(f"Best value: {best_value:.6f}\n\n")
            f.write("Mejores parámetros:\n")
            f.write("-" * 40 + "\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 10 TRIALS\n")
            f.write("="*80 + "\n\n")
            df_top = df_trials.nlargest(10, "value")
            f.write(df_top.to_string(index=False))

        self._safe_write_file(summary_path, _write_summary)
        
        print(f"✓ Resumen guardado: {summary_path}")
        
        print(f"\n{'='*80}")
        print("RESULTADOS GUARDADOS EXITOSAMENTE")
        print(f"{'='*80}\n")


def main():
    # Parser CLI: centraliza todos los argumentos de ejecución del optimizador.
    parser = argparse.ArgumentParser(
        description="Optimización de hiperparámetros E1 con Optuna + MLflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path al archivo de configuración",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Número de trials a ejecutar (default: 50)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="e1_hyperparameter_optimization",
        help="Nombre del estudio Optuna (default: e1_hyperparameter_optimization)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optimizar solo para un ticker (más rápido para pruebas)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Lista de tickers separados por coma (overridea universo E1)",
    )
    parser.add_argument(
        "--per_ticker",
        action="store_true",
        help="Optimiza un estudio independiente por ticker y genera un YAML con overrides por ticker",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: usar solo 3 tickers aleatorios",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout en segundos (default: sin límite)",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="local",
        help="URI de MLflow tracking server (default: 'local' para SQLite)",
    )
    parser.add_argument(
        "--optuna_db",
        type=str,
        default="sqlite:///runs/optuna_trials/optuna_studies.db",
        help="Path a base de datos de Optuna (default: sqlite:///runs/optuna_trials/optuna_studies.db)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/hyperparameter_optimization",
        help="Directorio de salida para resultados",
    )
    parser.add_argument(
        "--tau_buy_min",
        type=float,
        default=None,
        help="Mínimo para tau_buy (default: 0.02)",
    )
    parser.add_argument(
        "--tau_buy_max",
        type=float,
        default=None,
        help="Máximo para tau_buy (default: 0.10)",
    )
    parser.add_argument(
        "--tau_sell_min",
        type=float,
        default=None,
        help="Mínimo para tau_sell (default: -0.02)",
    )
    parser.add_argument(
        "--tau_sell_max",
        type=float,
        default=None,
        help="Máximo para tau_sell (default: 0.02)",
    )
    parser.add_argument(
        "--dropout_min",
        type=float,
        default=None,
        help="Mínimo para dropout (default: 0.2)",
    )
    parser.add_argument(
        "--dropout_max",
        type=float,
        default=None,
        help="Máximo para dropout (default: 0.5)",
    )
    parser.add_argument(
        "--learning_rate_min",
        type=float,
        default=None,
        help="Mínimo para learning_rate (default: 1e-4)",
    )
    parser.add_argument(
        "--learning_rate_max",
        type=float,
        default=None,
        help="Máximo para learning_rate (default: 1e-2)",
    )
    parser.add_argument(
        "--weight_decay_min",
        type=float,
        default=None,
        help="Mínimo para weight_decay (default: 1e-6)",
    )
    parser.add_argument(
        "--weight_decay_max",
        type=float,
        default=None,
        help="Máximo para weight_decay (default: 1e-2)",
    )
    parser.add_argument(
        "--gru_units_1_min",
        type=int,
        default=None,
        help="Mínimo para la primera capa GRU (default: 32)",
    )
    parser.add_argument(
        "--gru_units_1_max",
        type=int,
        default=None,
        help="Máximo para la primera capa GRU (default: 128)",
    )
    parser.add_argument(
        "--gru_units_2_min",
        type=int,
        default=None,
        help="Mínimo para la segunda capa GRU (default: 16)",
    )
    parser.add_argument(
        "--gru_units_2_max",
        type=int,
        default=None,
        help="Máximo para la segunda capa GRU (default: 64)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default=None,
        help="Lista de batch sizes separados por coma (default: 32,64,128)",
    )
    
    args = parser.parse_args()
    
    root = project_root()
    config_path = root / args.config
    
    # Tickers
    tickers = None
    if args.tickers:
        # Modo lista explícita de tickers (máxima prioridad).
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            tickers = None
        else:
            print(f"✓ Override manual de tickers: {tickers}")
        if args.quick:
            print("⚠️  Ignorando --quick porque --tickers fue proporcionado")
    elif args.ticker:
        # Modo single-ticker.
        tickers = [args.ticker]
    elif args.quick:
        # Modo rápido: usar solo 3 tickers para prueba
        config = load_yaml(config_path)
        all_tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e1_conservative", [])
        import random
        random.seed(42)
        tickers = random.sample(all_tickers, min(3, len(all_tickers)))
        print(f"🚀 Modo rápido: usando {tickers}")
    
    # Overrides del espacio de búsqueda
    search_overrides: Dict[str, Dict[str, float]] = {}

    def maybe_add_override(name: str, min_value: Optional[float], max_value: Optional[float]) -> None:
        # Helper para no repetir lógica de carga de overrides por parámetro.
        if min_value is None and max_value is None:
            return
        search_overrides[name] = {}
        if min_value is not None:
            search_overrides[name]["min"] = min_value
        if max_value is not None:
            search_overrides[name]["max"] = max_value

    maybe_add_override("tau_buy", args.tau_buy_min, args.tau_buy_max)
    maybe_add_override("tau_sell", args.tau_sell_min, args.tau_sell_max)
    maybe_add_override("dropout", args.dropout_min, args.dropout_max)
    maybe_add_override("learning_rate", args.learning_rate_min, args.learning_rate_max)
    maybe_add_override("weight_decay", args.weight_decay_min, args.weight_decay_max)
    maybe_add_override("gru_units_1", args.gru_units_1_min, args.gru_units_1_max)
    maybe_add_override("gru_units_2", args.gru_units_2_min, args.gru_units_2_max)

    if search_overrides:
        print(f"✓ Overrides de espacio de búsqueda: {search_overrides}")

    batch_sizes_list: Optional[List[int]] = None
    if args.batch_sizes:
        try:
            batch_sizes_list = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError("--batch_sizes debe contener enteros separados por coma") from exc
        if not batch_sizes_list:
            batch_sizes_list = None
        else:
            print(f"✓ Batch sizes personalizados: {batch_sizes_list}")
    
    # Crear optimizador
    output_dir = root / args.output_dir

    def _study_name_for_ticker(base_study_name: str, ticker_name: str) -> str:
        # Evita mezclar trials de distintos tickers en el mismo estudio al correr single/per-ticker.
        suffix = f"__{ticker_name}"
        return base_study_name if base_study_name.endswith(suffix) else f"{base_study_name}{suffix}"

    t_start = time.time()

    if args.per_ticker:
        # Resolver tickers si no se especificaron (usa universo E1)
        if tickers is None:
            cfg = load_yaml(config_path)
            tickers = list(
                cfg.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e1_conservative", [])
            )
        if not tickers:
            raise ValueError("No hay tickers para optimizar (--tickers o config universo)")

        per_ticker_dir = output_dir / "by_ticker"
        per_ticker_dir.mkdir(parents=True, exist_ok=True)

        overrides_by_ticker: Dict[str, Dict[str, Dict[str, object]]] = {}
        meta_by_ticker: Dict[str, Dict[str, object]] = {}

        for t in tickers:
            print(f"\n{'='*80}")
            print(f"OPTIMIZACIÓN POR TICKER: {t}")
            print(f"{'='*80}\n")

            optimizer = E1HyperparameterOptimizer(
                config_path=config_path,
                tickers=[t],
                mlflow_tracking_uri=args.mlflow_uri,
                optuna_db_path=args.optuna_db,
                search_space_overrides=search_overrides or None,
                batch_size_choices=batch_sizes_list,
            )

            study_name = _study_name_for_ticker(args.study_name, t)
            study = optimizer.optimize(
                n_trials=args.n_trials,
                study_name=study_name,
                timeout=args.timeout,
            )

            out_ticker = per_ticker_dir / t
            optimizer.save_results(study, out_ticker)

            bp = study.best_params
            # Convertir a formato consumible por los pipelines de training
            overrides_by_ticker[t] = {
                "thresholds": {
                    "tau_buy": float(bp.get("tau_buy")),
                    "tau_sell": float(bp.get("tau_sell")),
                },
                "model": {
                    "gru_units": [int(bp.get("gru_units_1")), int(bp.get("gru_units_2"))],
                    "dropout": float(bp.get("dropout")),
                    "learning_rate": float(bp.get("learning_rate")),
                    "weight_decay": float(bp.get("weight_decay", 0.0)),
                    "batch_size": int(bp.get("batch_size")),
                    "early_stopping_patience": 10,
                },
            }
            meta_by_ticker[t] = {
                # Metadata útil para auditoría/seguimiento de calidad del tuning.
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        tuned_path = output_dir / "e1_tuned_params_by_ticker.meta.yaml"
        meta_payload = {
            "strategy": "e1_conservative",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "meta": meta_by_ticker,
            "tickers": overrides_by_ticker,
        }
        E1HyperparameterOptimizer._safe_write_file(
            tuned_path,
            lambda f: yaml.dump(meta_payload, f, default_flow_style=False, sort_keys=False),
        )

        tuned_compact_path = output_dir / "e1_tuned_params_by_ticker.yaml"
        E1HyperparameterOptimizer._safe_write_file(
            tuned_compact_path,
            lambda f: yaml.dump(overrides_by_ticker, f, default_flow_style=False, sort_keys=False),
        )

        print(f"\n✓ YAML de tuned params por ticker (con meta): {tuned_path}")
        print(f"✓ YAML consumible por training: {tuned_compact_path}")
        print("Para usarlo en training: export E1_TUNED_PARAMS_PATH=<ruta_al_yaml>")

    else:
        study_name = args.study_name
        if tickers and len(tickers) == 1:
            single_ticker = tickers[0]
            ticker_scoped_study_name = _study_name_for_ticker(args.study_name, single_ticker)
            if ticker_scoped_study_name != args.study_name:
                print(
                    f"✓ Modo single-ticker: usando study aislado por ticker para evitar contaminación de metadata: {ticker_scoped_study_name}"
                )
            study_name = ticker_scoped_study_name

        optimizer = E1HyperparameterOptimizer(
            config_path=config_path,
            tickers=tickers,
            mlflow_tracking_uri=args.mlflow_uri,
            optuna_db_path=args.optuna_db,
            search_space_overrides=search_overrides or None,
            batch_size_choices=batch_sizes_list,
        )

        study = optimizer.optimize(
            n_trials=args.n_trials,
            study_name=study_name,
            timeout=args.timeout,
        )

        optimizer.save_results(study, output_dir)

        has_completed_trials = True
        try:
            bp = study.best_params
            best_value = float(study.best_value)
            best_trial_number = int(study.best_trial.number)
        except Exception:
            has_completed_trials = False
            bp = {}
            best_value = None
            best_trial_number = None

        if tickers and has_completed_trials:
            converted = {
                "thresholds": {
                    "tau_buy": float(bp.get("tau_buy")),
                    "tau_sell": float(bp.get("tau_sell")),
                },
                "model": {
                    "gru_units": [int(bp.get("gru_units_1")), int(bp.get("gru_units_2"))],
                    "dropout": float(bp.get("dropout")),
                    "learning_rate": float(bp.get("learning_rate")),
                    "weight_decay": float(bp.get("weight_decay", 0.0)),
                    "batch_size": int(bp.get("batch_size")),
                    "early_stopping_patience": 10,
                },
            }

            # Sincronización por defecto: aplicar best global a todos los tickers evaluados.
            agg_compact_path = output_dir / "e1_tuned_params_by_ticker.yaml"
            try:
                with open(agg_compact_path, "r") as f:
                    agg_compact = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_compact = {}

            for ticker_name in tickers:
                agg_compact[ticker_name] = converted

            E1HyperparameterOptimizer._safe_write_file(
                agg_compact_path,
                lambda f: yaml.dump(agg_compact, f, default_flow_style=False, sort_keys=False),
            )

            agg_meta_path = output_dir / "e1_tuned_params_by_ticker.meta.yaml"
            try:
                with open(agg_meta_path, "r") as f:
                    agg_meta = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_meta = {}

            if not agg_meta:
                agg_meta = {
                    "strategy": "e1_conservative",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "meta": {},
                    "tickers": {},
                }

            for ticker_name in tickers:
                agg_meta.setdefault("meta", {})[ticker_name] = {
                    "study_name": study.study_name,
                    "best_value": best_value,
                    "best_trial": best_trial_number,
                    "n_trials": int(len(study.trials)),
                }
                agg_meta.setdefault("tickers", {})[ticker_name] = converted

            agg_meta["generated_at"] = pd.Timestamp.utcnow().isoformat()
            E1HyperparameterOptimizer._safe_write_file(
                agg_meta_path,
                lambda f: yaml.dump(agg_meta, f, default_flow_style=False, sort_keys=False),
            )

            print(f"\n✓ YAML por ticker sincronizado (best global): {agg_compact_path}")
            print(f"✓ YAML meta sincronizado: {agg_meta_path}")
        elif tickers:
            print("\n⚠️  No hay trials completos; no se sincronizó YAML por ticker.")

        # Si solo hay un ticker y no se usó --per_ticker, guardar también en by_ticker
        if tickers and len(tickers) == 1 and has_completed_trials:
            t = tickers[0]
            per_ticker_dir = output_dir / "by_ticker" / t
            per_ticker_dir.mkdir(parents=True, exist_ok=True)

            # best_params_e1.yaml por ticker
            per_ticker_path = per_ticker_dir / "best_params_e1.yaml"
            E1HyperparameterOptimizer._safe_write_file(
                per_ticker_path,
                lambda f: yaml.dump(converted, f, default_flow_style=False, sort_keys=False),
            )

            # Actualizar agregados compactos
            agg_compact_path = output_dir / "e1_tuned_params_by_ticker.yaml"
            try:
                with open(agg_compact_path, "r") as f:
                    agg_compact = yaml.safe_load(f) or {}
            except FileNotFoundError:
                # Primera corrida: archivo agregado todavía no existe.
                agg_compact = {}
            agg_compact[t] = converted
            E1HyperparameterOptimizer._safe_write_file(
                agg_compact_path,
                lambda f: yaml.dump(agg_compact, f, default_flow_style=False, sort_keys=False),
            )

            # Actualizar meta
            agg_meta_path = output_dir / "e1_tuned_params_by_ticker.meta.yaml"
            try:
                with open(agg_meta_path, "r") as f:
                    agg_meta = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_meta = {}
            if not agg_meta:
                # Estructura mínima si el archivo meta no existe o está vacío.
                agg_meta = {"strategy": "e1_conservative", "generated_at": pd.Timestamp.utcnow().isoformat(), "meta": {}, "tickers": {}}
            agg_meta.setdefault("meta", {})[t] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }
            agg_meta.setdefault("tickers", {})[t] = converted
            agg_meta["generated_at"] = pd.Timestamp.utcnow().isoformat()
            E1HyperparameterOptimizer._safe_write_file(
                agg_meta_path,
                lambda f: yaml.dump(agg_meta, f, default_flow_style=False, sort_keys=False),
            )

            print(f"\n✓ Parámetros por ticker guardados en: {per_ticker_path}")
            print(f"✓ YAML agregados actualizados: {agg_compact_path} y {agg_meta_path}")

        print(f"\n{'='*80}")
        print("MEJORES PARÁMETROS ENCONTRADOS")
        print(f"{'='*80}\n")

        if has_completed_trials:
            displayed = False
            for key in DISPLAY_PARAM_KEYS:
                if key in study.best_params:
                    print(f"  {key}: {study.best_params[key]}")
                    displayed = True
            if not displayed:
                # Fallback para estudios con estructura distinta (compatibilidad retroactiva).
                for key, value in study.best_params.items():
                    print(f"  {key}: {value}")

            print(f"\nMejor valor objetivo: {study.best_value:.6f}")
            print(f"Trial #{study.best_trial.number}")
        else:
            print("No hay trials completos; se guardó resumen parcial.")

        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*80}\n")

        print(f"📊 Resultados en: {output_dir}")
        print(f"📈 MLflow UI: {args.mlflow_uri}")
        print(f"🔍 Optuna Dashboard: optuna-dashboard {args.optuna_db}")

        elapsed = time.time() - t_start
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\n⏱  Tiempo total de optimización: {hours:02d}h {minutes:02d}m {seconds:02d}s ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

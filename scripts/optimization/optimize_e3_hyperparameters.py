#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Estrategia E3 usando Optuna + MLflow

Este script busca automáticamente los mejores hiperparámetros para la
estrategia E3 Intraday (LSTM Ensemble, 5-min bars, 30-min horizon).

Parámetros optimizados y rangos de búsqueda:
  Trading thresholds:
    - tau_buy:         [0.001, 0.005] step=0.0005 — Umbral señal de compra (≈ bps)
    - tau_sell:        [0.001, 0.005] step=0.0005 — Umbral señal de venta/short

  Arquitectura LSTM Ensemble:
    - lstm_hidden_size: [64, 256]  step=32  — Unidades capa LSTM oculta
    - dense_units:      [16, 64]   step=16  — Unidades capa densa intermedia
    - dropout:          [0.10, 0.40] step=0.05
    - ensemble_members: [2, 5]     step=1   — Miembros del ensemble
    - consensus_tol:    [0.0001, 0.002] step=0.0001 — Filtro de consenso (std señal)

  Entrenamiento:
    - learning_rate: [1e-4, 5e-3] log scale
    - weight_decay:  [1e-6, 1e-2] log scale
    - batch_size:    {64, 128, 256, 512} categórico

Métrica objetivo (alineada con lifecycle scoring weights):
  0.30*Sharpe + 0.20*IC + 0.25*DirAcc + 0.25*Calmar + trade_penalty
  (trade_penalty = -5 si <50% de tickers generan trades)

Uso:
    # Optimización con defaults (MLflow local SQLite)
    python -m scripts.optimization.optimize_e3_hyperparameters --n_trials 10

    # Optimizar un ticker específico
    python -m scripts.optimization.optimize_e3_hyperparameters --ticker SPY --n_trials 30

    # Optimización per-ticker (genera YAML de overrides por ticker)
    python -m scripts.optimization.optimize_e3_hyperparameters --per_ticker --n_trials 30

    # Usar servidor MLflow remoto (Docker)
    python -m scripts.optimization.optimize_e3_hyperparameters --n_trials 50 --mlflow_uri http://localhost:5050

Outputs:
- MLflow tracking: MLFLOW_TRACKING_URI (.env) o runs/mlflow_local/mlflow.db (SQLite local)
- Optuna database: runs/optuna_trials/optuna_studies.db (persistencia entre corridas)
- Mejores parámetros: reports/hyperparameter_optimization/best_params_e3.yaml
- Per-ticker overrides: reports/hyperparameter_optimization/e3_tuned_params_by_ticker.yaml
- Visualizaciones: reports/hyperparameter_optimization/figures/
"""

import argparse
import copy
import logging
import os
import shutil
import socket
import sys
import tempfile
from urllib.parse import urlparse
import time
from pathlib import Path
from typing import Dict, List, Optional
import warnings
from optuna.trial import TrialState

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.e3.train_pipeline import run_for_ticker as run_e3_for_ticker
from src.utils import load_yaml, project_root


DISPLAY_PARAM_KEYS = [
    # Orden de impresión "humano" para el resumen final de mejores parámetros.
    # Evita mostrar params legacy o desordenados de studies anteriores.
    "tau_buy",
    "tau_sell",
    "lstm_hidden_size",
    "dense_units",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "ensemble_members",
    "consensus_tol",
]

OPTIMIZED_PARAM_KEYS = [
    "tau_buy",
    "tau_sell",
    "lstm_hidden_size",
    "dense_units",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "ensemble_members",
    "consensus_tol",
]


class E3HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para estrategia E3 (LSTM Ensemble Intraday).
    """

    def __init__(
        self,
        config_path: Path,
        tickers: Optional[List[str]] = None,
        mlflow_tracking_uri: str = "local",
        optuna_db_path: str = "sqlite:///runs/optuna_trials/optuna_studies.db",
        search_space_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size_choices: Optional[List[int]] = None,
        validation_method: Optional[str] = None,
        validation_folds: Optional[int] = None,
        use_latest_data: bool = False,
    ):
        """
        Args:
            config_path: Path al archivo de configuración base
            tickers: Lista de tickers a usar (None = usar todos de E3)
            mlflow_tracking_uri: URI de MLflow tracking server (use 'local' para almacenamiento local)
            optuna_db_path: Path a base de datos de Optuna
            search_space_overrides: Overrides opcionales para rangos de búsqueda
            batch_size_choices: Opciones de batch size (default: [64, 128, 256, 512])
            validation_method: "walk_forward" | "time_split" (None = leer de config)
            validation_folds: Número de folds para walk_forward (None = leer de config)
            use_latest_data: Si True, cada trial descarga/extiende datos hasta hoy
        """
        self.config = load_yaml(config_path)
        self.root = project_root()

        # Validación: CLI > config > default
        optuna_val = self.config.get("optuna", {}).get("e3_intraday", {}).get("validation", {})
        self.validation_method = validation_method or optuna_val.get("method", "walk_forward")
        self.validation_folds = validation_folds or optuna_val.get("folds", 3)

        self.use_latest_data = use_latest_data

        # Reducir ruido de logs de librerías de infraestructura
        logging.getLogger("alembic").setLevel(logging.WARNING)
        logging.getLogger("mlflow").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

        # Tickers
        if tickers:
            self.tickers = tickers
        else:
            self.tickers = list(
                self.config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e3_intraday", [])
            )

        # MLflow - cascade fallback: remoto → SQLite local → file store
        self.mlflow_enabled = False
        experiment_name = "E3_Hyperparameter_Optimization"
        remote_check_timeout_seconds = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))

        local_sqlite_dir = self.root / "runs" / "mlflow_local"
        local_sqlite_dir.mkdir(parents=True, exist_ok=True)
        local_sqlite_db = local_sqlite_dir / "mlflow.db"
        local_artifacts_dir = local_sqlite_dir / "artifacts"
        local_artifacts_dir.mkdir(parents=True, exist_ok=True)
        local_sqlite_uri = f"sqlite:///{local_sqlite_db}"

        fallback_local_tracking_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip()
        if fallback_local_tracking_uri:
            local_sqlite_uri = fallback_local_tracking_uri

        local_mlruns = str(self.root / "mlruns")
        isolated_mlruns = str(self.root / "runs" / "e3_intraday" / "mlflow_store")
        Path(isolated_mlruns).mkdir(parents=True, exist_ok=True)

        base_mlruns_path = Path(local_mlruns)
        has_malformed_mlruns = False
        if base_mlruns_path.exists():
            for _exp_dir in base_mlruns_path.iterdir():
                if not _exp_dir.is_dir() or not _exp_dir.name.isdigit():
                    continue
                if not (_exp_dir / "meta.yaml").exists():
                    has_malformed_mlruns = True
                    break

        def _activate_mlflow(
            uri: str,
            experiment_artifact_dir: Path | None = None,
        ) -> tuple[bool, str | None]:
            """Activa MLflow con la URI dada. Retorna (ok, error_msg)."""
            try:
                mlflow.set_tracking_uri(uri)
                if experiment_artifact_dir is not None:
                    exp = mlflow.get_experiment_by_name(experiment_name)
                    if exp is None:
                        mlflow.create_experiment(
                            experiment_name,
                            artifact_location=experiment_artifact_dir.resolve().as_uri(),
                        )
                    elif exp.lifecycle_stage == "deleted":
                        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
                    mlflow.set_experiment(experiment_name)
                else:
                    exp = mlflow.get_experiment_by_name(experiment_name)
                    if exp is not None and exp.lifecycle_stage == "deleted":
                        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
                    mlflow.set_experiment(experiment_name)
                return True, None
            except Exception as exc:
                return False, str(exc)

        def _is_tracking_uri_reachable(uri: str, timeout_seconds: float) -> tuple[bool, str | None]:
            """Chequeo rápido de conectividad (solo para URIs http/https)."""
            parsed = urlparse(uri)
            if parsed.scheme not in {"http", "https"}:
                return True, None
            host = parsed.hostname
            if not host:
                return True, None
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                with socket.create_connection((host, port), timeout=timeout_seconds):
                    return True, None
            except OSError as exc:
                return False, str(exc)

        _local_uris = [
            {"uri": local_sqlite_uri, "store_path": str(local_sqlite_db), "artifact_dir": local_artifacts_dir},
            {"uri": f"file://{isolated_mlruns}", "store_path": isolated_mlruns, "artifact_dir": None},
            {"uri": f"file://{local_mlruns}", "store_path": local_mlruns, "artifact_dir": None},
        ] if has_malformed_mlruns else [
            {"uri": local_sqlite_uri, "store_path": str(local_sqlite_db), "artifact_dir": local_artifacts_dir},
            {"uri": f"file://{local_mlruns}", "store_path": local_mlruns, "artifact_dir": None},
            {"uri": f"file://{isolated_mlruns}", "store_path": isolated_mlruns, "artifact_dir": None},
        ]

        remote_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if mlflow_tracking_uri != "local":
            remote_uri = mlflow_tracking_uri

        if remote_uri:
            reachable, reach_err = _is_tracking_uri_reachable(remote_uri, remote_check_timeout_seconds)
            if reachable:
                ok, err = _activate_mlflow(remote_uri)
                if ok:
                    self.mlflow_enabled = True
                    print(f"✓ MLflow habilitado: {remote_uri} (experiment={experiment_name})")
                else:
                    print(f"⚠️  MLflow servidor no disponible ({err})")
            else:
                print(
                    f"⚠️  MLflow remoto no accesible "
                    f"({remote_uri}, timeout={remote_check_timeout_seconds}s): {reach_err}"
                )

        if not self.mlflow_enabled:
            # Cascade local: sqlite → file store
            for _cfg in _local_uris:
                ok, err = _activate_mlflow(_cfg["uri"], experiment_artifact_dir=_cfg["artifact_dir"])
                if ok:
                    self.mlflow_enabled = True
                    mode = "local fallback" if remote_uri else "tracking local"
                    print(f"✓ MLflow habilitado ({mode}, experiment={experiment_name})")
                    print(f"  → Store URI: {_cfg['store_path']}")
                    print(f"  → Para visualizar: mlflow ui --backend-store-uri {_cfg['store_path']}")
                    break
                print(f"⚠️  Falló MLflow local en {_cfg['store_path']}: {err}")

        if not self.mlflow_enabled:
            print("⚠️  MLflow no disponible, continuando sin tracking")

        # Optuna
        self.optuna_db_path = optuna_db_path
        self.current_run_min_trial_number: int | None = None

        # Search space defaults (se pueden sobreescribir vía CLI)
        self.tau_buy_bounds = {"min": 0.001, "max": 0.005, "step": 0.0005}
        self.tau_sell_bounds = {"min": 0.001, "max": 0.005, "step": 0.0005}
        self.lstm_hidden_size_bounds = {"min": 64, "max": 256, "step": 32}
        self.dense_units_bounds = {"min": 16, "max": 64, "step": 16}
        self.dropout_bounds = {"min": 0.1, "max": 0.4, "step": 0.05}
        self.learning_rate_bounds = {"min": 1e-4, "max": 5e-3, "log": True}
        self.weight_decay_bounds = {"min": 1e-6, "max": 1e-2, "log": True}
        self.ensemble_members_bounds = {"min": 2, "max": 5, "step": 1}
        self.consensus_tol_bounds = {"min": 0.0001, "max": 0.002, "step": 0.0001}

        default_batch_sizes = batch_size_choices or [64, 128, 256, 512]
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
        update_bounds(self.lstm_hidden_size_bounds, "lstm_hidden_size")
        update_bounds(self.dense_units_bounds, "dense_units")
        update_bounds(self.dropout_bounds, "dropout")
        update_bounds(self.learning_rate_bounds, "learning_rate")
        update_bounds(self.weight_decay_bounds, "weight_decay")
        update_bounds(self.ensemble_members_bounds, "ensemble_members")
        update_bounds(self.consensus_tol_bounds, "consensus_tol")

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
        validate_bounds(self.lstm_hidden_size_bounds, "lstm_hidden_size")
        validate_bounds(self.dense_units_bounds, "dense_units")
        validate_bounds(self.dropout_bounds, "dropout")
        validate_bounds(self.learning_rate_bounds, "learning_rate")
        validate_bounds(self.weight_decay_bounds, "weight_decay")
        validate_bounds(self.ensemble_members_bounds, "ensemble_members")
        validate_bounds(self.consensus_tol_bounds, "consensus_tol")

        print(f"✓ Inicializado optimizador E3 para {len(self.tickers)} tickers")
        print(f"  Tickers: {', '.join(self.tickers[:5])}{'...' if len(self.tickers) > 5 else ''}")
        print(f"  Validación: {self.validation_method}" + (f" ({self.validation_folds} folds)" if self.validation_method == "walk_forward" else ""))
        print(f"  MLflow: {mlflow_tracking_uri}")
        print(f"  Optuna DB: {optuna_db_path}")

    @staticmethod
    def _suggest_float(trial: optuna.Trial, name: str, bounds: Dict[str, float]) -> float:
        """Sugiere un float según el espacio de búsqueda definido en bounds.

        Args:
            trial: Trial activo de Optuna.
            name: Nombre del hiperparámetro (clave en el trial).
            bounds: Dict con min, max y opcionalmente step (uniforme discreta)
                o log=True (log-uniforme). No se pueden combinar ambos.

        Returns:
            Valor sugerido por el sampler de Optuna.

        Raises:
            ValueError: Si se combinan step y log=True (Optuna no lo soporta).
        """
        low = float(bounds["min"])
        high = float(bounds["max"])
        step = bounds.get("step")
        log_flag = bool(bounds.get("log"))

        if log_flag:
            if step:
                raise ValueError(f"No se puede usar step con distribución log para {name}")
            return trial.suggest_float(name, low, high, log=True)

        if step:
            return trial.suggest_float(name, low, high, step=float(step))

        return trial.suggest_float(name, low, high)

    @staticmethod
    def _suggest_int(trial: optuna.Trial, name: str, bounds: Dict[str, float]) -> int:
        """Sugiere un entero dentro del rango especificado en bounds.

        Args:
            trial: Trial activo de Optuna.
            name: Nombre del hiperparámetro (clave en el trial).
            bounds: Dict con min, max y opcionalmente step (paso de discretización).

        Returns:
            Valor entero sugerido por el sampler de Optuna.
        """
        low = int(bounds["min"])
        high = int(bounds["max"])
        step = bounds.get("step")

        if step:
            return trial.suggest_int(name, low, high, step=int(step))

        return trial.suggest_int(name, low, high)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para Optuna.

        Retorna: 0.30*Sharpe + 0.20*IC + 0.25*DirAcc + 0.25*Calmar + trade_penalty
        """

        from contextlib import nullcontext
        trial_start = time.perf_counter()
        ctx = mlflow.start_run(run_name=f"trial_{trial.number}") if self.mlflow_enabled else nullcontext()
        with ctx:

            # 1. SUGERIR HIPERPARÁMETROS

            tau_buy = self._suggest_float(trial, "tau_buy", self.tau_buy_bounds)
            tau_sell = self._suggest_float(trial, "tau_sell", self.tau_sell_bounds)

            lstm_hidden_size = self._suggest_int(trial, "lstm_hidden_size", self.lstm_hidden_size_bounds)
            dense_units = self._suggest_int(trial, "dense_units", self.dense_units_bounds)

            dropout = self._suggest_float(trial, "dropout", self.dropout_bounds)
            learning_rate = self._suggest_float(trial, "learning_rate", self.learning_rate_bounds)
            weight_decay = self._suggest_float(trial, "weight_decay", self.weight_decay_bounds)

            batch_size = trial.suggest_categorical("batch_size", self.batch_size_choices)

            ensemble_members = self._suggest_int(trial, "ensemble_members", self.ensemble_members_bounds)
            consensus_tol = self._suggest_float(trial, "consensus_tol", self.consensus_tol_bounds)

            early_stopping_patience = 7

            # 2. ACTUALIZAR CONFIG CON PARÁMETROS SUGERIDOS

            config_trial = copy.deepcopy(self.config)

            splits_cfg = dict(config_trial.get("splits", {}))
            splits_cfg["method"] = self.validation_method
            if self.validation_method == "walk_forward":
                splits_cfg["folds"] = self.validation_folds
            config_trial["splits"] = splits_cfg

            config_trial["strategies"]["e3_intraday"]["thresholds"] = {
                "tau_buy": tau_buy,
                "tau_sell": tau_sell,
            }

            config_trial["strategies"]["e3_intraday"]["model"] = {
                **config_trial["strategies"]["e3_intraday"].get("model", {}),
                "lstm_hidden_size": lstm_hidden_size,
                "dense_units": dense_units,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "ensemble_members": ensemble_members,
                "consensus_tol": consensus_tol,
                "early_stopping_patience": early_stopping_patience,
            }

            # 3. ENTRENAR PARA TODOS LOS TICKERS

            print(
                f"\n{'─'*70}\n"
                f"[Trial {trial.number}] Params: "
                f"tau=({tau_buy:.4f},{tau_sell:.4f}) "
                f"lstm={lstm_hidden_size} dense={dense_units} do={dropout:.2f} "
                f"lr={learning_rate:.2e} wd={weight_decay:.2e} bs={batch_size} "
                f"ens={ensemble_members} ctol={consensus_tol:.4f} "
                f"val={self.validation_method}({self.validation_folds}f)",
                flush=True,
            )

            results = []
            raw_dir = self.root / "data/raw/intraday"

            out_dir = self.root / "runs/optuna_trials" / f"e3_trial_{trial.number}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for ticker_idx, ticker in enumerate(self.tickers, start=1):
                ticker_start = time.perf_counter()
                print(
                    f"\n  [{ticker_idx}/{len(self.tickers)}] {ticker} — entrenando...",
                    flush=True,
                )
                try:
                    os.environ.pop("TUNED_PARAMS_PATH", None)
                    result = run_e3_for_ticker(
                        config=config_trial,
                        ticker=ticker,
                        raw_dir=raw_dir,
                        out_dir=out_dir,
                        register_lifecycle=False,
                        use_latest_data=self.use_latest_data,
                    )

                    ticker_secs = time.perf_counter() - ticker_start
                    ic_t = result.get("ml_ic", float("nan"))
                    sharpe_t = result.get("bt_sharpe", float("nan"))
                    calmar_t = result.get("bt_calmar", float("nan"))
                    dir_acc_t = result.get("ml_directional_accuracy", float("nan"))
                    num_trades_t = result.get("bt_num_trades", 0)
                    print(
                        f"  ✓ {ticker} ({ticker_secs:.0f}s) — "
                        f"IC={ic_t:+.3f} Sharpe={sharpe_t:+.3f} "
                        f"Calmar={calmar_t:+.2f} DirAcc={dir_acc_t:.3f} "
                        f"trades={num_trades_t}",
                        flush=True,
                    )

                    results.append({
                        "ticker": ticker,
                        "ic": ic_t,
                        "sharpe": sharpe_t,
                        "calmar": calmar_t,
                        "directional_accuracy": dir_acc_t,
                        "num_trades": num_trades_t,
                        "win_rate": result.get("bt_win_rate", 0),
                        "max_dd": result.get("bt_max_drawdown", 0),
                    })

                except Exception as e:
                    ticker_secs = time.perf_counter() - ticker_start
                    print(f"  ⚠️  Error en {ticker} ({ticker_secs:.0f}s): {e}", flush=True)
                    results.append({
                        "ticker": ticker,
                        "ic": -1.0,
                        "sharpe": -10.0,
                        "calmar": 0.0,
                        "directional_accuracy": 0.0,
                        "num_trades": 0,
                        "win_rate": 0,
                        "max_dd": 1.0,
                    })

            # 4. CALCULAR MÉTRICAS AGREGADAS

            df_results = pd.DataFrame(results)

            ic_mean = df_results["ic"].mean()
            ic_median = df_results["ic"].median()
            sharpe_mean = df_results["sharpe"].mean()
            sharpe_median = df_results["sharpe"].median()
            calmar_mean = df_results["calmar"].mean()
            dir_acc_mean = df_results["directional_accuracy"].mean()

            ic_std = df_results["ic"].std()
            sharpe_std = df_results["sharpe"].std()

            ic_positive_pct = (df_results["ic"] > 0.05).sum() / len(df_results) * 100
            sharpe_positive_pct = (df_results["sharpe"] > 0).sum() / len(df_results) * 100

            tickers_with_trades = (df_results["num_trades"] > 0).sum()
            avg_num_trades = df_results["num_trades"].mean()

            # 5. REGISTRAR EN MLFLOW

            if self.mlflow_enabled:
                mlflow.log_params({
                    "tau_buy": tau_buy,
                    "tau_sell": tau_sell,
                    "lstm_hidden_size": lstm_hidden_size,
                    "dense_units": dense_units,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "ensemble_members": ensemble_members,
                    "consensus_tol": consensus_tol,
                    "trial_number": trial.number,
                    "validation_method": self.validation_method,
                    "validation_folds": self.validation_folds if self.validation_method == "walk_forward" else 1,
                })

                mlflow.log_metrics({
                    "ic_mean": ic_mean,
                    "ic_median": ic_median,
                    "ic_std": ic_std,
                    "sharpe_mean": sharpe_mean,
                    "sharpe_median": sharpe_median,
                    "sharpe_std": sharpe_std,
                    "calmar_mean": calmar_mean,
                    "directional_accuracy_mean": dir_acc_mean,
                    "ic_positive_pct": ic_positive_pct,
                    "sharpe_positive_pct": sharpe_positive_pct,
                    "tickers_with_trades": tickers_with_trades,
                    "avg_num_trades": avg_num_trades,
                })

            results_path = out_dir / "trial_results.csv"
            try:
                self._safe_write_file(results_path, lambda f: df_results.to_csv(f, index=False))
            except Exception as write_exc:
                print(f"  ⚠️  No se pudo escribir trial_results.csv: {write_exc}")
            if self.mlflow_enabled:
                try:
                    mlflow.log_artifact(str(results_path))
                except Exception as art_exc:
                    print(f"  ⚠️  MLflow artifact no guardado: {art_exc}")

            # 6. DEFINIR MÉTRICA OBJETIVO
            # Pesos alineados con lifecycle scoring weights de E3 en base.yaml:
            # bt_sharpe=0.30, ml_ic=0.20, ml_directional_accuracy=0.25, bt_calmar=0.25
            trade_penalty = 0 if tickers_with_trades >= len(self.tickers) * 0.5 else -5

            objective_raw = (
                0.30 * sharpe_mean +
                0.20 * ic_mean +
                0.25 * dir_acc_mean +
                0.25 * calmar_mean
            )
            objective_value = objective_raw + trade_penalty

            trial.set_user_attr("objective_raw", float(objective_raw))
            trial.set_user_attr("trade_penalty", float(trade_penalty))

            if self.mlflow_enabled:
                mlflow.log_metric("objective_value", objective_value)
                mlflow.log_metric("objective_raw", objective_raw)
                mlflow.log_metric("trade_penalty", trade_penalty)
                mlflow.log_metric("trial_duration_seconds", time.perf_counter() - trial_start)

            trial_seconds = time.perf_counter() - trial_start
            print(
                f"[Trial {trial.number:04d}] obj={objective_value:+.4f} "
                f"sharpe={sharpe_mean:+.4f} ic={ic_mean:+.4f} "
                f"calmar={calmar_mean:+.4f} dir_acc={dir_acc_mean:.4f} "
                f"trades={tickers_with_trades}/{len(self.tickers)} "
                f"time={trial_seconds:.1f}s | "
                f"tau=({tau_buy:.4f},{tau_sell:.4f}) "
                f"lstm={lstm_hidden_size} dense={dense_units} do={dropout:.2f} "
                f"lr={learning_rate:.2e} wd={weight_decay:.2e} bs={batch_size} "
                f"ens={ensemble_members} ctol={consensus_tol:.4f}"
            )

            return objective_value

    def optimize(
        self,
        n_trials: int = 50,
        study_name: str = "e3_hyperparameter_optimization",
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
        print(f"OPTIMIZACIÓN DE HIPERPARÁMETROS E3 INTRADAY")
        print(f"{'='*80}")
        print(f"Trials: {n_trials}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Study: {study_name}")
        print(f"{'='*80}\n")

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=study_name,
            storage=self.optuna_db_path,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        existing_numbers = [trial.number for trial in study.trials]
        self.current_run_min_trial_number = (max(existing_numbers) + 1) if existing_numbers else 0

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        return study

    @staticmethod
    def _safe_write_file(path: Path, write_fn, *, retries: int = 3, delay: float = 2.0) -> None:
        """Escribe un archivo con retry logic para manejar timeouts de iCloud/file-provider."""
        for attempt in range(1, retries + 1):
            try:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=path.suffix,
                )
                with os.fdopen(tmp_fd, "w") as f:
                    write_fn(f)
                shutil.move(tmp_path, str(path))
                return
            except (TimeoutError, OSError) as exc:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                if attempt < retries:
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
        Guarda resultados de optimización: YAML, CSVs, visualizaciones, resumen.
        """

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("GUARDANDO RESULTADOS")
        print(f"{'='*80}\n")

        completed_trials = [
            t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None
        ]
        has_completed_trials = len(completed_trials) > 0

        best_params: Dict = {}
        best_value: Optional[float] = None
        best_trial_number: Optional[int] = None

        if has_completed_trials:
            best_params = study.best_params
            best_value = float(study.best_value)
            best_trial_number = study.best_trial.number

            best_params_yaml = {
                "optimization": {
                    "study_name": study.study_name,
                    "n_trials": len(study.trials),
                    "n_completed_trials": len(completed_trials),
                    "best_value": best_value,
                    "best_trial": best_trial_number,
                },
                "best_params": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in best_params.items()
                },
            }

            yaml_path = output_dir / "best_params_e3.yaml"
            self._safe_write_file(yaml_path, lambda f: yaml.dump(best_params_yaml, f, default_flow_style=False, sort_keys=False))
            print(f"✓ Mejores parámetros guardados: {yaml_path}")
        else:
            print(
                "⚠️  No hay trials completos aún; se omite best_params_e3.yaml "
                "(se guardan trials y resumen parcial)."
            )

        # CSVs de trials
        df_trials = study.trials_dataframe()
        csv_path = output_dir / "e3_all_trials.csv"
        self._safe_write_file(csv_path, lambda f: df_trials.to_csv(f, index=False))
        print(f"✓ Todos los trials guardados: {csv_path}")

        # Visualizaciones
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        if self.current_run_min_trial_number is not None:
            df_run = df_trials[df_trials["number"] >= self.current_run_min_trial_number].copy()
        else:
            df_run = df_trials.copy()
        df_run_complete = df_run[df_run["state"] == "COMPLETE"].copy()

        run_csv_path = output_dir / "e3_run_trials.csv"
        self._safe_write_file(run_csv_path, lambda f: df_run.to_csv(f, index=False))

        df_all_complete = df_trials[df_trials["state"] == "COMPLETE"].copy()

        # Optimization history (estudio completo, objetivo penalizado)
        try:
            import plotly.graph_objects as go

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
            fig.write_image(str(figures_dir / "e3_optimization_history.png"), width=1200, height=600)
            print(f"✓ Gráfico: e3_optimization_history.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history: {e}")

        # Optimization history (solo corrida actual)
        try:
            import plotly.graph_objects as go

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
            fig.write_image(str(figures_dir / "e3_optimization_history_current_run.png"), width=1200, height=600)
            print(f"✓ Gráfico: e3_optimization_history_current_run.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_current_run: {e}")

        # Optimization history sin penalización (estudio completo)
        try:
            import plotly.graph_objects as go

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
                yaxis_title="Objective Raw (0.30*Sharpe + 0.20*IC + 0.25*DirAcc + 0.25*Calmar)",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "e3_optimization_history_raw.png"), width=1200, height=600)
            print(f"✓ Gráfico: e3_optimization_history_raw.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_raw: {e}")

        # Optimization history sin penalización (solo corrida actual)
        try:
            import plotly.graph_objects as go

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
                yaxis_title="Objective Raw (0.30*Sharpe + 0.20*IC + 0.25*DirAcc + 0.25*Calmar)",
                template="plotly_white",
            )
            fig.write_image(str(figures_dir / "e3_optimization_history_raw_current_run.png"), width=1200, height=600)
            print(f"✓ Gráfico: e3_optimization_history_raw_current_run.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history_raw_current_run: {e}")

        # Parameter importances
        try:
            import plotly.graph_objects as go

            try:
                importances = optuna.importance.get_param_importances(
                    study,
                    params=OPTIMIZED_PARAM_KEYS,
                )
            except Exception:
                importances = {}

            full_importances = {k: float(importances.get(k, 0.0)) for k in OPTIMIZED_PARAM_KEYS}

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
            fig.write_image(str(figures_dir / "e3_param_importances.png"), width=1200, height=700)
            print(f"✓ Gráfico: e3_param_importances.png")
        except Exception as e:
            print(f"⚠️  Error generando param_importances: {e}")

        # Parallel coordinate
        try:
            fig = plot_parallel_coordinate(study)
            fig.write_image(str(figures_dir / "e3_parallel_coordinate.png"), width=1400, height=800)
            print(f"✓ Gráfico: e3_parallel_coordinate.png")
        except Exception as e:
            print(f"⚠️  Error generando parallel_coordinate: {e}")

        # Resumen en texto
        summary_path = output_dir / "e3_optimization_summary.txt"

        def _write_summary(f):
            f.write("="*80 + "\n")
            f.write("OPTIMIZACIÓN DE HIPERPARÁMETROS E3 INTRADAY - RESUMEN\n")
            f.write("="*80 + "\n\n")
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Completed trials: {len(completed_trials)}\n")

            if has_completed_trials:
                f.write(f"Best trial: #{best_trial_number}\n")
                f.write(f"Best value: {best_value:.6f}\n\n")
                f.write("Mejores parámetros:\n")
                f.write("-" * 40 + "\n")
                for key, value in best_params.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write("Best trial: N/A\n")
                f.write("Best value: N/A\n\n")
                f.write(
                    "No se detectaron trials en estado COMPLETE. "
                    "Ejecutar más trials o revisar interrupciones/errores.\n"
                )

            f.write("\n" + "="*80 + "\n")
            f.write("TOP 10 TRIALS\n")
            f.write("="*80 + "\n\n")

            if "value" in df_trials.columns and not df_trials["value"].dropna().empty:
                df_top = df_trials.nlargest(10, "value")
                f.write(df_top.to_string(index=False))
            else:
                f.write("No hay valores objetivo disponibles aún.\n")

        self._safe_write_file(summary_path, _write_summary)
        print(f"✓ Resumen guardado: {summary_path}")

        print(f"\n{'='*80}")
        print("RESULTADOS GUARDADOS EXITOSAMENTE")
        print(f"{'='*80}\n")


def _best_params_to_e3_tuned_schema(best_params: Dict) -> Dict:
    """Convierte best_params flat de Optuna al esquema nested consumido por E3."""
    schema: Dict[str, Dict] = {
        "thresholds": {},
        "model": {},
    }

    if "tau_buy" in best_params:
        schema["thresholds"]["tau_buy"] = float(best_params["tau_buy"])
    if "tau_sell" in best_params:
        schema["thresholds"]["tau_sell"] = float(best_params["tau_sell"])

    model_keys = [
        "lstm_hidden_size",
        "dense_units",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "ensemble_members",
        "consensus_tol",
    ]
    for key in model_keys:
        if key in best_params:
            val = best_params[key]
            if key in ("lstm_hidden_size", "dense_units", "batch_size", "ensemble_members"):
                schema["model"][key] = int(val)
            else:
                schema["model"][key] = float(val)

    schema["model"]["early_stopping_patience"] = 7

    return {k: v for k, v in schema.items() if v}


def main():
    parser = argparse.ArgumentParser(
        description="Optimización de hiperparámetros E3 Intraday con Optuna + MLflow"
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
        default="e3_hyperparameter_optimization",
        help="Nombre del estudio Optuna (default: e3_hyperparameter_optimization)",
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
        help="Lista de tickers separados por coma (overridea universo E3)",
    )
    parser.add_argument(
        "--per_ticker",
        action="store_true",
        help="Optimiza un estudio independiente por ticker y genera un YAML con overrides por ticker",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: usar solo 2 tickers aleatorios",
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
        help="Path a base de datos de Optuna",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/hyperparameter_optimization",
        help="Directorio de salida para resultados",
    )
    # Search space overrides — thresholds
    parser.add_argument("--tau_buy_min", type=float, default=None, help="Mínimo para tau_buy (default: 0.001)")
    parser.add_argument("--tau_buy_max", type=float, default=None, help="Máximo para tau_buy (default: 0.005)")
    parser.add_argument("--tau_sell_min", type=float, default=None, help="Mínimo para tau_sell (default: 0.001)")
    parser.add_argument("--tau_sell_max", type=float, default=None, help="Máximo para tau_sell (default: 0.005)")
    # Search space overrides — arquitectura
    parser.add_argument("--lstm_hidden_size_min", type=int, default=None, help="Mínimo para lstm_hidden_size (default: 64)")
    parser.add_argument("--lstm_hidden_size_max", type=int, default=None, help="Máximo para lstm_hidden_size (default: 256)")
    parser.add_argument("--dense_units_min", type=int, default=None, help="Mínimo para dense_units (default: 16)")
    parser.add_argument("--dense_units_max", type=int, default=None, help="Máximo para dense_units (default: 64)")
    parser.add_argument("--dropout_min", type=float, default=None, help="Mínimo para dropout (default: 0.1)")
    parser.add_argument("--dropout_max", type=float, default=None, help="Máximo para dropout (default: 0.4)")
    # Search space overrides — entrenamiento
    parser.add_argument("--learning_rate_min", type=float, default=None, help="Mínimo para learning_rate (default: 1e-4)")
    parser.add_argument("--learning_rate_max", type=float, default=None, help="Máximo para learning_rate (default: 5e-3)")
    parser.add_argument("--weight_decay_min", type=float, default=None, help="Mínimo para weight_decay (default: 1e-6)")
    parser.add_argument("--weight_decay_max", type=float, default=None, help="Máximo para weight_decay (default: 1e-2)")
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default=None,
        help="Lista de batch sizes separados por coma (default: 64,128,256,512)",
    )
    # Search space overrides — ensemble
    parser.add_argument("--ensemble_members_min", type=int, default=None, help="Mínimo para ensemble_members (default: 2)")
    parser.add_argument("--ensemble_members_max", type=int, default=None, help="Máximo para ensemble_members (default: 5)")
    parser.add_argument("--consensus_tol_min", type=float, default=None, help="Mínimo para consensus_tol (default: 0.0001)")
    parser.add_argument("--consensus_tol_max", type=float, default=None, help="Máximo para consensus_tol (default: 0.002)")
    # Validación
    parser.add_argument(
        "--validation_method",
        type=str,
        choices=["walk_forward", "time_split"],
        default=None,
        help="Método de validación en trials (default: leer de base.yaml, walk_forward)",
    )
    parser.add_argument(
        "--optuna_folds",
        type=int,
        default=None,
        help="Folds para walk_forward en Optuna (default: leer de base.yaml, 3)",
    )
    parser.add_argument(
        "--use-latest-data",
        action="store_true",
        help="Propaga use_latest_data=True a cada trial",
    )

    args = parser.parse_args()

    root = project_root()
    config_path = root / args.config

    # Tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        tickers = list(dict.fromkeys(tickers))
        if not tickers:
            tickers = None
        else:
            print(f"✓ Override manual de tickers: {tickers}")
        if args.quick:
            print("⚠️  Ignorando --quick porque --tickers fue proporcionado")
    elif args.ticker:
        tickers = [args.ticker]
    elif args.quick:
        config = load_yaml(config_path)
        all_tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e3_intraday", [])
        import random
        random.seed(42)
        tickers = random.sample(all_tickers, min(2, len(all_tickers)))
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
    maybe_add_override("lstm_hidden_size", args.lstm_hidden_size_min, args.lstm_hidden_size_max)
    maybe_add_override("dense_units", args.dense_units_min, args.dense_units_max)
    maybe_add_override("dropout", args.dropout_min, args.dropout_max)
    maybe_add_override("learning_rate", args.learning_rate_min, args.learning_rate_max)
    maybe_add_override("weight_decay", args.weight_decay_min, args.weight_decay_max)
    maybe_add_override("ensemble_members", args.ensemble_members_min, args.ensemble_members_max)
    maybe_add_override("consensus_tol", args.consensus_tol_min, args.consensus_tol_max)

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

    output_dir = root / args.output_dir

    def _study_name_for_ticker(base_study_name: str, ticker_name: str) -> str:
        # Evita mezclar trials de distintos tickers en el mismo estudio al correr single/per-ticker.
        suffix = f"__{ticker_name}"
        return base_study_name if base_study_name.endswith(suffix) else f"{base_study_name}{suffix}"

    t_start = time.time()

    if args.per_ticker:
        if tickers is None:
            cfg = load_yaml(config_path)
            tickers = list(
                cfg.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e3_intraday", [])
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

            optimizer = E3HyperparameterOptimizer(
                config_path=config_path,
                tickers=[t],
                mlflow_tracking_uri=args.mlflow_uri,
                optuna_db_path=args.optuna_db,
                search_space_overrides=search_overrides or None,
                batch_size_choices=batch_sizes_list,
                validation_method=args.validation_method,
                validation_folds=args.optuna_folds,
                use_latest_data=args.use_latest_data,
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
            overrides_by_ticker[t] = _best_params_to_e3_tuned_schema(bp)
            meta_by_ticker[t] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        tuned_path = output_dir / "e3_tuned_params_by_ticker.meta.yaml"
        meta_payload = {
            "strategy": "e3_intraday",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "meta": meta_by_ticker,
            "tickers": overrides_by_ticker,
        }
        E3HyperparameterOptimizer._safe_write_file(
            tuned_path,
            lambda f: yaml.dump(meta_payload, f, default_flow_style=False, sort_keys=False),
        )

        tuned_compact_path = output_dir / "e3_tuned_params_by_ticker.yaml"
        E3HyperparameterOptimizer._safe_write_file(
            tuned_compact_path,
            lambda f: yaml.dump(overrides_by_ticker, f, default_flow_style=False, sort_keys=False),
        )

        print(f"\n✓ YAML de tuned params por ticker (con meta): {tuned_path}")
        print(f"✓ YAML consumible por training: {tuned_compact_path}")
        print("El path ya está registrado en base.yaml > optuna > e3_intraday > tuned_params_path")

    else:
        study_name = args.study_name
        if tickers and len(tickers) == 1:
            single_ticker = tickers[0]
            ticker_scoped_study_name = _study_name_for_ticker(args.study_name, single_ticker)
            if ticker_scoped_study_name != args.study_name:
                print(
                    f"✓ Modo single-ticker: usando study aislado por ticker: {ticker_scoped_study_name}"
                )
            study_name = ticker_scoped_study_name

        optimizer = E3HyperparameterOptimizer(
            config_path=config_path,
            tickers=tickers,
            mlflow_tracking_uri=args.mlflow_uri,
            optuna_db_path=args.optuna_db,
            search_space_overrides=search_overrides or None,
            batch_size_choices=batch_sizes_list,
            validation_method=args.validation_method,
            validation_folds=args.optuna_folds,
            use_latest_data=args.use_latest_data,
        )

        study = None
        try:
            study = optimizer.optimize(
                n_trials=args.n_trials,
                study_name=study_name,
                timeout=args.timeout,
            )
        except KeyboardInterrupt:
            print("\n⚠️  Optimización interrumpida por usuario. Guardando resultados parciales...")
            study = optuna.load_study(study_name=study_name, storage=args.optuna_db)

        if study is None:
            study = optuna.load_study(study_name=study_name, storage=args.optuna_db)

        optimizer.save_results(study, output_dir)

        completed_trials = [
            trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None
        ]
        has_completed_trials = len(completed_trials) > 0
        converted: Dict[str, Dict] = {}

        if tickers and has_completed_trials:
            bp = study.best_params
            converted = _best_params_to_e3_tuned_schema(bp)

            per_ticker_tuned = {
                t for t in tickers
                if (output_dir / "by_ticker" / t / "best_params_e3.yaml").exists()
            }
            agg_compact_path = output_dir / "e3_tuned_params_by_ticker.yaml"
            try:
                with open(agg_compact_path, "r") as f:
                    agg_compact = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_compact = {}

            for ticker_name in tickers:
                if ticker_name in per_ticker_tuned:
                    continue
                agg_compact[ticker_name] = converted

            E3HyperparameterOptimizer._safe_write_file(
                agg_compact_path,
                lambda f: yaml.dump(agg_compact, f, default_flow_style=False, sort_keys=False),
            )

            agg_meta_path = output_dir / "e3_tuned_params_by_ticker.meta.yaml"
            try:
                with open(agg_meta_path, "r") as f:
                    agg_meta = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_meta = {}

            if not agg_meta:
                agg_meta = {
                    "strategy": "e3_intraday",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "meta": {},
                    "tickers": {},
                }

            for ticker_name in tickers:
                if ticker_name in per_ticker_tuned:
                    continue
                agg_meta.setdefault("meta", {})[ticker_name] = {
                    "study_name": study.study_name,
                    "best_value": float(study.best_value),
                    "best_trial": int(study.best_trial.number),
                    "n_trials": int(len(study.trials)),
                }
                agg_meta.setdefault("tickers", {})[ticker_name] = converted

            agg_meta["generated_at"] = pd.Timestamp.utcnow().isoformat()
            E3HyperparameterOptimizer._safe_write_file(
                agg_meta_path,
                lambda f: yaml.dump(agg_meta, f, default_flow_style=False, sort_keys=False),
            )

            updated_tickers = [t for t in tickers if t not in per_ticker_tuned]
            print(f"\n✓ YAML por ticker sincronizado (best global): {agg_compact_path}")
            if updated_tickers:
                print(f"  Actualizado con best global: {', '.join(updated_tickers)}")
            if per_ticker_tuned:
                print(f"  ⏭  Preservados (estudio propio): {', '.join(sorted(per_ticker_tuned))}")
            print(f"✓ YAML meta sincronizado: {agg_meta_path}")
        elif tickers:
            print("\n⚠️  No hay trials completos; no se sincronizó YAML por ticker.")

        # Si solo hay un ticker y no se usó --per_ticker, guardar también en by_ticker
        if tickers and len(tickers) == 1 and has_completed_trials:
            t = tickers[0]
            per_ticker_dir = output_dir / "by_ticker" / t
            per_ticker_dir.mkdir(parents=True, exist_ok=True)

            per_ticker_path = per_ticker_dir / "best_params_e3.yaml"
            E3HyperparameterOptimizer._safe_write_file(
                per_ticker_path,
                lambda f: yaml.dump(converted, f, default_flow_style=False, sort_keys=False),
            )

            agg_compact_path = output_dir / "e3_tuned_params_by_ticker.yaml"
            try:
                with open(agg_compact_path, "r") as f:
                    agg_compact = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_compact = {}
            agg_compact[t] = converted
            E3HyperparameterOptimizer._safe_write_file(
                agg_compact_path,
                lambda f: yaml.dump(agg_compact, f, default_flow_style=False, sort_keys=False),
            )

            agg_meta_path = output_dir / "e3_tuned_params_by_ticker.meta.yaml"
            try:
                with open(agg_meta_path, "r") as f:
                    agg_meta = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_meta = {}
            if not agg_meta:
                agg_meta = {"strategy": "e3_intraday", "generated_at": pd.Timestamp.utcnow().isoformat(), "meta": {}, "tickers": {}}
            agg_meta.setdefault("meta", {})[t] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }
            agg_meta.setdefault("tickers", {})[t] = converted
            agg_meta["generated_at"] = pd.Timestamp.utcnow().isoformat()
            E3HyperparameterOptimizer._safe_write_file(
                agg_meta_path,
                lambda f: yaml.dump(agg_meta, f, default_flow_style=False, sort_keys=False),
            )

            print(f"\n✓ Parámetros por ticker guardados en: {per_ticker_path}")
            print(f"✓ YAML agregados actualizados: {agg_compact_path} y {agg_meta_path}")

        # Resumen final

        print(f"\n{'='*80}")
        print("MEJORES PARÁMETROS ENCONTRADOS")
        print(f"{'='*80}\n")

        if completed_trials:
            displayed = False
            for key in DISPLAY_PARAM_KEYS:
                if key in study.best_params:
                    print(f"  {key}: {study.best_params[key]}")
                    displayed = True
            if not displayed:
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

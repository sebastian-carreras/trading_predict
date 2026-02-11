#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Estrategia E2 usando Optuna + MLflow

Este script busca automáticamente los mejores hiperparámetros para:
- Thresholds de trading (tau_buy, tau_sell)
- Arquitectura del modelo LSTM (lstm_units, dropout)
- Hiperparámetros de entrenamiento (learning_rate, batch_size)
- Filtros de entrada (rsi14_min, rsi14_max)

Uso:
    # Optimización local (por defecto, sin Docker)
    python scripts/optimize_e2_hyperparameters.py --n_trials 10
    
    # Optimización rápida (3 tickers, 10 trials)
    python scripts/optimize_e2_hyperparameters.py --n_trials 10 --quick
    
    # Optimizar solo un ticker
    python scripts/optimize_e2_hyperparameters.py --ticker NVDA --n_trials 30
    
    # Usar servidor MLflow remoto (Docker)
    python scripts/optimize_e2_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
    
    # Continuar estudio existente
    python scripts/optimize_e2_hyperparameters.py --study_name e2_optimization --n_trials 20
    
    # Optimización por ticker (genera YAML con mejores params por ticker)
    python scripts/optimize_e2_hyperparameters.py --per_ticker --n_trials 30

Outputs:
- MLflow tracking: runs/mlflow_local/mlflow.db (SQLite local) o servidor remoto
- Optuna database: optuna_studies.db (persistencia)
- Mejores parámetros: reports/hyperparameter_optimization/best_params_e2.yaml
- Visualizaciones: reports/hyperparameter_optimization/figures/e2_*.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
import yaml

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train_e2_pipeline import run_e2_for_ticker, load_ohlcv_csv
from src.utils import load_yaml, project_root


class E2HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para estrategia E2 (LSTM).
    """
    
    def __init__(
        self,
        config_path: Path,
        tickers: Optional[List[str]] = None,
        mlflow_tracking_uri: str = "local",
        optuna_db_path: str = "sqlite:///optuna_studies.db",
        search_space_overrides: Optional[Dict[str, Dict[str, float]]] = None,
        batch_size_choices: Optional[List[int]] = None,
    ):
        """
        Args:
            config_path: Path al archivo de configuración base
            tickers: Lista de tickers a usar (None = usar todos de E2)
            mlflow_tracking_uri: URI de MLflow tracking server (use 'local' para almacenamiento local)
            optuna_db_path: Path a base de datos de Optuna
            search_space_overrides: Overrides opcionales para rangos de búsqueda
            batch_size_choices: Opciones de batch size a evaluar (default: [32, 64, 128])
        """
        self.config = load_yaml(config_path)
        self.root = project_root()
        
        # Tickers
        if tickers:
            self.tickers = tickers
        else:
            self.tickers = list(
                self.config.get("universe", {})
                .get("tickers_by_strategy", {})
                .get("e2_moderate", [])
            )
        
        # MLflow - Configurar tracking URI
        if mlflow_tracking_uri == "local":
            # Modo local: usar SQLite backend (recomendado por MLflow)
            mlflow_dir = self.root / "runs/mlflow_local"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            mlflow_db = mlflow_dir / "mlflow.db"
            tracking_uri = f"sqlite:///{mlflow_db}"
            print(f"✓ Usando MLflow local (SQLite): {mlflow_db}")
        else:
            # Modo remoto: usar servidor MLflow (ej: Docker)
            tracking_uri = mlflow_tracking_uri
            print(f"✓ Conectando a MLflow server: {tracking_uri}")
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Configurar experimento
        experiment_name = "E2_Hyperparameter_Optimization"
        
        # Forzar artifact location local para evitar conflictos con Docker
        artifact_location = str(self.root / "mlruns")
        
        try:
            # Intentar establecer experimento (crearlo si no existe)
            mlflow.set_experiment(experiment_name)
            print(f"✓ Usando experimento '{experiment_name}'")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  Error configurando experimento: {e}")
            
            # Si es un error de revisión de Alembic, recrear la base de datos
            if "Can't locate revision" in error_msg or "alembic" in error_msg.lower():
                if mlflow_tracking_uri == "local":
                    print(f"⚠️  Base de datos MLflow corrupta. Recreando...")
                    # Eliminar base de datos corrupta
                    if mlflow_db.exists():
                        mlflow_db.unlink()
                        print(f"✓ Base de datos antigua eliminada: {mlflow_db}")
                    
                    # Reconectar con base de datos nueva
                    mlflow.set_tracking_uri(tracking_uri)
                    print(f"✓ Nueva base de datos MLflow creada")
            
            # Crear experimento manualmente con artifact location local
            try:
                mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                mlflow.set_experiment(experiment_name)
                print(f"✓ Experimento '{experiment_name}' creado")
            except Exception as create_error:
                # Si el experimento ya existe, solo establecerlo
                mlflow.set_experiment(experiment_name)
                print(f"✓ Experimento '{experiment_name}' configurado")
        
        # Optuna
        self.optuna_db_path = optuna_db_path

        # Search space defaults (se pueden sobreescribir vía CLI / DAG)
        self.tau_buy_bounds = {"min": 0.015, "max": 0.05, "step": 0.005}
        self.tau_sell_bounds = {"min": -0.01, "max": 0.01, "step": 0.005}
        self.lstm_units_1_bounds = {"min": 64, "max": 256, "step": 32}
        self.lstm_units_2_bounds = {"min": 32, "max": 128, "step": 16}
        self.dropout_bounds = {"min": 0.1, "max": 0.4, "step": 0.05}
        self.learning_rate_bounds = {"min": 5e-5, "max": 5e-3, "log": True}
        self.rsi14_min_bounds = {"min": 25, "max": 40, "step": 5}
        self.rsi14_max_bounds = {"min": 60, "max": 75, "step": 5}

        # Opciones de batch size
        default_batch_sizes = batch_size_choices or [32, 64, 128]
        cleaned_batch_sizes = sorted({int(b) for b in default_batch_sizes if int(b) > 0})
        if not cleaned_batch_sizes:
            raise ValueError("Se requiere al menos un batch_size válido para la optimización")
        
        # Walk-forward parameters (opcional - afecta validación)
        self.n_folds_bounds = {"min": 3, "max": 7}  # Balance robustez vs costo computacional
        self.internal_val_fraction_bounds = {"min": 0.10, "max": 0.25, "step": 0.05}  # Fracción validación interna
        self.batch_size_choices = cleaned_batch_sizes

        # Aplicar overrides si se proporcionaron
        overrides = search_space_overrides or {}

        def update_bounds(bounds: Dict[str, float], key: str) -> None:
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
        update_bounds(self.lstm_units_1_bounds, "lstm_units_1")
        update_bounds(self.lstm_units_2_bounds, "lstm_units_2")
        update_bounds(self.dropout_bounds, "dropout")
        update_bounds(self.learning_rate_bounds, "learning_rate")
        update_bounds(self.rsi14_min_bounds, "rsi14_min")
        update_bounds(self.rsi14_max_bounds, "rsi14_max")

        def validate_bounds(bounds: Dict[str, float], name: str) -> None:
            if bounds["min"] >= bounds["max"]:
                raise ValueError(
                    f"Rango inválido para {name}: min ({bounds['min']}) debe ser < max ({bounds['max']})"
                )
            if "step" in bounds and bounds.get("step", 0) <= 0:
                raise ValueError(f"El step debe ser > 0 para {name}")

        validate_bounds(self.tau_buy_bounds, "tau_buy")
        validate_bounds(self.tau_sell_bounds, "tau_sell")
        validate_bounds(self.lstm_units_1_bounds, "lstm_units_1")
        validate_bounds(self.lstm_units_2_bounds, "lstm_units_2")
        validate_bounds(self.dropout_bounds, "dropout")
        validate_bounds(self.learning_rate_bounds, "learning_rate")
        validate_bounds(self.rsi14_min_bounds, "rsi14_min")
        validate_bounds(self.rsi14_max_bounds, "rsi14_max")
        
        # Benchmark deshabilitado
        self.benchmark_df = None
        
        print(f"✓ Inicializado optimizador para {len(self.tickers)} tickers")
        print(f"  Tickers: {', '.join(self.tickers[:5])}{'...' if len(self.tickers) > 5 else ''}")
        print(f"  MLflow: {mlflow_tracking_uri}")
        print(f"  Optuna DB: {optuna_db_path}")
    
    @staticmethod
    def _suggest_float(trial: optuna.Trial, name: str, bounds: Dict[str, float]) -> float:
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
            Métrica objetivo (combinación de Sharpe ratio y profit factor)
        """
        
        # Iniciar run de MLflow para este trial
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            
            # 1. SUGERIR HIPERPARÁMETROS
            
            # Trading thresholds (críticos para backtesting)
            tau_buy = self._suggest_float(trial, "tau_buy", self.tau_buy_bounds)
            tau_sell = self._suggest_float(trial, "tau_sell", self.tau_sell_bounds)
            
            # Arquitectura LSTM
            lstm_units_1 = self._suggest_int(trial, "lstm_units_1", self.lstm_units_1_bounds)
            lstm_units_2 = self._suggest_int(trial, "lstm_units_2", self.lstm_units_2_bounds)
            
            # Regularización
            dropout = self._suggest_float(trial, "dropout", self.dropout_bounds)
            
            # Entrenamiento
            learning_rate = self._suggest_float(trial, "learning_rate", self.learning_rate_bounds)
            batch_size = trial.suggest_categorical("batch_size", self.batch_size_choices)
            
            # Filtros RSI
            rsi14_min = self._suggest_int(trial, "rsi14_min", self.rsi14_min_bounds)
            rsi14_max = self._suggest_int(trial, "rsi14_max", self.rsi14_max_bounds)
            
            # Validar que rsi14_min < rsi14_max
            if rsi14_min >= rsi14_max:
                # Penalizar configuración inválida
                mlflow.log_metric("objective_value", -100.0)
                return -100.0
            
            # Walk-forward parameters
            n_folds = self._suggest_int(trial, "n_folds", self.n_folds_bounds)
            internal_val_fraction = self._suggest_float(trial, "internal_val_fraction", self.internal_val_fraction_bounds)
            
            # Early stopping (fijar patience para consistencia)
            early_stopping_patience = 12
            
            # 2. ACTUALIZAR CONFIG CON PARÁMETROS SUGERIDOS
            
            config_trial = self.config.copy()
            
            # Walk-forward splits (sobrescribir configuración base)
            config_trial.setdefault("splits", {})["folds"] = n_folds
            config_trial["splits"]["internal_val_fraction"] = internal_val_fraction
            
            # Thresholds
            config_trial["strategies"]["e2_moderate"]["thresholds"] = {
                "tau_buy": tau_buy,
                "tau_sell": tau_sell,
            }
            
            # Filtros
            config_trial["strategies"]["e2_moderate"]["filters"] = {
                **config_trial["strategies"]["e2_moderate"].get("filters", {}),
                "rsi14_min": rsi14_min,
                "rsi14_max": rsi14_max,
            }
            
            # Modelo
            config_trial["strategies"]["e2_moderate"]["model"] = {
                **config_trial["strategies"]["e2_moderate"].get("model", {}),
                "lstm_units": [lstm_units_1, lstm_units_2],
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "early_stopping_patience": early_stopping_patience,
            }
            
            # 3. ENTRENAR PARA TODOS LOS TICKERS
            
            results = []
            raw_dir = self.root / "data/raw/daily"
            
            # Output temporal para este trial
            out_dir = self.root / "runs/optuna_trials" / f"e2_trial_{trial.number}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for ticker in self.tickers:
                try:
                    result = run_e2_for_ticker(
                        config=config_trial,
                        ticker=ticker,
                        raw_dir=raw_dir,
                        out_dir=out_dir,
                        benchmark_df=self.benchmark_df,
                    )
                    
                    results.append({
                        "ticker": ticker,
                        "ic": result.get("ml_ic", 0),
                        "sharpe": result.get("bt_sharpe", 0),
                        "profit_factor": result.get("bt_profit_factor", 0),
                        "num_trades": result.get("bt_num_trades", 0),
                        "win_rate": result.get("bt_win_rate", 0),
                        "max_dd": result.get("bt_max_drawdown", 0),
                        "cagr": result.get("bt_cagr", 0),
                    })
                    
                except Exception as e:
                    print(f"  ⚠️  Error en {ticker}: {e}")
                    # Penalizar trials que fallan
                    results.append({
                        "ticker": ticker,
                        "ic": -1.0,
                        "sharpe": -10.0,
                        "profit_factor": 0.0,
                        "num_trades": 0,
                        "win_rate": 0,
                        "max_dd": 1.0,
                        "cagr": -1.0,
                    })
            
            # 4. CALCULAR MÉTRICAS AGREGADAS
            
            df_results = pd.DataFrame(results)
            
            # Métricas promedio
            ic_mean = df_results["ic"].mean()
            ic_median = df_results["ic"].median()
            sharpe_mean = df_results["sharpe"].mean()
            sharpe_median = df_results["sharpe"].median()
            profit_factor_mean = df_results["profit_factor"].mean()
            cagr_mean = df_results["cagr"].mean()
            
            # Métricas de consistencia
            ic_std = df_results["ic"].std()
            sharpe_std = df_results["sharpe"].std()
            
            # Tickers con performance positiva
            ic_positive_pct = (df_results["ic"] > 0.05).sum() / len(df_results) * 100
            sharpe_positive_pct = (df_results["sharpe"] > 0).sum() / len(df_results) * 100
            profit_factor_above_1_pct = (df_results["profit_factor"] > 1.0).sum() / len(df_results) * 100
            
            # Tickers con trades (evitar thresholds muy altos que no generan señales)
            tickers_with_trades = (df_results["num_trades"] > 0).sum()
            avg_num_trades = df_results["num_trades"].mean()
            
            # 5. REGISTRAR EN MLFLOW (dentro del trial)
            
            # Registrar parámetros
            mlflow.log_params({
                "tau_buy": tau_buy,
                "tau_sell": tau_sell,
                "lstm_units_1": lstm_units_1,
                "lstm_units_2": lstm_units_2,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "rsi14_min": rsi14_min,
                "rsi14_max": rsi14_max,
                "n_folds": n_folds,
                "internal_val_fraction": internal_val_fraction,
                "trial_number": trial.number,
            })
            
            # Registrar métricas
            mlflow.log_metrics({
                "ic_mean": ic_mean,
                "ic_median": ic_median,
                "ic_std": ic_std,
                "sharpe_mean": sharpe_mean,
                "sharpe_median": sharpe_median,
                "sharpe_std": sharpe_std,
                "profit_factor_mean": profit_factor_mean,
                "cagr_mean": cagr_mean,
                "ic_positive_pct": ic_positive_pct,
                "sharpe_positive_pct": sharpe_positive_pct,
                "profit_factor_above_1_pct": profit_factor_above_1_pct,
                "tickers_with_trades": tickers_with_trades,
                "avg_num_trades": avg_num_trades,
            })
            
            # Guardar tabla de resultados como artifact
            results_path = out_dir / "trial_results.csv"
            df_results.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            
            # 6. DEFINIR MÉTRICA OBJETIVO
            
            # Para E2 (moderada): balance entre Sharpe, Profit Factor y CAGR
            # Penalizar si pocos tickers generan trades
            trade_penalty = 0 if tickers_with_trades >= len(self.tickers) * 0.5 else -5
            
            # Objetivo combinado: 40% Sharpe + 30% Profit Factor + 30% CAGR
            objective_value = (
                0.4 * sharpe_mean + 
                0.3 * min(profit_factor_mean, 3.0) +  # Cap profit factor en 3 para evitar outliers
                0.3 * cagr_mean * 10 +  # Escalar CAGR (típicamente 0.1-0.3)
                trade_penalty
            )
            
            # Registrar métrica objetivo
            mlflow.log_metric("objective_value", objective_value)
            
            print(f"\nTrial {trial.number}:")
            print(f"  tau_buy={tau_buy:.3f}, tau_sell={tau_sell:.3f}")
            print(f"  lstm_units=[{lstm_units_1}, {lstm_units_2}], dropout={dropout:.2f}")
            print(f"  rsi14=[{rsi14_min}, {rsi14_max}], lr={learning_rate:.6f}")
            print(f"  IC mean={ic_mean:.4f}, Sharpe mean={sharpe_mean:.4f}")
            print(f"  PF mean={profit_factor_mean:.4f}, CAGR mean={cagr_mean:.4f}")
            print(f"  Tickers with trades: {tickers_with_trades}/{len(self.tickers)}")
            print(f"  → Objective: {objective_value:.4f}")
            
            return objective_value
    
    def optimize(
        self,
        n_trials: int = 50,
        study_name: str = "e2_hyperparameter_optimization",
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
        print(f"OPTIMIZACIÓN DE HIPERPARÁMETROS E2")
        print(f"{'='*80}")
        print(f"Trials: {n_trials}")
        print(f"Tickers: {len(self.tickers)}")
        print(f"Study: {study_name}")
        print(f"{'='*80}\n")
        
        # Crear o cargar estudio
        study = optuna.create_study(
            study_name=study_name,
            storage=self.optuna_db_path,
            direction="maximize",  # Maximizar objetivo combinado
            load_if_exists=True,  # Continuar si existe
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducibilidad
        )
        
        # Ejecutar optimización
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        
        return study
    
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
            "optimization": {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "best_value": float(best_value),
                "best_trial": study.best_trial.number,
            },
            "best_params": {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in best_params.items()},
        }
        
        yaml_path = output_dir / "best_params_e2.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(best_params_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Mejores parámetros guardados: {yaml_path}")
        
        # 2. Todos los trials en CSV
        
        df_trials = study.trials_dataframe()
        csv_path = output_dir / "e2_all_trials.csv"
        df_trials.to_csv(csv_path, index=False)
        
        print(f"✓ Todos los trials guardados: {csv_path}")
        
        # 3. Visualizaciones
        
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        def save_figure(fig, name: str, description: str):
            """Guarda figura en PNG (o HTML si kaleido falla)."""
            png_path = figures_dir / f"{name}.png"
            html_path = figures_dir / f"{name}.html"
            
            try:
                # Intentar PNG primero
                fig.write_image(str(png_path), width=1200, height=600)
                print(f"✓ Gráfico PNG: {name}.png")
            except Exception as e_png:
                try:
                    # Fallback a HTML interactivo
                    fig.write_html(str(html_path))
                    print(f"✓ Gráfico HTML: {name}.html (PNG falló: kaleido no disponible)")
                except Exception as e_html:
                    print(f"⚠️  Error generando {description}: {e_png}")
        
        # Optimization history
        try:
            fig = plot_optimization_history(study)
            save_figure(fig, "e2_optimization_history", "optimization_history")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history: {e}")
        
        # Parameter importances
        try:
            fig = plot_param_importances(study)
            save_figure(fig, "e2_param_importances", "param_importances")
        except Exception as e:
            print(f"⚠️  Error generando param_importances: {e}")
        
        # Parallel coordinate
        try:
            fig = plot_parallel_coordinate(study)
            # Parallel coordinate necesita más ancho
            png_path = figures_dir / "e2_parallel_coordinate.png"
            html_path = figures_dir / "e2_parallel_coordinate.html"
            
            try:
                fig.write_image(str(png_path), width=1400, height=800)
                print(f"✓ Gráfico PNG: e2_parallel_coordinate.png")
            except Exception:
                fig.write_html(str(html_path))
                print(f"✓ Gráfico HTML: e2_parallel_coordinate.html (PNG falló: kaleido no disponible)")
        except Exception as e:
            print(f"⚠️  Error generando parallel_coordinate: {e}")
        
        # 4. Resumen en texto
        
        summary_path = output_dir / "e2_optimization_summary.txt"
        with open(summary_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("OPTIMIZACIÓN DE HIPERPARÁMETROS E2 - RESUMEN\n")
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
        
        print(f"✓ Resumen guardado: {summary_path}")
        
        print(f"\n{'='*80}")
        print("RESULTADOS GUARDADOS EXITOSAMENTE")
        print(f"{'='*80}\n")


def optimize_per_ticker(
    config_path: Path,
    tickers: List[str],
    n_trials_per_ticker: int,
    mlflow_uri: str,
    optuna_db: str,
    output_dir: Path,
) -> Dict[str, Dict]:
    """
    Ejecuta optimización independiente para cada ticker.
    
    Returns:
        Dict con mejores parámetros por ticker
    """
    
    print(f"\n{'='*80}")
    print("OPTIMIZACIÓN POR TICKER (E2)")
    print(f"{'='*80}")
    print(f"Tickers: {len(tickers)}")
    print(f"Trials por ticker: {n_trials_per_ticker}")
    print(f"{'='*80}\n")
    
    results_by_ticker = {}
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Optimizando {ticker}...")
        print("-" * 60)
        
        optimizer = E2HyperparameterOptimizer(
            config_path=config_path,
            tickers=[ticker],  # Un solo ticker
            mlflow_tracking_uri=mlflow_uri,
            optuna_db_path=optuna_db,
        )
        
        study = optimizer.optimize(
            n_trials=n_trials_per_ticker,
            study_name=f"e2_optimization_{ticker}",
        )
        
        # Guardar mejores parámetros (incluir TODOS los params de Optuna)
        best_params = study.best_params.copy()
        
        # Asegurar que n_folds e internal_val_fraction estén presentes
        if "n_folds" not in best_params:
            best_params["n_folds"] = 5  # Fallback al default
        if "internal_val_fraction" not in best_params:
            best_params["internal_val_fraction"] = 0.15  # Fallback al default
        
        results_by_ticker[ticker] = best_params
        
        print(f"✓ {ticker} completado - Best value: {study.best_value:.4f}")
    
    # Guardar YAML con overrides por ticker
    tuned_params_path = output_dir / "e2_tuned_params_by_ticker.yaml"
    with open(tuned_params_path, "w") as f:
        yaml.dump(results_by_ticker, f, default_flow_style=False, sort_keys=True)
    
    print(f"\n✓ Parámetros optimizados por ticker guardados: {tuned_params_path}")
    
    return results_by_ticker


def main():
    parser = argparse.ArgumentParser(
        description="Optimización de hiperparámetros E2 con Optuna + MLflow"
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
        default="e2_hyperparameter_optimization",
        help="Nombre del estudio Optuna (default: e2_hyperparameter_optimization)",
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
        help="Lista de tickers separados por coma (overridea universo E2)",
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
        help="MLflow tracking URI (default: local, use http://localhost:5050 for Docker)",
    )
    parser.add_argument(
        "--optuna_db",
        type=str,
        default="sqlite:///optuna_studies.db",
        help="Path a base de datos de Optuna (default: sqlite:///optuna_studies.db)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reports/hyperparameter_optimization",
        help="Directorio de salida (default: reports/hyperparameter_optimization)",
    )
    
    # ============================================================================
    # Overrides del espacio de búsqueda (opcionales)
    # ============================================================================
    
    # Thresholds de trading
    parser.add_argument(
        "--tau_buy_min",
        type=float,
        default=None,
        help="Mínimo para tau_buy (default: 0.015)",
    )
    parser.add_argument(
        "--tau_buy_max",
        type=float,
        default=None,
        help="Máximo para tau_buy (default: 0.05)",
    )
    parser.add_argument(
        "--tau_sell_min",
        type=float,
        default=None,
        help="Mínimo para tau_sell (default: -0.01)",
    )
    parser.add_argument(
        "--tau_sell_max",
        type=float,
        default=None,
        help="Máximo para tau_sell (default: 0.01)",
    )
    
    # Arquitectura LSTM
    parser.add_argument(
        "--lstm_units_1_min",
        type=int,
        default=None,
        help="Mínimo para la primera capa LSTM (default: 64)",
    )
    parser.add_argument(
        "--lstm_units_1_max",
        type=int,
        default=None,
        help="Máximo para la primera capa LSTM (default: 256)",
    )
    parser.add_argument(
        "--lstm_units_2_min",
        type=int,
        default=None,
        help="Mínimo para la segunda capa LSTM (default: 32)",
    )
    parser.add_argument(
        "--lstm_units_2_max",
        type=int,
        default=None,
        help="Máximo para la segunda capa LSTM (default: 128)",
    )
    
    # Regularización
    parser.add_argument(
        "--dropout_min",
        type=float,
        default=None,
        help="Mínimo para dropout (default: 0.1)",
    )
    parser.add_argument(
        "--dropout_max",
        type=float,
        default=None,
        help="Máximo para dropout (default: 0.4)",
    )
    
    # Entrenamiento
    parser.add_argument(
        "--learning_rate_min",
        type=float,
        default=None,
        help="Mínimo para learning_rate (default: 5e-5)",
    )
    parser.add_argument(
        "--learning_rate_max",
        type=float,
        default=None,
        help="Máximo para learning_rate (default: 5e-3)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default=None,
        help="Lista de batch sizes a probar (ej: '32,64,128')",
    )
    
    # Filtros RSI (específico E2)
    parser.add_argument(
        "--rsi14_min_min",
        type=int,
        default=None,
        help="Mínimo para rsi14_min (default: 25)",
    )
    parser.add_argument(
        "--rsi14_min_max",
        type=int,
        default=None,
        help="Máximo para rsi14_min (default: 40)",
    )
    parser.add_argument(
        "--rsi14_max_min",
        type=int,
        default=None,
        help="Mínimo para rsi14_max (default: 60)",
    )
    parser.add_argument(
        "--rsi14_max_max",
        type=int,
        default=None,
        help="Máximo para rsi14_max (default: 75)",
    )
    
    # Walk-forward validation
    parser.add_argument(
        "--n_folds_min",
        type=int,
        default=None,
        help="Mínimo número de folds (default: 3)",
    )
    parser.add_argument(
        "--n_folds_max",
        type=int,
        default=None,
        help="Máximo número de folds (default: 7)",
    )
    parser.add_argument(
        "--internal_val_fraction_min",
        type=float,
        default=None,
        help="Mínimo fracción validación interna (default: 0.10)",
    )
    parser.add_argument(
        "--internal_val_fraction_max",
        type=float,
        default=None,
        help="Máximo fracción validación interna (default: 0.25)",
    )
    
    args = parser.parse_args()
    
    # Configurar paths
    root = project_root()
    config_path = root / args.config
    output_dir = root / args.output_dir
    
    # ============================================================================
    # Construir overrides del espacio de búsqueda desde CLI
    # ============================================================================
    
    search_overrides: Dict[str, Dict[str, float]] = {}
    
    # Trading thresholds
    if args.tau_buy_min is not None or args.tau_buy_max is not None:
        search_overrides["tau_buy"] = {}
        if args.tau_buy_min is not None:
            search_overrides["tau_buy"]["min"] = args.tau_buy_min
        if args.tau_buy_max is not None:
            search_overrides["tau_buy"]["max"] = args.tau_buy_max
    
    if args.tau_sell_min is not None or args.tau_sell_max is not None:
        search_overrides["tau_sell"] = {}
        if args.tau_sell_min is not None:
            search_overrides["tau_sell"]["min"] = args.tau_sell_min
        if args.tau_sell_max is not None:
            search_overrides["tau_sell"]["max"] = args.tau_sell_max
    
    # Arquitectura LSTM
    if args.lstm_units_1_min is not None or args.lstm_units_1_max is not None:
        search_overrides["lstm_units_1"] = {}
        if args.lstm_units_1_min is not None:
            search_overrides["lstm_units_1"]["min"] = args.lstm_units_1_min
        if args.lstm_units_1_max is not None:
            search_overrides["lstm_units_1"]["max"] = args.lstm_units_1_max
    
    if args.lstm_units_2_min is not None or args.lstm_units_2_max is not None:
        search_overrides["lstm_units_2"] = {}
        if args.lstm_units_2_min is not None:
            search_overrides["lstm_units_2"]["min"] = args.lstm_units_2_min
        if args.lstm_units_2_max is not None:
            search_overrides["lstm_units_2"]["max"] = args.lstm_units_2_max
    
    # Regularización
    if args.dropout_min is not None or args.dropout_max is not None:
        search_overrides["dropout"] = {}
        if args.dropout_min is not None:
            search_overrides["dropout"]["min"] = args.dropout_min
        if args.dropout_max is not None:
            search_overrides["dropout"]["max"] = args.dropout_max
    
    # Entrenamiento
    if args.learning_rate_min is not None or args.learning_rate_max is not None:
        search_overrides["learning_rate"] = {}
        if args.learning_rate_min is not None:
            search_overrides["learning_rate"]["min"] = args.learning_rate_min
        if args.learning_rate_max is not None:
            search_overrides["learning_rate"]["max"] = args.learning_rate_max
    
    # Filtros RSI
    if args.rsi14_min_min is not None or args.rsi14_min_max is not None:
        search_overrides["rsi14_min"] = {}
        if args.rsi14_min_min is not None:
            search_overrides["rsi14_min"]["min"] = args.rsi14_min_min
        if args.rsi14_min_max is not None:
            search_overrides["rsi14_min"]["max"] = args.rsi14_min_max
    
    if args.rsi14_max_min is not None or args.rsi14_max_max is not None:
        search_overrides["rsi14_max"] = {}
        if args.rsi14_max_min is not None:
            search_overrides["rsi14_max"]["min"] = args.rsi14_max_min
        if args.rsi14_max_max is not None:
            search_overrides["rsi14_max"]["max"] = args.rsi14_max_max
    
    # Walk-forward validation
    if args.n_folds_min is not None or args.n_folds_max is not None:
        search_overrides["n_folds"] = {}
        if args.n_folds_min is not None:
            search_overrides["n_folds"]["min"] = args.n_folds_min
        if args.n_folds_max is not None:
            search_overrides["n_folds"]["max"] = args.n_folds_max
    
    if args.internal_val_fraction_min is not None or args.internal_val_fraction_max is not None:
        search_overrides["internal_val_fraction"] = {}
        if args.internal_val_fraction_min is not None:
            search_overrides["internal_val_fraction"]["min"] = args.internal_val_fraction_min
        if args.internal_val_fraction_max is not None:
            search_overrides["internal_val_fraction"]["max"] = args.internal_val_fraction_max
    
    # Batch sizes
    batch_sizes_list = None
    if args.batch_sizes:
        batch_sizes_list = [int(b.strip()) for b in args.batch_sizes.split(",")]
    
    # Determinar tickers
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        config = load_yaml(config_path)
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )
        
        if args.quick:
            # Modo rápido: solo 3 tickers aleatorios
            import random
            random.seed(42)
            tickers = random.sample(tickers, min(3, len(tickers)))
            print(f"Modo rápido: usando {len(tickers)} tickers aleatorios")
    
    # Ejecutar optimización
    if args.per_ticker:
        # Optimización por ticker (genera YAML con overrides)
        optimize_per_ticker(
            config_path=config_path,
            tickers=tickers,
            n_trials_per_ticker=args.n_trials,
            mlflow_uri=args.mlflow_uri,
            optuna_db=args.optuna_db,
            output_dir=output_dir,
        )
    else:
        # Optimización global (mejores parámetros para todos)
        optimizer = E2HyperparameterOptimizer(
            config_path=config_path,
            tickers=tickers,
            mlflow_tracking_uri=args.mlflow_uri,
            optuna_db_path=args.optuna_db,
            search_space_overrides=search_overrides or None,
            batch_size_choices=batch_sizes_list,
        )
        
        study = optimizer.optimize(
            n_trials=args.n_trials,
            study_name=args.study_name,
            timeout=args.timeout,
        )
        
        optimizer.save_results(study, output_dir)

        # Si se optimiza un único ticker sin --per_ticker, guardar también en by_ticker
        if len(tickers) == 1:
            ticker = tickers[0]
            by_ticker_dir = output_dir / "by_ticker" / ticker
            by_ticker_dir.mkdir(parents=True, exist_ok=True)

            per_ticker_path = by_ticker_dir / "best_params_e2.yaml"
            with open(per_ticker_path, "w") as f:
                yaml.dump(study.best_params, f, default_flow_style=False, sort_keys=False)

            # Actualizar YAML agregado por ticker
            agg_path = output_dir / "e2_tuned_params_by_ticker.yaml"
            try:
                with open(agg_path, "r") as f:
                    agg_data = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_data = {}

            agg_data[ticker] = study.best_params
            with open(agg_path, "w") as f:
                yaml.dump(agg_data, f, default_flow_style=False, sort_keys=False)

            print(f"✓ Parámetros por ticker guardados en: {per_ticker_path}")
            print(f"✓ YAML agregado actualizado: {agg_path}")
        
        # Mostrar resumen final
        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*80}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print("\nMejores parámetros:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResultados guardados en: {output_dir}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

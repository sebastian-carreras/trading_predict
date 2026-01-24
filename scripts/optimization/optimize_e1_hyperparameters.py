#!/usr/bin/env python3
"""
Optimización de Hiperparámetros para Estrategia E1 usando Optuna + MLflow

Este script busca automáticamente los mejores hiperparámetros para:
- Thresholds de trading (tau_buy, tau_sell)
- Arquitectura del modelo (gru_units, dropout)
- Hiperparámetros de entrenamiento (learning_rate, batch_size)

Uso:
    # Optimización local (por defecto, sin Docker)
    python scripts/optimize_e1_hyperparameters.py --n_trials 10
    
    # Optimización rápida (3 tickers, 10 trials)
    python scripts/optimize_e1_hyperparameters.py --n_trials 10 --quick
    
    # Optimizar solo un ticker
    python scripts/optimize_e1_hyperparameters.py --ticker AAPL --n_trials 30
    
    # Usar servidor MLflow remoto (Docker)
    python scripts/optimize_e1_hyperparameters.py --n_trials 50 --mlflow_uri http://localhost:5050
    
    # Continuar estudio existente
    python scripts/optimize_e1_hyperparameters.py --study_name e1_optimization --n_trials 20

Outputs:
- MLflow tracking: runs/mlflow_local/mlflow.db (SQLite local) o servidor remoto
- Optuna database: optuna_studies.db (persistencia)
- Mejores parámetros: reports/best_params_e1.yaml
- Visualizaciones: reports/figures/optuna_*.png
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

from src.train_e1_pipeline import run_e1_for_ticker, load_ohlcv_csv
from src.utils import load_yaml, project_root


class E1HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para estrategia E1.
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
            tickers: Lista de tickers a usar (None = usar todos de E1)
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
                .get("e1_conservative", [])
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
        experiment_name = "E1_Hyperparameter_Optimization"
        
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

        # Search space defaults (se pueden overridear vía CLI / DAG)
        self.tau_buy_bounds = {"min": 0.02, "max": 0.10, "step": 0.01}
        self.tau_sell_bounds = {"min": -0.02, "max": 0.02, "step": 0.01}
        self.gru_units_1_bounds = {"min": 32, "max": 128, "step": 16}
        self.gru_units_2_bounds = {"min": 16, "max": 64, "step": 16}
        self.dropout_bounds = {"min": 0.2, "max": 0.5, "step": 0.05}
        self.learning_rate_bounds = {"min": 1e-4, "max": 1e-2, "log": True}

        # Opciones de batch size
        default_batch_sizes = batch_size_choices or [32, 64, 128]
        cleaned_batch_sizes = sorted({int(b) for b in default_batch_sizes if int(b) > 0})
        if not cleaned_batch_sizes:
            raise ValueError("Se requiere al menos un batch_size válido para la optimización")
        self.batch_size_choices = cleaned_batch_sizes
        
        # Walk-forward parameters (opcional - afecta validación)
        self.n_folds_bounds = {"min": 3, "max": 7}  # Balance robustez vs costo computacional
        self.internal_val_fraction_bounds = {"min": 0.10, "max": 0.25, "step": 0.05}  # Fracción validación interna

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
        update_bounds(self.gru_units_1_bounds, "gru_units_1")
        update_bounds(self.gru_units_2_bounds, "gru_units_2")
        update_bounds(self.dropout_bounds, "dropout")
        update_bounds(self.learning_rate_bounds, "learning_rate")

        def validate_bounds(bounds: Dict[str, float], name: str) -> None:
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
        
        # Benchmark
        benchmark = self.config.get("universe", {}).get("benchmark", "SPY")
        raw_dir = self.root / "data/raw/daily"
        benchmark_path = raw_dir / f"{benchmark}_daily.csv"
        self.benchmark_df = load_ohlcv_csv(benchmark_path) if benchmark_path.exists() else None
        
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
            Métrica objetivo (Sharpe ratio promedio o IC promedio)
        """
        
        # Iniciar run de MLflow para este trial
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            
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
            batch_size = trial.suggest_categorical("batch_size", self.batch_size_choices)
                        # Walk-forward parameters
            n_folds = self._suggest_int(trial, "n_folds", self.n_folds_bounds)
            internal_val_fraction = self._suggest_float(trial, "internal_val_fraction", self.internal_val_fraction_bounds)
                        # Early stopping (fijar patience para consistencia)
            early_stopping_patience = 10
            
            # 2. ACTUALIZAR CONFIG CON PARÁMETROS SUGERIDOS
            
            config_trial = self.config.copy()
            
            # Walk-forward splits (sobrescribir configuración base)
            config_trial.setdefault("splits", {})["folds"] = n_folds
            config_trial["splits"]["internal_val_fraction"] = internal_val_fraction
            
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
                try:
                    result = run_e1_for_ticker(
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
                        "num_trades": result.get("bt_num_trades", 0),
                        "win_rate": result.get("bt_win_rate", 0),
                        "max_dd": result.get("bt_max_drawdown", 0),
                    })
                    
                except Exception as e:
                    print(f"  ⚠️  Error en {ticker}: {e}")
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
            
            # Registrar parámetros
            mlflow.log_params({
                "tau_buy": tau_buy,
                "tau_sell": tau_sell,
                "gru_units_1": gru_units_1,
                "gru_units_2": gru_units_2,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
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
                "ic_positive_pct": ic_positive_pct,
                "sharpe_positive_pct": sharpe_positive_pct,
                "tickers_with_trades": tickers_with_trades,
                "avg_num_trades": avg_num_trades,
            })
            
            # Guardar tabla de resultados como artifact
            results_path = out_dir / "trial_results.csv"
            df_results.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            
            # 6. DEFINIR MÉTRICA OBJETIVO
            
            # Opción 1: Sharpe ratio promedio (preferido para trading)
            # objective_value = sharpe_mean
            
            # Opción 2: IC promedio (preferido para predicción)
            # objective_value = ic_mean
            
            # Opción 3: Combinación (balance predicción + trading)
            # Penalizar si pocos tickers generan trades
            trade_penalty = 0 if tickers_with_trades >= len(self.tickers) * 0.5 else -5
            objective_value = 0.5 * ic_mean + 0.5 * sharpe_mean + trade_penalty
            
            # Registrar métrica objetivo
            mlflow.log_metric("objective_value", objective_value)
            
            print(f"\nTrial {trial.number}:")
            print(f"  tau_buy={tau_buy:.3f}, tau_sell={tau_sell:.3f}")
            print(f"  gru_units=[{gru_units_1}, {gru_units_2}], dropout={dropout:.2f}")
            print(f"  IC mean={ic_mean:.4f}, Sharpe mean={sharpe_mean:.4f}")
            print(f"  Tickers with trades: {tickers_with_trades}/{len(self.tickers)}")
            print(f"  → Objective: {objective_value:.4f}")
            
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
        
        # Crear o cargar estudio
        study = optuna.create_study(
            study_name=study_name,
            storage=self.optuna_db_path,
            direction="maximize",  # Maximizar Sharpe/IC
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
        
        yaml_path = output_dir / "best_params_e1.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(best_params_yaml, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Mejores parámetros guardados: {yaml_path}")
        
        # 2. Todos los trials en CSV
        
        df_trials = study.trials_dataframe()
        csv_path = output_dir / "e1_all_trials.csv"
        df_trials.to_csv(csv_path, index=False)
        
        print(f"✓ Todos los trials guardados: {csv_path}")
        
        # 3. Visualizaciones
        
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Optimization history
        try:
            fig = plot_optimization_history(study)
            fig.write_image(str(figures_dir / "optimization_history.png"), width=1200, height=600)
            print(f"✓ Gráfico: optimization_history.png")
        except Exception as e:
            print(f"⚠️  Error generando optimization_history: {e}")
        
        # Parameter importances
        try:
            fig = plot_param_importances(study)
            fig.write_image(str(figures_dir / "param_importances.png"), width=1200, height=600)
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
        with open(summary_path, "w") as f:
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
        
        print(f"✓ Resumen guardado: {summary_path}")
        
        print(f"\n{'='*80}")
        print("RESULTADOS GUARDADOS EXITOSAMENTE")
        print(f"{'='*80}\n")


def main():
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
        help="URI de MLflow tracking server (default: 'local' para tracking local, use 'http://localhost:5050' para servidor Docker)",
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
                search_space_overrides=search_overrides or None,
                batch_size_choices=batch_sizes_list,
            )

            study_name = f"{args.study_name}__{t}"
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
                    "batch_size": int(bp.get("batch_size")),
                    "early_stopping_patience": 10,
                },
                # Parámetros de walk-forward validation (opcional, con fallbacks)
                "n_folds": int(bp.get("n_folds", 5)),
                "internal_val_fraction": float(bp.get("internal_val_fraction", 0.15)),
            }
            meta_by_ticker[t] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }

        output_dir.mkdir(parents=True, exist_ok=True)
        tuned_path = output_dir / "e1_tuned_params_by_ticker.meta.yaml"
        with open(tuned_path, "w") as f:
            yaml.dump(
                {
                    "strategy": "e1_conservative",
                    "generated_at": pd.Timestamp.utcnow().isoformat(),
                    "meta": meta_by_ticker,
                    "tickers": overrides_by_ticker,
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        tuned_compact_path = output_dir / "e1_tuned_params_by_ticker.yaml"
        with open(tuned_compact_path, "w") as f:
            yaml.dump(
                overrides_by_ticker,
                f,
                default_flow_style=False,
                sort_keys=False,
            )

        print(f"\n✓ YAML de tuned params por ticker (con meta): {tuned_path}")
        print(f"✓ YAML consumible por training: {tuned_compact_path}")
        print("Para usarlo en training: export E1_TUNED_PARAMS_PATH=<ruta_al_yaml>")

    else:
        optimizer = E1HyperparameterOptimizer(
            config_path=config_path,
            tickers=tickers,
            mlflow_tracking_uri=args.mlflow_uri,
            search_space_overrides=search_overrides or None,
            batch_size_choices=batch_sizes_list,
        )

        study = optimizer.optimize(
            n_trials=args.n_trials,
            study_name=args.study_name,
            timeout=args.timeout,
        )

        optimizer.save_results(study, output_dir)

        # Si solo hay un ticker y no se usó --per_ticker, guardar también en by_ticker
        if tickers and len(tickers) == 1:
            t = tickers[0]
            per_ticker_dir = output_dir / "by_ticker" / t
            per_ticker_dir.mkdir(parents=True, exist_ok=True)

            bp = study.best_params
            converted = {
                "thresholds": {
                    "tau_buy": float(bp.get("tau_buy")),
                    "tau_sell": float(bp.get("tau_sell")),
                },
                "model": {
                    "gru_units": [int(bp.get("gru_units_1")), int(bp.get("gru_units_2"))],
                    "dropout": float(bp.get("dropout")),
                    "learning_rate": float(bp.get("learning_rate")),
                    "batch_size": int(bp.get("batch_size")),
                    "early_stopping_patience": 10,
                },
                "n_folds": int(bp.get("n_folds", 5)),
                "internal_val_fraction": float(bp.get("internal_val_fraction", 0.15)),
            }

            # best_params_e1.yaml por ticker
            per_ticker_path = per_ticker_dir / "best_params_e1.yaml"
            with open(per_ticker_path, "w") as f:
                yaml.dump(converted, f, default_flow_style=False, sort_keys=False)

            # Actualizar agregados compactos
            agg_compact_path = output_dir / "e1_tuned_params_by_ticker.yaml"
            try:
                with open(agg_compact_path, "r") as f:
                    agg_compact = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_compact = {}
            agg_compact[t] = converted
            with open(agg_compact_path, "w") as f:
                yaml.dump(agg_compact, f, default_flow_style=False, sort_keys=False)

            # Actualizar meta
            agg_meta_path = output_dir / "e1_tuned_params_by_ticker.meta.yaml"
            try:
                with open(agg_meta_path, "r") as f:
                    agg_meta = yaml.safe_load(f) or {}
            except FileNotFoundError:
                agg_meta = {}
            if not agg_meta:
                agg_meta = {"strategy": "e1_conservative", "generated_at": pd.Timestamp.utcnow().isoformat(), "meta": {}, "tickers": {}}
            agg_meta.setdefault("meta", {})[t] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_trial": int(study.best_trial.number),
                "n_trials": int(len(study.trials)),
            }
            agg_meta.setdefault("tickers", {})[t] = converted
            agg_meta["generated_at"] = pd.Timestamp.utcnow().isoformat()
            with open(agg_meta_path, "w") as f:
                yaml.dump(agg_meta, f, default_flow_style=False, sort_keys=False)

            print(f"\n✓ Parámetros por ticker guardados en: {per_ticker_path}")
            print(f"✓ YAML agregados actualizados: {agg_compact_path} y {agg_meta_path}")

        print(f"\n{'='*80}")
        print("MEJORES PARÁMETROS ENCONTRADOS")
        print(f"{'='*80}\n")

        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        print(f"\nMejor valor objetivo: {study.best_value:.6f}")
        print(f"Trial #{study.best_trial.number}")

        print(f"\n{'='*80}")
        print("OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*80}\n")

        print(f"📊 Resultados en: {output_dir}")
        print(f"📈 MLflow UI: {args.mlflow_uri}")
        print(f"🔍 Optuna Dashboard: optuna-dashboard sqlite:///optuna_studies.db")


if __name__ == "__main__":
    main()

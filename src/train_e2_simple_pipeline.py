"""
Pipeline simplificado E2 - Sin walk-forward, decision score simple.

Simplificaciones:
1. Decision score con solo 5 métricas: IC, directional_accuracy, sharpe, MAE, RMSE
2. Sin walk-forward validation (solo un split temporal train/val/test)
3. Arquitectura LSTM simplificada (1 capa en lugar de 2)

Flujo:
1. Descargar datos (opcional, con --skip-download)
2. Limpiar datos (opcional, con --skip-cleaning)
3. Calcular features E2 (momentum-focused: RSI, MACD, Stochastic, OBV)
4. Crear target y secuencias
5. Split temporal simple (train/val/test)
6. Entrenar LSTM simplificado
7. Evaluar y guardar
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .features.build_features_e2 import compute_e2_features, make_target_e2
from .features.build_sequences import make_sequences, time_split
from .models.e2_lstm import LSTMRegressor
from .backtest.backtest_daily import backtest_daily_signals, summarize_backtest
from .utils import ensure_dir, load_yaml, project_root


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y lo prepara.
    
    - Lee datos OHLCV desde un archivo CSV
    - Convierte la columna 'timestamp' a datetime con timezone UTC
    - Ordena datos cronológicamente y los indexa por timestamp
    - Valida que todas las columnas requeridas (OHLCV) estén presentes
    - Retorna DataFrame indexado por timestamp (timezone-aware)
    """
    # Cargar datos desde CSV
    df = pd.read_csv(path)
    
    # Validar que existe la columna timestamp
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    # Convertir timestamp a datetime con timezone UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
    
    # Ordenar por timestamp (cronológicamente ascendente)
    df = df.sort_values("timestamp")
    
    # Usar timestamp como índice (más eficiente para acceso temporal)
    df = df.set_index("timestamp")

    # Validar que todas las columnas OHLCV estén presentes
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Information Coefficient (correlación de Spearman).
    
    El IC mide qué tan bien las predicciones están correlacionadas con valores reales:
    - IC > 0.05 se considera significativo en finanzas
    - IC < 0 indica overfitting o falta de capacidad predictiva
    - IC ≈ 0 indica predicciones aleatorias
    """
    # Si hay menos de 2 muestras, no se puede calcular correlación
    if len(y_true) <= 1:
        return float("nan")

    # Si alguno de los valores es constante (sin variación), la correlación es indefinida
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    try:
        # Usar Spearman (correlación de rangos) que es más robusta a outliers
        from scipy.stats import spearmanr
        ic, _ = spearmanr(y_true, y_pred)
        # Asegurar que devolvemos un número finito
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        # Fallback a correlación de Pearson si scipy no está disponible
        return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_simple_decision_score(
    *,
    ic: float,
    directional_accuracy: float,
    sharpe: float,
    mae: float,
    rmse: float,
    targets: dict,
    weights: dict,
) -> tuple[float, dict]:
    """
    Calcula decision score simplificado con solo 5 métricas.
    
    Métricas (en escala normalizada 0-1.5):
    - IC: higher is better (target: ic_min)
    - Directional Accuracy: higher is better (target: directional_accuracy_min)
    - Sharpe: higher is better (target: sharpe_min)
    - MAE: lower is better (target: mae_max)
    - RMSE: lower is better (target: rmse_max)
    
    Returns:
        (decision_score, components_dict): Score ponderado y componentes individuales
    """
    
    def safe_ratio(value: float, target: float, higher_is_better: bool) -> float:
        """Calcula ratio normalizado [0, 1.5].
        
        - Si métrica debe ser alta (IC, Sharpe, Accuracy): ratio = value / target
        - Si métrica debe ser baja (MAE, RMSE): ratio = target / value
        - Limita resultado a rango [0, 1.5] para evitar dominancias
        """
        if not np.isfinite(value) or not np.isfinite(target) or target <= 0:
            return 0.0
        
        if higher_is_better:
            ratio = value / target
        else:
            if value <= 0:
                return 1.5
            ratio = target / value
        
        # Clipping a [0, 1.5] previene métricas outlier
        return float(np.clip(ratio, 0.0, 1.5))
    
    # Calcular componentes normalizados
    components = {
        'ic': safe_ratio(ic, targets.get('ic_min', 0.05), higher_is_better=True),
        'directional_accuracy': safe_ratio(
            directional_accuracy, 
            targets.get('directional_accuracy_min', 0.55), 
            higher_is_better=True
        ),
        'sharpe': safe_ratio(sharpe, targets.get('sharpe_min', 1.0), higher_is_better=True),
        'mae': safe_ratio(mae, targets.get('mae_max', 0.03), higher_is_better=False),
        'rmse': safe_ratio(rmse, targets.get('rmse_max', 0.05), higher_is_better=False),
    }
    
    # Normalizar pesos para que sumen a 1
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Score ponderado = suma de (componente * peso normalizado)
    score = sum(components[k] * normalized_weights.get(k, 0.0) for k in components)
    
    return float(score), components


def run_e2_simple_for_ticker(
    config: dict, 
    ticker: str, 
    raw_dir: Path, 
    out_dir: Path, 
    benchmark_df: pd.DataFrame | None
) -> dict:
    """Entrena y evalúa E2 simple para un ticker (sin walk-forward)."""
    
    config = copy.deepcopy(config)
    
    out_dir = Path(out_dir)
    if out_dir.name != ticker:
        out_dir = out_dir / ticker
    ensure_dir(out_dir)
    
    # Parámetros de la estrategia E2 simple
    e2_simple = config.get("strategies", {}).get("e2_simple", {})
    lookback_days = int(e2_simple.get("lookback_days", 60))
    horizon_days = int(e2_simple.get("horizon_days", 20))
    
    # Modelo (arquitectura simplificada)
    model_cfg = e2_simple.get("model", {})
    lstm_units = model_cfg.get("lstm_units", [128, 64])  # 2 capas LSTM por defecto
    dropout = float(model_cfg.get("dropout", 0.2))
    dense_units = int(model_cfg.get("dense_units", 16))
    lr = float(model_cfg.get("learning_rate", 0.001))
    batch_size = int(model_cfg.get("batch_size", 64))
    max_epochs = int(model_cfg.get("max_epochs", 100))
    patience = int(model_cfg.get("early_stopping_patience", 10))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))
    
    seed = int(config.get("project", {}).get("seed", 42))
    
    # Cargar datos
    clean_dir = raw_dir.parent.parent / "clean"
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"
    
    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")
        print(f"  ⚠️  Usando datos raw: {csv_path.name}")
    
    ohlcv = load_ohlcv_csv(csv_path)
    
    # Features y target
    features = compute_e2_features(ohlcv, benchmark_df)
    target = make_target_e2(ohlcv, horizon_days=horizon_days)
    
    # Secuencias
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)
    
    if len(X) == 0:
        raise ValueError(f"No hay suficientes datos para {ticker}")
    
    print(f"  Samples: {len(X)} | Features: {X.shape[-1]} | Lookback: {lookback_days}d")
    
    # Split temporal simple (70/15/15 por defecto)
    idx_train, idx_val, idx_test = time_split(len(X))
    
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = ts[idx_test]
    
    print(f"  Split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
    
    # Normalizar features
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    mean_X = Xtr2d.mean(axis=0)
    std_X = Xtr2d.std(axis=0) + 1e-12
    
    def scale_X(data: np.ndarray) -> np.ndarray:
        return ((data - mean_X) / std_X).astype(np.float32)
    
    # Normalizar target
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-12
    
    def scale_y(data: np.ndarray) -> np.ndarray:
        return ((data - mean_y) / std_y).astype(np.float32)
    
    def unscale_y(data: np.ndarray) -> np.ndarray:
        return (data * std_y + mean_y).astype(np.float32)
    
    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)
    
    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)
    
    # Entrenar modelo LSTM simplificado
    print(f"  Entrenando LSTM {lstm_units}...")
    model = LSTMRegressor(
        input_size=X_train.shape[-1],
        hidden_sizes=list(lstm_units),
        dropout=dropout,
        dense_units=dense_units,
        seed=seed,
    )
    
    res = model.fit(
        X_train_s,
        y_train_s,
        X_val_s,
        y_val_s,
        learning_rate=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        loss=loss,
        huber_delta=huber_delta,
    )
    
    # Predicciones
    y_pred_s = model.predict(X_test_s)
    y_pred = unscale_y(y_pred_s)
    
    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic = compute_information_coefficient(y_test, y_pred)
    
    # Sanitizar valores NaN para IC
    if not np.isfinite(ic):
        ic = 0.0
    
    # Backtest
    thresholds = e2_simple.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.025))
    tau_sell = float(thresholds.get("tau_sell", 0.00))
    
    costs_cfg = config.get("costs", {})
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))
    
    bt_cfg = e2_simple.get("backtest", {})
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    allow_short = bool(bt_cfg.get("allow_short", False))
    max_position = float(bt_cfg.get("max_position", 1.0))
    
    close_prices = ohlcv.loc[ts_test, "close"].to_numpy()
    bt = backtest_daily_signals(
        timestamps=ts_test,
        close_prices=close_prices,
        pred_returns=y_pred,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
        max_position=max_position,
    )
    trading_metrics = summarize_backtest(bt)
    
    sharpe = float(trading_metrics.get("sharpe", float("nan")))
    
    # Decision score simplificado
    decision_cfg = config.get("decision", {})
    targets = decision_cfg.get("targets", {})
    weights = decision_cfg.get("weights", {})
    threshold = float(decision_cfg.get("threshold", 0.70))
    
    # Valores por defecto para targets
    default_targets = {
        'ic_min': 0.05,
        'directional_accuracy_min': 0.55,
        'sharpe_min': 1.0,
        'mae_max': 0.03,
        'rmse_max': 0.05,
    }
    targets = {**default_targets, **targets}
    
    # Valores por defecto para weights
    default_weights = {
        'ic': 0.25,
        'directional_accuracy': 0.20,
        'sharpe': 0.30,
        'mae': 0.15,
        'rmse': 0.10,
    }
    weights = {**default_weights, **weights}
    
    decision_score, components = compute_simple_decision_score(
        ic=ic,
        directional_accuracy=dir_acc,
        sharpe=sharpe,
        mae=mae,
        rmse=rmse,
        targets=targets,
        weights=weights,
    )
    
    decision_signal = "buy" if decision_score >= threshold else "hold"
    
    # Print resultados
    ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"
    sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"
    
    print(f"  ✓ Epochs: {res.epochs_ran}/{max_epochs} | Val Loss: {res.best_val_loss:.6f}")
    print(f"  ✓ MAE={mae:.4f} RMSE={rmse:.4f} IC={ic_str} Dir={dir_acc:.1%} Sharpe={sharpe_str}")
    print(f"  ✓ Decision score {decision_score:.3f} (threshold {threshold:.2f}) → {decision_signal.upper()}")
    
    # Guardar resultados
    pred_df = pd.DataFrame(
        {'y_true': y_test, 'y_pred': y_pred},
        index=ts_test
    )
    pred_df.index.name = 'timestamp'
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")
    
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")
    
    scaler_df = pd.DataFrame({
        'feature': list(feat_names),
        'mean': mean_X,
        'std': std_X,
    })
    scaler_df.to_csv(out_dir / f"{ticker}_scaler.csv", index=False)
    
    # Guardar modelo
    model_payload = {
        "ticker": ticker,
        "strategy": "e2_simple",
        "created_at": datetime.utcnow().isoformat(),
        "model_class": "LSTMRegressor",
        "model_kwargs": {
            "input_size": int(X_train.shape[-1]),
            "hidden_sizes": list(lstm_units),
            "dropout": float(dropout),
            "dense_units": int(dense_units),
            "seed": int(seed),
        },
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "feature_names": list(feat_names),
        "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},
        "scaler_y": {"mean": float(mean_y), "std": float(std_y)},
        "train_result": {
            "epochs_ran": int(res.epochs_ran), 
            "best_val_loss": float(res.best_val_loss)
        },
        "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
    }
    
    model_path = out_dir / f"{ticker}_model.pth"
    try:
        model.torch.save(model_payload, model_path)
        print(f"  ✓ Modelo guardado: {model_path.name}")
    except Exception as exc:
        print(f"  ⚠️  No se pudo guardar modelo: {exc}")
    
    # Summary
    root = project_root()
    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    
    summary = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "split_method": "time_split",
        "epochs_ran": int(res.epochs_ran),
        "val_loss": float(res.best_val_loss),
        "ml_mae": mae,
        "ml_rmse": rmse,
        "ml_directional_accuracy": dir_acc,
        "ml_ic": ic,
        **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},
        "decision_score": decision_score,
        "decision_signal": decision_signal,
        "decision_threshold": float(threshold),
        # Target columns for reference
        "ic_target_min": float(targets['ic_min']),
        "directional_accuracy_target_min": float(targets['directional_accuracy_min']),
        "sharpe_target_min": float(targets['sharpe_min']),
        "mae_target_max": float(targets['mae_max']),
        "rmse_target_max": float(targets['rmse_max']),
        "predictions_file": as_relative(out_dir / f"{ticker}_predictions.csv"),
        "backtest_file": as_relative(out_dir / f"{ticker}_backtest.csv"),
        "scaler_file": as_relative(out_dir / f"{ticker}_scaler.csv"),
        "model_file": as_relative(model_path),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / f"{ticker}_summary.csv", index=False)
    
    return summary


def main():
    """Ejecuta el pipeline E2 simple para uno o más tickers."""
    parser = argparse.ArgumentParser(description="Pipeline E2 Simple (sin walk-forward, LSTM simplificado)")
    parser.add_argument(
        "--tickers",
        type=str,
        help="Tickers separados por coma (ej: NVDA,GOOGL,AMZN). Si se omite, usa los del config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Omite la descarga de datos (usa datos existentes)"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Omite la limpieza de datos (usa datos raw)"
    )
    args = parser.parse_args()
    
    root = project_root()
    config_path = root / args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config no encontrado: {config_path}")
    
    config = load_yaml(config_path)
    
    # Determinar tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = config.get("universe", {}).get("tickers_by_strategy", {}).get("e2_simple", [])
        if not tickers:
            raise ValueError("No se especificaron tickers ni en args ni en config para e2_simple")
    
    if not tickers:
        raise ValueError("No se especificaron tickers ni en args ni en config")
    
    # Benchmark
    benchmark_ticker = config.get("universe", {}).get("benchmark", "SPY")
    all_tickers = tickers + [benchmark_ticker] if benchmark_ticker not in tickers else tickers
    
    # Directorios
    raw_dir = root / "data" / "raw" / "daily"
    clean_dir = root / "data" / "clean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "runs" / "e2_simple" / timestamp
    ensure_dir(out_dir)
    
    # Guardar config usado
    config_used_path = out_dir / "config_used.yaml"
    import yaml
    with open(config_used_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"Pipeline E2 Simple - {len(tickers)} tickers")
    print(f"Output: {out_dir.relative_to(root)}")
    print(f"{'='*60}\n")
    
    # Paso 1: Descargar datos
    if not args.skip_download:
        print("Paso 1/3: Descargando datos...")
        print("-" * 60)
        from .data.download_daily import download_daily_ohlcv
        
        try:
            written = download_daily_ohlcv(
                all_tickers,
                out_dir=raw_dir,
                period="10y",
                skip_existing=True,
                min_days_fresh=1,
            )
            print(f"✓ Descargados/actualizados {len(written)} archivos\n")
        except Exception as exc:
            print(f"⚠️  Error en descarga: {exc}")
            print("Continuando con datos existentes...\n")
    else:
        print("Paso 1/3: Descarga omitida (usando datos existentes)\n")
    
    # Paso 2: Limpiar datos
    if not args.skip_cleaning:
        print("Paso 2/3: Limpiando datos...")
        print("-" * 60)
        from .data.clean_daily import process_daily_data_with_cleaning
        
        try:
            reports = process_daily_data_with_cleaning(
                raw_dir=raw_dir,
                clean_dir=clean_dir,
                strategy="forward_fill",
                min_days=252,
                remove_zero_volume=True,
                verbose=False,
            )
            cleaned = sum(1 for r in reports.values() if r.get("status") == "cleaned")
            rejected = sum(1 for r in reports.values() if r.get("status") == "rejected")
            print(f"✓ Limpiados: {cleaned} | Rechazados: {rejected}\n")
        except Exception as exc:
            print(f"⚠️  Error en limpieza: {exc}")
            print("Continuando con datos raw...\n")
    else:
        print("Paso 2/3: Limpieza omitida (usando datos raw)\n")
    
    # Paso 3: Entrenar modelos
    print("Paso 3/3: Entrenando modelos...")
    print("-" * 60)
    
    # Cargar benchmark
    bench_path = clean_dir / f"{benchmark_ticker}_daily.csv"
    if not bench_path.exists():
        bench_path = raw_dir / f"{benchmark_ticker}_daily.csv"
    
    if bench_path.exists():
        benchmark_df = load_ohlcv_csv(bench_path)
        print(f"Benchmark: {benchmark_ticker} ({len(benchmark_df)} días)\n")
    else:
        benchmark_df = None
        print(f"⚠️  Benchmark {benchmark_ticker} no encontrado\n")
    
    # Entrenar cada ticker
    summaries = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        try:
            summary = run_e2_simple_for_ticker(
                config=config,
                ticker=ticker,
                raw_dir=raw_dir,
                out_dir=out_dir,
                benchmark_df=benchmark_df,
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"  ❌ Error: {exc}\n")
            continue
        print()
    
    # Guardar summary agregado
    if summaries:
        summary_all = pd.DataFrame(summaries)
        summary_all.to_csv(out_dir / "summary_all.csv", index=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Completado: {len(summaries)}/{len(tickers)} tickers")
        print(f"  Resultados en: {out_dir.relative_to(root)}/")
        print(f"{'='*60}\n")
    else:
        print("\n❌ No se completó ningún ticker exitosamente\n")


if __name__ == "__main__":
    main()

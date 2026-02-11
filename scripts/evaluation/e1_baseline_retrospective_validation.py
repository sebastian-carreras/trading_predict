#!/usr/bin/env python3
"""
Validación Retrospectiva (Out-of-Time) para E1 Baseline con LinearRegression.

Simula entrenar modelo en el pasado y evaluar con datos de hoy.
Esto valida que el modelo funciona en condiciones de producción reales.
Usa LinearRegression simple (mismo que e1_baseline_linear.py) sin regularización.

Ejemplo:
    # Entrenar con datos hasta hace 360 días, evaluar con datos de hoy
    python scripts/e1_baseline_retrospective_validation.py \
        --ticker AAPL \
        --train-days-ago 360 \
        --horizon 90

Esto:
1. Descarga datos históricos
2. "Viaja al pasado" (limita datos a fecha antigua)
3. Entrena LinearRegression como si estuvieras en esa fecha
4. Evalúa con datos reales hasta hoy
5. Compara con métricas de backtesting original
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import mlflow

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_yaml
from src.features.build_features_e1 import compute_e1_features
from src.backtest.backtest_daily import backtest_daily_signals, summarize_backtest


def download_data_with_cutoff(ticker: str, cutoff_date: datetime, buffer_days: int = 1825) -> pd.DataFrame:
    """
    Descarga datos históricos hasta cutoff_date.
    
    Args:
        ticker: Símbolo
        cutoff_date: Fecha límite (solo usar datos hasta esta fecha)
        buffer_days: Días adicionales hacia atrás (default 5 años)
    
    Returns:
        DataFrame con OHLCV
    """
    import yfinance as yf
    
    start_date = cutoff_date - timedelta(days=buffer_days)
    end_date = cutoff_date + timedelta(days=5)  # Buffer pequeño
    
    print(f"📥 Descargando {ticker} desde {start_date.date()} hasta {cutoff_date.date()}...")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError(f"No hay datos para {ticker}")
    
    # Filtrar estrictamente hasta cutoff_date
    data = data[data.index <= cutoff_date]
    
    # Renombrar columnas a lowercase
    if isinstance(data.columns, pd.MultiIndex):
        # Si es MultiIndex, aplanar
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]
    
    print(f"✓ Descargados {len(data)} días (hasta {data.index[-1].date()})")
    
    return data


def train_retrospective_model(
    ticker: str,
    ohlcv: pd.DataFrame,
    config: dict,
    benchmark_df: pd.DataFrame = None
) -> tuple:
    """
    Entrena modelo E1 Simple con datos limitados (simulando pasado).
    
    Returns:
        (model, scaler_X, scaler_y, feature_names, train_end_date)
    """
    # Calcular features
    print(f"\n Calculando features...")
    features = compute_e1_features(ohlcv, benchmark_df)
    
    # Configuración del modelo
    lookback_days = config.get("lookback_days", 360)
    horizon_days = config.get("horizon_days", 90)
    
    # Crear targets (retorno forward)
    close = ohlcv["close"].values
    target = np.full(len(close), np.nan)
    
    for i in range(len(close) - horizon_days):
        ret = (close[i + horizon_days] - close[i]) / close[i]
        target[i] = ret
    
    # Alinear features con targets
    valid_mask = ~np.isnan(target)
    X = features[valid_mask].values
    y = target[valid_mask]
    timestamps = ohlcv.index[valid_mask]
    
    print(f"✓ Features calculadas: {X.shape[1]} features, {len(X)} samples")
    
    # Split temporal: 70% train, 15% val, 15% test
    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = timestamps[idx_test]
    
    train_end_date = timestamps[n_train - 1]
    
    print(f"✓ Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Train hasta: {train_end_date.date()}")
    
    # Normalizar
    mean_X = np.nanmean(X_train, axis=0)
    std_X = np.nanstd(X_train, axis=0) + 1e-8
    mean_y = np.nanmean(y_train)
    std_y = np.nanstd(y_train) + 1e-8
    
    X_train_s = (X_train - mean_X) / std_X
    y_train_s = (y_train - mean_y) / std_y
    X_val_s = (X_val - mean_X) / std_X
    y_val_s = (y_val - mean_y) / std_y
    X_test_s = (X_test - mean_X) / std_X
    
    # Manejar NaN
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0)
    
    # Entrenar modelo
    print(f"\n Entrenando modelo LinearRegression...")
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train_s, y_train_s)
    
    print(f"✓ Entrenamiento completado (R² train: {model.score(X_train_s, y_train_s):.4f})")
    
    # Predecir en test
    y_pred_s = model.predict(X_test_s)
    y_pred = y_pred_s * std_y + mean_y
    
    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    
    # IC (Spearman)
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_test, y_pred)
    
    ml_metrics = {
        'mae': mae,
        'rmse': rmse,
        'ic': float(ic),
        'directional_accuracy': dir_acc,
        'n_test': len(y_test)
    }
    
    print(f"\n📊 Métricas ML (período test):")
    print(f"  MAE: {mae:.4f}")
    print(f"  IC: {ic:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")
    
    # Guardar scalers
    scaler_X = {'mean': mean_X, 'std': std_X}
    scaler_y = {'mean': mean_y, 'std': std_y}
    feature_names = features.columns.tolist()
    
    return model, scaler_X, scaler_y, feature_names, train_end_date, ml_metrics, (X_test_s, y_test, y_pred, ts_test)


def evaluate_out_of_time(
    ticker: str,
    train_cutoff: datetime,
    eval_date: datetime,
    model,
    scaler_X: dict,
    scaler_y: dict,
    feature_names: list,
    config: dict,
    benchmark_df: pd.DataFrame = None
) -> dict:
    """
    Evalúa modelo con datos desde train_cutoff hasta eval_date.
    
    Esto simula evaluar el modelo en producción con datos nuevos.
    """
    import yfinance as yf
    
    # Descargar datos desde train_cutoff hasta hoy
    buffer_start = train_cutoff - timedelta(days=365)  # Buffer para features
    
    print(f"\n🔍 Evaluando out-of-time ({train_cutoff.date()} → {eval_date.date()})...")
    
    data = yf.download(ticker, start=buffer_start, end=eval_date + timedelta(days=5), progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]
    
    # Calcular features
    features = compute_e1_features(data, benchmark_df)
    
    # Crear targets
    lookback_days = config.get("lookback_days", 360)
    horizon_days = config.get("horizon_days", 90)
    
    close = data["close"].values
    target = np.full(len(close), np.nan)
    
    for i in range(len(close) - horizon_days):
        ret = (close[i + horizon_days] - close[i]) / close[i]
        target[i] = ret
    
    # Filtrar período de evaluación
    eval_start = train_cutoff
    eval_mask = (data.index >= eval_start) & (data.index <= eval_date) & (~np.isnan(target))
    
    features_eval = features[eval_mask][feature_names]
    target_eval = target[eval_mask]
    timestamps_eval = data.index[eval_mask]
    
    if len(features_eval) == 0:
        print(f"⚠️  No hay datos suficientes para evaluar")
        return {}
    
    print(f"✓ Período evaluación: {len(features_eval)} samples ({timestamps_eval[0].date()} → {timestamps_eval[-1].date()})")
    
    # Normalizar y predecir
    X_eval = features_eval.values
    X_eval_s = (X_eval - scaler_X['mean']) / scaler_X['std']
    
    y_pred_s = model.predict(X_eval_s)
    y_pred = y_pred_s * scaler_y['std'] + scaler_y['mean']
    
    # Métricas ML
    mae = float(np.mean(np.abs(target_eval - y_pred)))
    rmse = float(np.sqrt(np.mean((target_eval - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(target_eval) == np.sign(y_pred)))
    
    from scipy.stats import spearmanr
    ic, _ = spearmanr(target_eval, y_pred)
    
    print(f"\n📊 Métricas Out-of-Time:")
    print(f"  MAE: {mae:.4f}")
    print(f"  IC: {ic:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")
    
    # Backtest simulado
    close_eval = data.loc[timestamps_eval, "close"].values
    
    tau_buy = config.get("thresholds", {}).get("tau_buy", 0.05)
    tau_sell = config.get("thresholds", {}).get("tau_sell", 0.0)
    
    bt = backtest_daily_signals(
        timestamps=timestamps_eval,
        close_prices=close_eval,
        pred_returns=y_pred,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=config.get("costs", {}).get("round_trip_bps", 10),
        holding_period_days=horizon_days,
        allow_short=config.get("position", {}).get("allow_short", False),
        max_position=config.get("position", {}).get("max_position", 1.0)
    )
    
    trading_metrics = summarize_backtest(bt)
    
    print(f"\n💰 Backtesting Out-of-Time:")
    print(f"  Sharpe: {trading_metrics.get('sharpe', 0):.2f}")
    print(f"  CAGR: {trading_metrics.get('cagr', 0):.2%}")
    print(f"  Max DD: {trading_metrics.get('max_drawdown', 0):.2%}")
    print(f"  Trades: {trading_metrics.get('num_trades', 0):.0f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'ic': float(ic),
        'directional_accuracy': dir_acc,
        'n_samples': len(target_eval),
        **{f'bt_{k}': float(v) for k, v in trading_metrics.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Validación Retrospectiva E1 Simple")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker a evaluar")
    parser.add_argument("--train-days-ago", type=int, default=360, help="Días atrás para entrenar")
    parser.add_argument("--horizon", type=int, default=90, help="Horizonte de predicción")
    parser.add_argument("--config", type=str, default="src/config/base.yaml", help="Path a config YAML")
    parser.add_argument("--mlflow-docker", action="store_true", help="Usar MLflow en Docker")
    
    args = parser.parse_args()
    
    # Configurar fechas
    today = datetime.now()
    train_cutoff = today - timedelta(days=args.train_days_ago)
    
    print("=" * 80)
    print("E1 Simple - Validación Retrospectiva (Out-of-Time)")
    print("=" * 80)
    print(f"Ticker: {args.ticker}")
    print(f"Fecha entrenamiento: {train_cutoff.date()} (hace {args.train_days_ago} días)")
    print(f"Fecha evaluación: {today.date()} (hoy)")
    print(f"Horizonte: {args.horizon} días")
    print("=" * 80)
    print()
    
    # Cargar config
    config = load_yaml(Path(args.config))
    
    # Usar config de E1 conservative como base
    if 'strategies' in config and 'e1_conservative' in config['strategies']:
        model_config = config['strategies']['e1_conservative']
        model_config['horizon_days'] = args.horizon
    else:
        # Fallback: crear config mínima
        model_config = {
            'lookback_days': 360,
            'horizon_days': args.horizon,
            'thresholds': {'tau_buy': 0.05, 'tau_sell': 0.0},
            'costs': {'round_trip_bps': 10},
            'position': {'allow_short': False, 'max_position': 1.0}
        }
        config['strategies'] = {'e1_conservative': model_config}
    
    # 1. Descargar datos hasta train_cutoff
    ohlcv = download_data_with_cutoff(args.ticker, train_cutoff)
    
    # Benchmark deshabilitado
    benchmark_df = None
    
    # 2. Entrenar modelo (simulando estar en train_cutoff)
    model, scaler_X, scaler_y, feature_names, train_end, ml_metrics_test, test_data = train_retrospective_model(
        args.ticker, ohlcv, model_config, benchmark_df
    )
    
    # 3. Evaluar out-of-time (train_cutoff → hoy)
    oot_metrics = evaluate_out_of_time(
        args.ticker, train_cutoff, today, model, scaler_X, scaler_y, feature_names, model_config, benchmark_df
    )
    
    # 4. Comparar métricas
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN: In-Sample Test vs. Out-of-Time")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'In-Sample Test': [
            ml_metrics_test['mae'],
            ml_metrics_test['ic'],
            ml_metrics_test['directional_accuracy']
        ],
        'Out-of-Time': [
            oot_metrics.get('mae', np.nan),
            oot_metrics.get('ic', np.nan),
            oot_metrics.get('directional_accuracy', np.nan)
        ]
    }, index=['MAE', 'IC', 'Dir Acc'])
    
    comparison['Degradación %'] = ((comparison['Out-of-Time'] - comparison['In-Sample Test']) / 
                                     comparison['In-Sample Test'].abs() * 100)
    
    print(comparison.to_string())
    print()
    
    # Alertas
    if oot_metrics:
        alerts = []
        
        if abs(oot_metrics['ic'] - ml_metrics_test['ic']) / abs(ml_metrics_test['ic']) > 0.20:
            alerts.append(f"⚠️  IC degradó {comparison.loc['IC', 'Degradación %']:.1f}%")
        
        if abs(oot_metrics['mae'] - ml_metrics_test['mae']) / ml_metrics_test['mae'] > 0.20:
            alerts.append(f"⚠️  MAE empeoró {comparison.loc['MAE', 'Degradación %']:.1f}%")
        
        if alerts:
            print("🚨 ALERTAS:")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print("✅ Performance out-of-time es consistente con in-sample")
    
    print()
    
    # 5. Guardar en MLflow
    if oot_metrics:
        try:
            mlflow_uri = "http://localhost:5050" if args.mlflow_docker else f"sqlite:///{Path.cwd()}/runs/mlflow_local/mlflow.db"
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("E1_Simple_Retrospective_Validation")
            
            with mlflow.start_run(run_name=f"retrospective_{args.ticker}_{train_cutoff.strftime('%Y%m%d')}"):
                # Params
                mlflow.log_param("ticker", args.ticker)
                mlflow.log_param("train_cutoff_date", train_cutoff.strftime("%Y-%m-%d"))
                mlflow.log_param("eval_date", today.strftime("%Y-%m-%d"))
                mlflow.log_param("train_days_ago", args.train_days_ago)
                mlflow.log_param("horizon_days", args.horizon)
                
                # Métricas in-sample test
                mlflow.log_metric("test_mae", ml_metrics_test['mae'])
                mlflow.log_metric("test_ic", ml_metrics_test['ic'])
                mlflow.log_metric("test_dir_acc", ml_metrics_test['directional_accuracy'])
                
                # Métricas out-of-time
                for k, v in oot_metrics.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        mlflow.log_metric(f"oot_{k}", float(v))
                
                # Degradación
                mlflow.log_metric("ic_degradation_pct", comparison.loc['IC', 'Degradación %'])
                
                print(f"✓ Resultados guardados en MLflow")
                print(f"  Experimento: E1_Simple_Retrospective_Validation")
        
        except Exception as e:
            print(f"⚠️  No se pudo guardar en MLflow: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Validación retrospectiva completada")
    print("=" * 80)


if __name__ == "__main__":
    main()

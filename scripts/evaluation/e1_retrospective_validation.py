#!/usr/bin/env python3
"""
Validación Retrospectiva (Out-of-Time) para E1 Simple con GRU.

Simula entrenar modelo GRU en el pasado y evaluar con datos de hoy.
Usa datos locales de data/clean/ si existen.

Ejemplo:
    # Entrenar con datos hasta hace 180 días, evaluar con datos de hoy
    python scripts/e1_retrospective_validation.py \
        --ticker AAPL \
        --train-days-ago 180 \
        --horizon 90

Diferencias con baseline:
- Usa GRU (modelo real E1 Simple) en vez de LinearRegression
- Crea secuencias 3D para LSTM/GRU
- Verifica data/clean/ antes de descargar
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

from src.utils import load_yaml, project_root
from src.features.build_features_e1 import compute_e1_features
from src.features.build_sequences import make_sequences
from src.models.e1_gru import GRURegressor
from src.backtest.backtest_daily import backtest_daily_signals, summarize_backtest


def load_or_download_data(ticker: str, cutoff_date: datetime, data_dir: Path) -> pd.DataFrame:
    """
    Carga datos desde data/clean/ si existen, sino descarga desde yfinance.
    
    Args:
        ticker: Símbolo
        cutoff_date: Fecha límite
        data_dir: Directorio data/clean/
    
    Returns:
        DataFrame con OHLCV hasta cutoff_date
    """
    clean_file = data_dir / f"{ticker}_daily.csv"
    
    if clean_file.exists():
        print(f"📂 Cargando {ticker} desde archivo local: {clean_file}")
        df = pd.read_csv(clean_file)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)
        
        # Asegurar que cutoff_date tenga timezone si df.index lo tiene
        cutoff_tz = cutoff_date
        if df.index.tz is not None and cutoff_date.tzinfo is None:
            cutoff_tz = cutoff_date.replace(tzinfo=df.index.tz)
        elif df.index.tz is None and cutoff_date.tzinfo is not None:
            cutoff_tz = cutoff_date.replace(tzinfo=None)
        
        # Filtrar hasta cutoff_date
        df = df[df.index <= cutoff_tz]
        
        if len(df) > 0:
            print(f"✓ Cargados {len(df)} días desde archivo (hasta {df.index[-1].date()})")
            return df
        else:
            print(f"⚠️  Archivo no tiene datos hasta {cutoff_date.date()}, descargando...")
    
    # Si no existe archivo o no tiene datos suficientes, descargar
    import yfinance as yf
    
    buffer_days = 1825  # 5 años
    start_date = cutoff_date - timedelta(days=buffer_days)
    end_date = cutoff_date + timedelta(days=5)
    
    print(f"📥 Descargando {ticker} desde yfinance ({start_date.date()} → {cutoff_date.date()})...")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        raise ValueError(f"No hay datos para {ticker}")
    
    # Filtrar hasta cutoff_date
    data = data[data.index <= cutoff_date]
    
    # Renombrar columnas
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(c).lower() for c in data.columns]
    
    print(f"✓ Descargados {len(data)} días (hasta {data.index[-1].date()})")
    
    return data


def train_retrospective_model_gru(
    ticker: str,
    ohlcv: pd.DataFrame,
    config: dict,
    benchmark_df: pd.DataFrame = None
) -> tuple:
    """
    Entrena modelo GRU E1 Simple con datos limitados (simulando pasado).
    
    Returns:
        (model, scaler_X, scaler_y, feature_names, train_end_date, ml_metrics_test)
    """
    # Calcular features
    print(f"\n🔧 Calculando features...")
    features = compute_e1_features(ohlcv, benchmark_df)
    
    # Configuración
    lookback_days = config.get("lookback_days", 180)
    horizon_days = config.get("horizon_days", 90)
    
    # Crear targets (retorno forward)
    close = ohlcv["close"].values
    target = np.full(len(close), np.nan)
    
    for i in range(len(close) - horizon_days):
        ret = (close[i + horizon_days] - close[i]) / close[i]
        target[i] = ret
    
    # Alinear features con targets - crear Series con target
    target_series = pd.Series(target, index=ohlcv.index, name='target')
    
    print(f"✓ Features calculadas: {features.shape[1]} features, {len(features)} samples")
    
    # Crear secuencias 3D para GRU
    print(f"🔄 Creando secuencias (lookback={lookback_days})...")
    
    X_seq, y_seq, seq_timestamps, feature_names_seq = make_sequences(
        features=features,
        target=target_series,
        lookback=lookback_days
    )
    
    print(f"✓ Secuencias creadas: {X_seq.shape} (samples, lookback, features)")
    
    # Split temporal: 70% train, 15% val, 15% test
    n = len(X_seq)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    
    X_train, y_train = X_seq[idx_train], y_seq[idx_train]
    X_val, y_val = X_seq[idx_val], y_seq[idx_val]
    X_test, y_test = X_seq[idx_test], y_seq[idx_test]
    ts_test = seq_timestamps[idx_test]
    
    train_end_date = seq_timestamps[n_train - 1]
    
    print(f"✓ Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Train hasta: {train_end_date.date()}")
    
    # Normalizar: calcular stats en train, aplicar a todos
    # Shape: (samples, lookback, features) -> calcular mean/std sobre samples y lookback
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    
    mean_X = np.nanmean(X_train_flat, axis=0)
    std_X = np.nanstd(X_train_flat, axis=0) + 1e-8
    mean_y = np.nanmean(y_train)
    std_y = np.nanstd(y_train) + 1e-8
    
    # Aplicar normalización a secuencias
    X_train_s = (X_train - mean_X) / std_X
    y_train_s = (y_train - mean_y) / std_y
    X_val_s = (X_val - mean_X) / std_X
    y_val_s = (y_val - mean_y) / std_y
    X_test_s = (X_test - mean_X) / std_X
    
    # Manejar NaN
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_val_s = np.nan_to_num(X_val_s, nan=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0)
    
    # Entrenar GRU
    print(f"\n🚂 Entrenando modelo GRU...")
    
    model = GRURegressor(
        input_size=X_train.shape[-1],  # número de features
        hidden_sizes=[128, 64],  # 2 capas como en E1 Simple
        dropout=0.2,
        dense_units=32,
        device='cpu'
    )
    
    result = model.fit(
        X_train_s, y_train_s,
        X_val_s, y_val_s,
        max_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    print(f"✓ Entrenamiento completado: {result.epochs_ran} epochs, val_loss={result.best_val_loss:.6f}")
    
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
    print(f"  RMSE: {rmse:.4f}")
    print(f"  IC: {ic:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")
    
    # Guardar scalers y feature names
    scaler = {
        'mean_X': mean_X,
        'std_X': std_X,
        'mean_y': mean_y,
        'std_y': std_y
    }
    
    return model, scaler, feature_names_seq, train_end_date, ml_metrics


def evaluate_out_of_time_gru(
    ticker: str,
    train_cutoff: datetime,
    eval_date: datetime,
    model: GRURegressor,
    scaler: dict,
    feature_names: list,
    config: dict,
    data_dir: Path,
    benchmark_df: pd.DataFrame = None
) -> dict:
    """
    Evalúa modelo GRU con datos desde train_cutoff hasta eval_date.
    """
    print(f"\n🔍 Evaluando out-of-time ({train_cutoff.date()} → {eval_date.date()})...")
    
    # Cargar datos (verificar local primero)
    buffer_start = train_cutoff - timedelta(days=365)  # Buffer para features
    
    # Intentar cargar desde archivo local
    clean_file = data_dir / f"{ticker}_daily.csv"
    
    if clean_file.exists():
        print(f"📂 Cargando datos desde archivo local...")
        df = pd.read_csv(clean_file)
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)
        
        data = df[(df.index >= buffer_start) & (df.index <= eval_date)]
        
        if len(data) == 0:
            print(f"⚠️  Archivo no tiene datos recientes, descargando...")
            import yfinance as yf
            data = yf.download(ticker, start=buffer_start, end=eval_date + timedelta(days=5), progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]
    else:
        print(f"📥 Descargando datos desde yfinance...")
        import yfinance as yf
        data = yf.download(ticker, start=buffer_start, end=eval_date + timedelta(days=5), progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).lower() for c in data.columns]
    
    # Calcular features
    features = compute_e1_features(data, benchmark_df)
    
    # Crear targets
    lookback_days = config.get("lookback_days", 180)
    horizon_days = config.get("horizon_days", 90)
    
    close = data["close"].values
    target = np.full(len(close), np.nan)
    
    for i in range(len(close) - horizon_days):
        ret = (close[i + horizon_days] - close[i]) / close[i]
        target[i] = ret
    
    # Filtrar período de evaluación - crear Series con target
    target_series = pd.Series(target, index=data.index, name='target')
    
    # Filtrar solo las features que necesitamos
    features_eval = features[feature_names]
    
    # Filtrar por fechas
    eval_start = train_cutoff
    mask = (features_eval.index >= eval_start) & (features_eval.index <= eval_date)
    features_eval = features_eval[mask]
    target_eval = target_series[mask]
    
    if len(features_eval) < lookback_days:
        print(f"⚠️  No hay datos suficientes para evaluar (necesita {lookback_days} días, tiene {len(features_eval)})")
        return {}
    
    # Crear secuencias
    X_seq, y_seq, seq_timestamps, _ = make_sequences(
        features=features_eval,
        target=target_eval,
        lookback=lookback_days
    )
    
    print(f"✓ Período evaluación: {len(X_seq)} secuencias ({seq_timestamps[0].date()} → {seq_timestamps[-1].date()})")
    
    # Normalizar y predecir
    X_eval_s = (X_seq - scaler['mean_X']) / scaler['std_X']
    X_eval_s = np.nan_to_num(X_eval_s, nan=0.0)
    
    y_pred_s = model.predict(X_eval_s)
    y_pred = y_pred_s * scaler['std_y'] + scaler['mean_y']
    
    # Métricas ML
    mae = float(np.mean(np.abs(y_seq - y_pred)))
    rmse = float(np.sqrt(np.mean((y_seq - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_seq) == np.sign(y_pred)))
    
    from scipy.stats import spearmanr
    ic, _ = spearmanr(y_seq, y_pred)
    
    print(f"\n📊 Métricas Out-of-Time:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  IC: {ic:.4f}")
    print(f"  Dir Acc: {dir_acc:.2%}")
    
    # Backtest
    close_eval = data.loc[seq_timestamps, "close"].values
    
    tau_buy = config.get("thresholds", {}).get("tau_buy", 0.05)
    tau_sell = config.get("thresholds", {}).get("tau_sell", 0.0)
    
    bt = backtest_daily_signals(
        timestamps=seq_timestamps,
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
        'n_samples': len(y_seq),
        **{f'bt_{k}': float(v) for k, v in trading_metrics.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Validación Retrospectiva E1 Simple (GRU)")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker a evaluar")
    parser.add_argument("--train-days-ago", type=int, default=180, help="Días atrás para entrenar")
    parser.add_argument("--horizon", type=int, default=90, help="Horizonte de predicción")
    parser.add_argument("--config", type=str, default="src/config/base.yaml", help="Path a config YAML")
    parser.add_argument("--mlflow-docker", action="store_true", help="Usar MLflow en Docker")
    
    args = parser.parse_args()
    
    # Paths
    root = project_root()
    data_dir = root / "data" / "clean"
    
    # Configurar fechas
    today = datetime.now()
    train_cutoff = today - timedelta(days=args.train_days_ago)
    
    print("=" * 80)
    print("E1 Simple - Validación Retrospectiva (Out-of-Time) con GRU")
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
        # Fallback
        model_config = {
            'lookback_days': 180,
            'horizon_days': args.horizon,
            'thresholds': {'tau_buy': 0.05, 'tau_sell': 0.0},
            'costs': {'round_trip_bps': 10},
            'position': {'allow_short': False, 'max_position': 1.0}
        }
    
    # 1. Cargar/descargar datos hasta train_cutoff
    ohlcv = load_or_download_data(args.ticker, train_cutoff, data_dir)
    
    # Benchmark
    benchmark_ticker = config.get("universe", {}).get("benchmark", "SPY")
    try:
        benchmark_df = load_or_download_data(benchmark_ticker, train_cutoff, data_dir)
    except:
        print(f"⚠️  No se pudo cargar benchmark {benchmark_ticker}")
        benchmark_df = None
    
    # 2. Entrenar modelo GRU (simulando estar en train_cutoff)
    model, scaler, feature_names, train_end, ml_metrics_test = train_retrospective_model_gru(
        args.ticker, ohlcv, model_config, benchmark_df
    )
    
    # 3. Evaluar out-of-time (train_cutoff → hoy)
    oot_metrics = evaluate_out_of_time_gru(
        args.ticker, train_cutoff, today, model, scaler, feature_names, 
        model_config, data_dir, benchmark_df
    )
    
    # 4. Comparar métricas
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN: In-Sample Test vs. Out-of-Time")
    print("=" * 80)
    
    comparison = pd.DataFrame({
        'In-Sample Test': [
            ml_metrics_test['mae'],
            ml_metrics_test['rmse'],
            ml_metrics_test['ic'],
            ml_metrics_test['directional_accuracy']
        ],
        'Out-of-Time': [
            oot_metrics.get('mae', np.nan),
            oot_metrics.get('rmse', np.nan),
            oot_metrics.get('ic', np.nan),
            oot_metrics.get('directional_accuracy', np.nan)
        ]
    }, index=['MAE', 'RMSE', 'IC', 'Dir Acc'])
    
    comparison['Degradación %'] = ((comparison['Out-of-Time'] - comparison['In-Sample Test']) / 
                                     comparison['In-Sample Test'].abs() * 100)
    
    print(comparison.to_string())
    print()
    
    # Alertas
    if oot_metrics:
        alerts = []
        
        ic_deg = abs(comparison.loc['IC', 'Degradación %'])
        mae_deg = abs(comparison.loc['MAE', 'Degradación %'])
        
        if ic_deg > 20:
            alerts.append(f"⚠️  IC degradó {ic_deg:.1f}%")
        
        if mae_deg > 20:
            alerts.append(f"⚠️  MAE empeoró {mae_deg:.1f}%")
        
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
            mlflow_uri = "http://localhost:5050" if args.mlflow_docker else f"sqlite:///{root}/runs/mlflow_local/mlflow.db"
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("E1_Simple_Retrospective_Validation")
            
            with mlflow.start_run(run_name=f"retrospective_{args.ticker}_{train_cutoff.strftime('%Y%m%d')}"):
                # Params
                mlflow.log_param("ticker", args.ticker)
                mlflow.log_param("train_cutoff_date", train_cutoff.strftime("%Y-%m-%d"))
                mlflow.log_param("eval_date", today.strftime("%Y-%m-%d"))
                mlflow.log_param("train_days_ago", args.train_days_ago)
                mlflow.log_param("horizon_days", args.horizon)
                mlflow.log_param("model_type", "GRU")
                
                # Métricas in-sample test
                mlflow.log_metric("test_mae", ml_metrics_test['mae'])
                mlflow.log_metric("test_rmse", ml_metrics_test['rmse'])
                mlflow.log_metric("test_ic", ml_metrics_test['ic'])
                mlflow.log_metric("test_dir_acc", ml_metrics_test['directional_accuracy'])
                
                # Métricas out-of-time
                for k, v in oot_metrics.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        mlflow.log_metric(f"oot_{k}", float(v))
                
                # Degradación
                mlflow.log_metric("ic_degradation_pct", comparison.loc['IC', 'Degradación %'])
                mlflow.log_metric("mae_degradation_pct", comparison.loc['MAE', 'Degradación %'])
                
                print(f"✓ Resultados guardados en MLflow")
                print(f"  Experimento: E1_Simple_Retrospective_Validation")
        
        except Exception as e:
            print(f"⚠️  No se pudo guardar en MLflow: {e}")
    
    print("\n" + "=" * 80)
    print("✓ Validación retrospectiva completada")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Pipeline completo E2 (Estrategia Moderada).

Flujo:
1. Cargar datos raw
2. Calcular features E2 (momentum-focused)
3. Crear target y secuencias
4. Split temporal
5. Estandarizar features
6. Entrenar LSTM
7. Evaluar y guardar resultados

Diferencias vs E1:
- Features: build_features_e2 (RSI, Stochastic, ROC, OBV)
- Modelo: LSTM en lugar de GRU
- Lookback: 60 días (vs 180 de E1)
- Horizon: 20 días (vs 90 de E1)
- Early stopping patience: 12 (vs 15 de E1)
- Max epochs: 150 (vs 200 de E1)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features.build_features_e2 import compute_e2_features, make_target_e2
from .features.build_sequences import make_sequences, time_split
from .models.e2_lstm import LSTMRegressor
from .backtest.daily import backtest_daily_signals, summarize_backtest
from .utils import ensure_dir, load_yaml, project_root


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Carga CSV OHLCV y lo prepara."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    return df


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Information Coefficient (correlación) de forma robusta."""
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_e2_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    """Ejecuta pipeline E2 para un ticker."""
    ensure_dir(out_dir)

    # 1. Cargar datos
    raw_path = raw_dir / f"{ticker}_daily.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"No existe {raw_path}")

    df_raw = load_ohlcv_csv(raw_path)
    print(f"\n{'='*60}")
    print(f"Procesando {ticker} (E2 - Moderada)")
    print(f"{'='*60}")
    print(f"Período: {df_raw.index[0].date()} a {df_raw.index[-1].date()} ({len(df_raw)} días)")

    # 2. Calcular features E2
    feat_df = compute_e2_features(df_raw, benchmark_df=benchmark_df)

    # 3. Crear target
    e2_cfg = config.get("strategies", {}).get("e2_moderate", {})
    horizon_days = e2_cfg.get("horizon_days", 20)
    target_series = make_target_e2(df_raw, horizon_days=horizon_days)

    # Combinar
    df_full = feat_df.join(target_series, how="inner")
    df_full = df_full.dropna()

    if df_full.empty:
        raise ValueError(f"{ticker}: Sin datos después de dropna")

    print(f"Features calculadas: {len(feat_df.columns)}")
    print(f"Datos válidos: {len(df_full)}")

    # 4. Separar X, y
    feat_cols = [c for c in df_full.columns if c != target_series.name]
    lookback_days = e2_cfg.get("lookback_days", 60)
    
    print(f"Lookback: {lookback_days} días, Horizon: {horizon_days} días")

    # Adaptación para usar make_sequences de build_sequences.py
    # La firma retornada es: X, y, ts, feat_cols
    # PERO, make_sequences en train_e2_pipeline invocado recibe 5 args en el código actual: X_raw, y_raw, timestamps, lookback, dropna
    # Y build_sequences.make_sequences espera (features_df, target_series, lookback)
    
    # CORRECCIÓN: Usar build_sequences correctamente
    # Primero necesitamos DataFrame y Series
    # X_raw proviene de df_full[feat_cols]
    # y_raw proviene de df_full[target_series.name]
    
    X_seq, y_seq, ts_seq, _ = make_sequences(
        df_full[feat_cols], df_full[target_series.name], lookback=lookback_days
    )

    print(f"Secuencias: {len(X_seq)} (shape: {X_seq.shape})")

    # 6. Split temporal
    splits_cfg = config.get("splits", {})
    train_ratio = 0.7 
    val_ratio = 0.15
    # Simplification: use time_split instead of walk-forward for now to keep it minimal
    
    idx_train, idx_val, idx_test = time_split(
        len(X_seq), train_frac=train_ratio, val_frac=val_ratio
    )

    X_train, y_train, ts_train = X_seq[idx_train], y_seq[idx_train], ts_seq[idx_train]
    X_val, y_val, ts_val = X_seq[idx_val], y_seq[idx_val], ts_seq[idx_val]
    X_test, y_test, ts_test = X_seq[idx_test], y_seq[idx_test], ts_seq[idx_test]

    print(f"\nSplit temporal:")
    print(f"  Train: {len(X_train)} ({ts_train[0]} → {ts_train[-1]})")
    print(f"  Val:   {len(X_val)} ({ts_val[0]} → {ts_val[-1]})")
    print(f"  Test:  {len(X_test)} ({ts_test[0]} → {ts_test[-1]})")

    # 7. Estandarizar features (solo sobre train)
    mean_X = X_train.mean(axis=(0, 1))
    std_X = X_train.std(axis=(0, 1)) + 1e-8

    X_train_norm = (X_train - mean_X) / std_X
    X_val_norm = (X_val - mean_X) / std_X
    X_test_norm = (X_test - mean_X) / std_X

    # Estandarizar target
    mean_y = y_train.mean()
    std_y = y_train.std() + 1e-8

    y_train_norm = (y_train - mean_y) / std_y
    y_val_norm = (y_val - mean_y) / std_y

    print(f"\nTarget statistics (train):")
    print(f"  Mean: {mean_y:.4f}, Std: {std_y:.4f}")

    # 8. Entrenar LSTM
    print("\nEntrenando modelo LSTM...")
    model_cfg = e2_cfg.get("model", {}) # Usa config correcta

    model = LSTMRegressor(
        input_size=X_train_norm.shape[2],
        hidden_sizes=model_cfg.get("lstm_units", [128, 64]), # Use lstm_units from yaml
        dropout=model_cfg.get("dropout", 0.2),
        dense_units=model_cfg.get("dense_units", 32),
        seed=config.get("project", {}).get("seed", 42), # Use project seed
    )

    res = model.fit(
        X_train_norm,
        y_train_norm,
        X_val_norm,
        y_val_norm,
        learning_rate=model_cfg.get("learning_rate", 1e-3),
        batch_size=model_cfg.get("batch_size", 64),
        max_epochs=model_cfg.get("max_epochs", 150),
        early_stopping_patience=model_cfg.get("early_stopping_patience", 12),
        loss=model_cfg.get("loss", "huber"),
        huber_delta=model_cfg.get("huber_delta", 1.0),
    )

    print(f"✓ Entrenamiento completado:")
    print(f"  Epochs: {res.epochs_ran}")
    print(f"  Val loss: {res.best_val_loss:.6f}")

    # 8.5 Guardar modelo entrenado (por ticker)
    # Guardamos state_dict + metadata mínima para inferencia/reproducibilidad.
    try:
        torch = model.torch
        model_path = out_dir / f"{ticker}_model.pth"
        payload = {
            "ticker": ticker,
            "strategy": "e2_moderate",
            "created_at": datetime.utcnow().isoformat(),
            "model_class": "LSTMRegressor",
            "model_kwargs": {
                "input_size": int(X_train_norm.shape[2]),
                "hidden_sizes": list(model_cfg.get("lstm_units", [128, 64])),
                "dropout": float(model_cfg.get("dropout", 0.2)),
                "dense_units": int(model_cfg.get("dense_units", 32)),
                "seed": int(config.get("project", {}).get("seed", 42)),
            },
            "lookback_days": int(lookback_days),
            "horizon_days": int(horizon_days),
            "feature_names": list(feat_cols),
            "scaler_X": {"mean": mean_X.tolist(), "std": std_X.tolist()},
            "scaler_y": {"mean": float(mean_y), "std": float(std_y)},
            "train_result": {"epochs_ran": int(res.epochs_ran), "best_val_loss": float(res.best_val_loss)},
            "state_dict": {k: v.detach().cpu() for k, v in model.model.state_dict().items()},
        }
        torch.save(payload, model_path)
        print(f"✓ Modelo guardado: {model_path}")
    except Exception as exc:
        print(f"⚠️  No se pudo guardar el modelo para {ticker}: {exc}")

    # 9. Predicciones en test (desnormalizar)
    y_pred_norm = model.predict(X_test_norm)
    y_pred = y_pred_norm * std_y + mean_y

    # 10. Métricas ML
    mae = np.abs(y_test - y_pred).mean()
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
    ic = compute_information_coefficient(y_test, y_pred)

    # Directional accuracy
    dir_acc = float(np.mean((np.sign(y_test) == np.sign(y_pred))))

    ml_metrics = {
        "mae": mae,
        "rmse": rmse,
        "ic": ic,
        "directional_accuracy": dir_acc,
    }

    print(f"\nMétricas ML (test):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  IC:   {ic:.3f}")
    print(f"  Directional Accuracy: {dir_acc:.2%}")

    # 11. Backtest simple
    pred_df = pd.DataFrame(
        {
            "timestamp": ts_test,
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df = pred_df.set_index("timestamp")

    # Obtener precios de test
    df_test = df_raw.loc[pred_df.index]

    # Señales básicas E2 (Vectorizado simplificado para reporte rápido)
    # Nota: Para backtest riguroso con reglas complejas (TP/SL/TimeStop/Filtros),
    # se debería usar el módulo src.backtest con loop evento a evento.
    # Aquí hacemos una aproximación vectorizada para validar el modelo.
    
    tau_buy = e2_cfg.get("thresholds", {}).get("tau_buy", 0.025)
    tau_sell = e2_cfg.get("thresholds", {}).get("tau_sell", 0.00)

    signals = pd.Series(0, index=pred_df.index)
    signals[pred_df["y_pred"] > tau_buy] = 1   # Long
    signals[pred_df["y_pred"] < tau_sell] = -1  # Short/Exit
    
    # Aplicar filtros simples vectores si existen columnas (RSI, MACD) en df_test
    # Esto es una aproximación de rules_e2.py
    if "rsi_14" in df_test.columns:
         rsi_mask = (df_test["rsi_14"] >= 35) & (df_test["rsi_14"] <= 70)
         # Solo permitimos entrada (1) si RSI ok. Salidas (-1) siempre permitidas.
         signals[(signals == 1) & (~rsi_mask)] = 0
         
    if "macd_hist" in df_test.columns:
        macd_mask = df_test["macd_hist"] > 0
        signals[(signals == 1) & (~macd_mask)] = 0

    # backtest_daily_signals necesita timestamps, close, pred_returns, tau_buy, tau_sell
    # Usar wrapper simplificado
    bt_results = backtest_daily_signals(
        timestamps=pd.DatetimeIndex(pred_df.index),
        close_prices=df_test["close"].to_numpy(),
        pred_returns=pred_df["y_pred"].to_numpy(),
        tau_buy=tau_buy,
        tau_sell=tau_sell,
    )
    trading_metrics = summarize_backtest(bt_results)

    print(f"\nMétricas Trading (test):")
    print(f"  Total Return: {trading_metrics['total_return']:.2%}")
    print(f"  Sharpe:       {trading_metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {trading_metrics['max_drawdown']:.2%}")

    # 12. Guardar resultados
    pred_df.to_csv(out_dir / f"{ticker}_predictions.csv")

    # Guardar scaler de features
    feat_names = feat_cols
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")

    # Guardar scaler de target
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],
        "std_y": [std_y]
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    meta = {
        "ticker": ticker,
        "n_samples": int(len(X_seq)),
        "n_test": int(len(X_test)),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "split_method": splits_cfg.get("method", "time_split"),
        "epochs_ran": res.epochs_ran,
        "val_loss": res.best_val_loss,
    }

    summary = {
        **meta,
        **{f"ml_{k}": v for k, v in ml_metrics.items()},
        **{f"bt_{k}": v for k, v in trading_metrics.items()},
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E2 (LSTM moderado)")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e2_moderate)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    # Tickers E2
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e2_moderate", [])
        )

    if not tickers:
        raise ValueError("No tickers for E2")

    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data" / "raw" / "daily"

    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        print(f"⚠️  Benchmark {benchmark} no encontrado, usando valores vacíos")
        benchmark_df = None

    out_base = root / "runs" / "e2_moderate" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_base)

    # Guardar config usado
    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    summaries: list[dict] = []
    for ticker in tickers:
        ticker_out = out_base / ticker
        try:
            summary = run_e2_for_ticker(
                config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out, benchmark_df=benchmark_df
            )
            summaries.append(summary)
            print(f"✓ {ticker}: MAE={summary['ml_mae']:.4f} IC={summary['ml_ic']:.3f}\n")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)
    print(f"\n✓ Resultados guardados en {out_base}")


if __name__ == "__main__":
    main()

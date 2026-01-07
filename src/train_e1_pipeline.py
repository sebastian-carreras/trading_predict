"""
Pipeline completo E1 (Estrategia Conservadora).

Flujo:
1. Cargar datos raw
2. Calcular features
3. Crear target y secuencias
4. Split temporal
5. Estandarizar features
6. Entrenar GRU
7. Evaluar y guardar resultados
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features.build_features_e1 import compute_e1_features, make_target_e1
from .features.build_sequences import make_sequences, time_split
from .models.e1_gru import GRURegressor
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


def run_e1_for_ticker(
    config: dict, ticker: str, raw_dir: Path, out_dir: Path, benchmark_df: pd.DataFrame
) -> dict:
    """Entrena y evalúa E1 para un ticker."""

    # Parámetros E1 del config
    e1 = config.get("strategies", {}).get("e1_conservative", {})
    lookback_days = int(e1.get("lookback_days", 180))
    horizon_days = int(e1.get("horizon_days", 90))

    model_cfg = e1.get("model", {})
    gru_units = model_cfg.get("gru_units", [128, 64])
    dropout = float(model_cfg.get("dropout", 0.2))
    dense_units = int(model_cfg.get("dense_units", 32))
    lr = float(model_cfg.get("learning_rate", 1e-3))
    batch_size = int(model_cfg.get("batch_size", 64))
    max_epochs = int(model_cfg.get("max_epochs", 200))
    patience = int(model_cfg.get("early_stopping_patience", 15))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))

    seed = int(config.get("project", {}).get("seed", 42))

    # Cargar datos (priorizar datos limpios si existen)
    # raw_dir es data/raw/daily, entonces data/clean está en raw_dir.parent.parent / "clean"
    clean_dir = raw_dir.parent.parent / "clean"
    clean_csv_path = clean_dir / f"{ticker}_daily.csv"
    
    if clean_csv_path.exists():
        csv_path = clean_csv_path
        print(f"  ✓ Usando datos limpios: {csv_path.name}")
    else:
        csv_path = raw_dir / f"{ticker}_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing daily CSV: {csv_path}")
        print(f"  ⚠️  Usando datos raw (limpieza no ejecutada): {csv_path.name}")

    ohlcv = load_ohlcv_csv(csv_path)

    # Features
    features = compute_e1_features(ohlcv, benchmark_df)
    target = make_target_e1(ohlcv, horizon_days=horizon_days)

    # Secuencias
    X, y, ts, feat_names = make_sequences(features, target, lookback=lookback_days)

    # Split temporal
    idx_train, idx_val, idx_test = time_split(len(X))
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = ts[idx_test]

    # Estandarizar features X (fit solo en train)
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    mean_X = Xtr2d.mean(axis=0)
    std_X = Xtr2d.std(axis=0) + 1e-12

    def scale_X(Xa: np.ndarray) -> np.ndarray:
        return ((Xa - mean_X) / std_X).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    # Estandarizar targets y (fit solo en train) para reducir bias
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-12

    def scale_y(ya: np.ndarray) -> np.ndarray:
        return ((ya - mean_y) / std_y).astype(np.float32)

    def unscale_y(ya_scaled: np.ndarray) -> np.ndarray:
        return (ya_scaled * std_y + mean_y).astype(np.float32)

    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)
    y_test_s = scale_y(y_test)

    # Entrenar GRU
    model = GRURegressor(
        input_size=X_train_s.shape[-1],
        hidden_sizes=gru_units,
        dropout=dropout,
        dense_units=dense_units,
        seed=seed,
    )

    print(f"Entrenando GRU para {ticker}...")
    res = model.fit(
        X_train_s,
        y_train_s,  # Targets normalizados
        X_val_s,
        y_val_s,    # Targets normalizados
        learning_rate=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        loss=loss,
        huber_delta=huber_delta,
    )

    print(f"  Epochs: {res.epochs_ran}, Val Loss: {res.best_val_loss:.6f}")

    # Predicciones en test (normalizadas)
    y_pred_s = model.predict(X_test_s)
    
    # Desnormalizar predicciones para métricas y backtesting
    y_pred = unscale_y(y_pred_s)

    # Métricas ML
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic = float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 1 else 0.0

    ml = {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "ic": ic}

    # Backtesting
    # Obtener precios de cierre del período de test
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()
    
    # Umbrales de trading del config
    thresholds = e1.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.02))
    tau_sell = float(thresholds.get("tau_sell", 0.02))
    
    # Costos de transacción
    costs = config.get("costs", {})
    round_trip_bps = float(costs.get("daily_round_trip_bps", 10))
    
    # Parámetros de backtest
    bt_cfg = e1.get("backtest", {})
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    allow_short = bool(bt_cfg.get("allow_short", False))
    
    # Ejecutar backtest
    bt = backtest_daily_signals(
        timestamps=ts_test,
        close_prices=close_prices_test,
        pred_returns=y_pred,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
    )
    
    # Métricas de trading
    trading_metrics = summarize_backtest(bt)

    # Guardar outputs
    ensure_dir(out_dir)

    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")
    
    # Guardar backtest
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")

    # Guardar scaler de features X
    scaler_X_df = pd.DataFrame({"mean": mean_X, "std": std_X}, index=feat_names)
    scaler_X_df.to_csv(out_dir / f"{ticker}_scaler.csv")
    
    # Guardar scaler de target y (para inferencia futura)
    scaler_y_df = pd.DataFrame({
        "mean_y": [mean_y],
        "std_y": [std_y]
    })
    scaler_y_df.to_csv(out_dir / f"{ticker}_target_scaler.csv", index=False)

    meta = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_test": int(len(X_test)),
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        "epochs_ran": res.epochs_ran,
        "val_loss": res.best_val_loss,
    }

    summary = {
        **meta, 
        **{f"ml_{k}": v for k, v in ml.items()},
        **{f"bt_{k}": v for k, v in trading_metrics.items()},
    }
    pd.Series(summary).to_csv(out_dir / f"{ticker}_summary.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline E1 (GRU conservador)")
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
        help="Comma-separated tickers (or uses universe.tickers_by_strategy.e1_conservative)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    # Tickers E1
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e1_conservative", [])
        )

    if not tickers:
        raise ValueError("No tickers for E1")

    # Benchmark
    benchmark = config.get("universe", {}).get("benchmark", "SPY")
    raw_dir = root / "data" / "raw" / "daily"

    benchmark_path = raw_dir / f"{benchmark}_daily.csv"
    if benchmark_path.exists():
        benchmark_df = load_ohlcv_csv(benchmark_path)
    else:
        print(f"⚠️  Benchmark {benchmark} no encontrado, usando valores vacíos")
        benchmark_df = None

    out_base = root / "runs" / "e1_conservative" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_base)

    # Guardar config usado
    import shutil
    shutil.copy(cfg_path, out_base / "config_used.yaml")

    summaries: list[dict] = []
    for ticker in tickers:
        ticker_out = out_base / ticker
        try:
            summary = run_e1_for_ticker(
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

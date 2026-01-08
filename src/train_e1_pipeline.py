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
from sklearn.model_selection import TimeSeriesSplit

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


def compute_information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Information Coefficient (correlación) de forma robusta."""
    if len(y_true) <= 1:
        return float("nan")

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def save_walkforward_plot(fold_df: pd.DataFrame, ticker: str, out_path: Path) -> None:
    """Guarda grafico de IC y Sharpe por fold."""
    if fold_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - fallback si matplotlib no está
        print(f"⚠️  No se pudo generar el gráfico walk-forward para {ticker}: {exc}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(fold_df["fold"], fold_df["ml_ic"], marker="o")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("IC")
    axes[0].set_title(f"{ticker} - Walk-Forward Validation")

    axes[1].plot(fold_df["fold"], fold_df["bt_sharpe"], marker="o", color="#2ca02c")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xlabel("Fold")

    if "window" in fold_df.columns:
        axes[1].set_xticks(fold_df["fold"].to_numpy())
        axes[1].set_xticklabels(fold_df["window"].to_list(), rotation=35, ha="right")

    for ax in axes:
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_e1_walk_forward(
    *,
    X: np.ndarray,
    y: np.ndarray,
    ts: pd.DatetimeIndex,
    ticker: str,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    splits_cfg: dict,
    gru_units: list[int],
    dropout: float,
    dense_units: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    loss: str,
    huber_delta: float,
    tau_buy: float,
    tau_sell: float,
    round_trip_bps: float,
    holding_period: int,
    allow_short: bool,
    max_position: float,
    seed: int,
    lookback_days: int,
    horizon_days: int,
) -> dict:
    """Ejecuta validación walk-forward para la estrategia E1."""

    ensure_dir(out_dir)

    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("No hay muestras disponibles para walk-forward")

    folds = int(splits_cfg.get("folds", 5))
    if folds < 1:
        raise ValueError("'folds' debe ser >= 1 para walk-forward")

    embargo_cfg = splits_cfg.get("embargo_days", {})
    if isinstance(embargo_cfg, dict):
        embargo_days = int(embargo_cfg.get("e1", 0))
    else:
        embargo_days = int(embargo_cfg or 0)

    # Para datos diarios aproximamos gap como cantidad de muestras (1 muestra ≈ 1 día hábil)
    gap_samples = max(0, int(embargo_days))

    # Tamaño de test (en muestras) por fold. Por defecto usamos partición uniforme.
    default_test_size = max(1, n_samples // (folds + 1))
    test_size = int(splits_cfg.get("test_size", default_test_size))
    if test_size <= gap_samples:
        test_size = gap_samples + 1

    splitter = TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap_samples)

    fold_summaries: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    root = project_root()

    def as_relative(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    for fold_idx, (train_full_idx, test_idx) in enumerate(
        splitter.split(np.arange(n_samples)), start=1
    ):
        if len(test_idx) == 0 or len(train_full_idx) == 0:
            continue

        # Split interno train/val dentro del bloque de entrenamiento
        tr_idx_sub, val_idx_sub, _ = time_split(len(train_full_idx))
        train_idx = train_full_idx[tr_idx_sub]
        val_idx = train_full_idx[val_idx_sub]

        if len(val_idx) == 0:
            split_point = max(1, int(len(train_full_idx) * 0.8))
            train_idx = train_full_idx[:split_point]
            val_idx = train_full_idx[split_point:]

        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        X_tr_2d = X_train.reshape(-1, X_train.shape[-1])
        mean_X = X_tr_2d.mean(axis=0)
        std_X = X_tr_2d.std(axis=0) + 1e-12

        def scale_X(data: np.ndarray) -> np.ndarray:
            return ((data - mean_X) / std_X).astype(np.float32)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

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

        model = GRURegressor(
            input_size=X_train.shape[-1],
            hidden_sizes=list(gru_units),
            dropout=dropout,
            dense_units=dense_units,
            seed=seed,
        )

        res = model.fit(
            X_train_s,
            y_train_s,
            X_val_s,
            y_val_s,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            loss=loss,
            huber_delta=huber_delta,
        )

        y_pred_s = model.predict(X_test_s)
        y_pred = unscale_y(y_pred_s)

        mae = float(np.mean(np.abs(y_test - y_pred)))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
        ic = compute_information_coefficient(y_test, y_pred)

        ts_test = ts[test_idx]
        window_label = f"{ts_test[0].date()} -> {ts_test[-1].date()}"

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

        sharpe = trading_metrics.get("sharpe", float("nan"))

        ic_str = "nan" if np.isnan(ic) else f"{ic:.3f}"
        sharpe_str = "nan" if np.isnan(sharpe) else f"{sharpe:.2f}"
        print(
            f"    Fold {fold_idx}: {window_label} | MAE={mae:.4f} IC={ic_str} Sharpe={sharpe_str}"
        )

        bt.to_csv(out_dir / f"{ticker}_fold{fold_idx}_backtest.csv")

        fold_summaries.append(
            {
                "fold": fold_idx,
                "window": window_label,
                "test_start": ts_test[0],
                "test_end": ts_test[-1],
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "epochs_ran": int(res.epochs_ran),
                "val_loss": float(res.best_val_loss),
                "ml_mae": mae,
                "ml_rmse": rmse,
                "ml_directional_accuracy": dir_acc,
                "ml_ic": ic,
                **{f"bt_{k}": float(v) for k, v in trading_metrics.items()},
            }
        )

        preds_df = pd.DataFrame(
            {
                "fold": fold_idx,
                "y_true": y_test,
                "y_pred": y_pred,
            },
            index=ts_test,
        )
        preds_df.index.name = "timestamp"
        pred_frames.append(preds_df)

    if not fold_summaries:
        raise ValueError("No se generaron folds válidos para walk-forward")

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(out_dir / f"{ticker}_walkforward_folds.csv", index=False)

    combined_preds = pd.concat(pred_frames).sort_index()
    combined_preds.to_csv(out_dir / f"{ticker}_walkforward_predictions.csv")

    y_true_all = combined_preds["y_true"].to_numpy()
    y_pred_all = combined_preds["y_pred"].to_numpy()

    mae_all = float(np.mean(np.abs(y_true_all - y_pred_all)))
    rmse_all = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    dir_acc_all = float(np.mean(np.sign(y_true_all) == np.sign(y_pred_all)))
    ic_all = compute_information_coefficient(y_true_all, y_pred_all)

    bt_all = backtest_daily_signals(
        timestamps=combined_preds.index,
        close_prices=ohlcv.loc[combined_preds.index, "close"].to_numpy(),
        pred_returns=y_pred_all,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        holding_period_days=holding_period,
        allow_short=allow_short,
        max_position=max_position,
    )
    trading_metrics_all = summarize_backtest(bt_all)
    bt_all.to_csv(out_dir / f"{ticker}_walkforward_backtest.csv")

    plot_path = out_dir / f"{ticker}_walkforward_metrics.png"
    save_walkforward_plot(fold_df, ticker, plot_path)

    summary = {
        "ticker": ticker,
        "n_samples": int(n_samples),
        "n_test": int(len(combined_preds)),
        "folds": int(len(fold_df)),
        "lookback_days": int(lookback_days),
        "horizon_days": int(horizon_days),
        "split_method": "walk_forward",
        "walkforward_test_size": int(test_size),
        "walkforward_gap": int(gap_samples),
        "ml_mae": mae_all,
        "ml_rmse": rmse_all,
        "ml_directional_accuracy": dir_acc_all,
        "ml_ic": ic_all,
        **{f"bt_{k}": float(v) for k, v in trading_metrics_all.items()},
        "folds_file": as_relative(out_dir / f"{ticker}_walkforward_folds.csv"),
        "predictions_file": as_relative(out_dir / f"{ticker}_walkforward_predictions.csv"),
        "backtest_file": as_relative(out_dir / f"{ticker}_walkforward_backtest.csv"),
        "plot_file": as_relative(plot_path),
    }

    return summary


def run_e1_for_ticker(
    config: dict, ticker: str, raw_dir: Path, out_dir: Path, benchmark_df: pd.DataFrame
) -> dict:
    """Entrena y evalúa E1 para un ticker."""

    # Parámetros E1 del config
    e1 = config.get("strategies", {}).get("e1_conservative", {})
    lookback_days = int(e1.get("lookback_days", 180))
    horizon_days = int(e1.get("horizon_days", 90))

    model_cfg = e1.get("model", {})
    gru_units = model_cfg.get("gru_units", [96, 32])
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

    thresholds = e1.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.04))
    tau_sell = float(thresholds.get("tau_sell", -0.02))

    costs_cfg = config.get("costs", {})
    round_trip_bps = float(costs_cfg.get("daily_round_trip_bps", 10))

    bt_cfg = e1.get("backtest", {})
    holding_period = int(bt_cfg.get("holding_period_days", horizon_days))
    allow_short = bool(bt_cfg.get("allow_short", False))
    max_position = float(bt_cfg.get("max_position", 1.0))

    splits_cfg = config.get("splits", {})
    if splits_cfg.get("method") == "walk_forward":
        print(
            f"  ▶ Ejecutando walk-forward ({splits_cfg.get('folds', 5)} folds, test={splits_cfg.get('test_size', 'auto')})"
        )
        summary = run_e1_walk_forward(
            X=X,
            y=y,
            ts=ts,
            ticker=ticker,
            out_dir=out_dir,
            ohlcv=ohlcv,
            splits_cfg=splits_cfg,
            gru_units=list(gru_units),
            dropout=dropout,
            dense_units=dense_units,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            loss=loss,
            huber_delta=huber_delta,
            tau_buy=tau_buy,
            tau_sell=tau_sell,
            round_trip_bps=round_trip_bps,
            holding_period=holding_period,
            allow_short=allow_short,
            max_position=max_position,
            seed=seed,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
        )
        return summary

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
    ic = compute_information_coefficient(y_test, y_pred)

    ml = {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "ic": ic}

    # Backtesting
    # Obtener precios de cierre del período de test
    close_prices_test = ohlcv.loc[ts_test, "close"].to_numpy()
    
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
        max_position=max_position,
    )
    
    # Métricas de trading
    trading_metrics = summarize_backtest(bt)

    # Guardar outputs
    ensure_dir(out_dir)

    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=ts_test)
    preds_df.index.name = "timestamp"
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
        "split_method": splits_cfg.get("method", "time_split"),
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

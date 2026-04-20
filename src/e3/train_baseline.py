"""
Pipeline de baseline E3 - Ridge Regression (5-min intradiario).

Establece el punto de referencia del lifecycle para comparar contra el LSTM intradiario.
Usa split temporal simple (70% train / 15% val / 15% test) sobre datos de 5-min.

Requiere que los datos 5-min ya existan en data/raw/intraday/.
Para descargar datos previamente:
    python -m src.e3.intraday_data --tickers SPY

Uso:
    python -m src.e3.train_baseline --tickers SPY
    python -m src.e3.train_baseline --tickers SPY,AAPL,NVDA
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import socket
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .build_features import compute_intraday_features, make_target_return, make_sequences
from .intraday_data import load_ohlcv_csv
from .intraday_metrics import mae, rmse, directional_accuracy, information_coefficient
from ..backtest.backtest_intraday import (
    backtest_intraday_signals,
    compute_profit_factor,
    compute_max_drawdown,
)
from .baseline_linear import IntradayRidgeBaseline
from ..utils import apply_training_window, ensure_dir, load_yaml, project_root

# 5-min bars per year: 252 trading days × 6.5 trading hours × 12 bars/hour
_BARS_PER_YEAR = 252 * 78


def summarize_backtest_intraday(bt_df: pd.DataFrame) -> dict:
    """Computa métricas de trading desde el DataFrame de backtest intradiario."""
    net_ret = bt_df["net_ret"].to_numpy(dtype=float)
    equity = bt_df["equity"].to_numpy(dtype=float)
    pos = bt_df["pos"].to_numpy(dtype=float)
    turnover = bt_df["turnover"].to_numpy(dtype=float)

    n_bars = len(net_ret)
    n_years = n_bars / _BARS_PER_YEAR

    total_return = float(equity[-1] - 1.0) if len(equity) > 0 else 0.0

    cagr = 0.0
    if n_years > 0 and equity[-1] > 0:
        cagr = float(equity[-1] ** (1.0 / n_years) - 1.0)

    std_ret = float(np.std(net_ret))
    mean_ret = float(np.mean(net_ret))
    sharpe = float(mean_ret / (std_ret + 1e-12) * np.sqrt(_BARS_PER_YEAR))

    max_dd = compute_max_drawdown(equity)
    calmar = cagr / (max_dd + 1e-12) if max_dd > 0 else 0.0

    profit_factor = compute_profit_factor(net_ret)
    time_in_market = float(np.mean(pos != 0))
    avg_turnover = float(np.mean(turnover))
    total_costs = float(bt_df["cost"].sum())

    def _safe(x: float) -> float:
        return float(x) if np.isfinite(x) else 0.0

    return {
        "total_return": _safe(total_return),
        "cagr": _safe(cagr),
        "sharpe": _safe(sharpe),
        "max_drawdown": _safe(max_dd),
        "calmar": _safe(calmar),
        "profit_factor": _safe(profit_factor),
        "time_in_market": _safe(time_in_market),
        "avg_turnover": _safe(avg_turnover),
        "total_costs": _safe(total_costs),
    }


def run_baseline_for_ticker(
    config: dict,
    ticker: str,
    raw_dir: Path,
    out_dir: Path,
    mlflow_enabled: bool = False,
    mlflow=None,
    timestamp: str | None = None,
    use_latest_data: bool = False,
) -> dict:
    """
    Ejecuta pipeline de baseline E3 para un ticker.

    Usa Ridge Regression con split temporal simple (70/15/15) sobre datos 5-min.
    El objetivo es establecer un floor de métricas para el lifecycle registry.
    """
    start_time = time.perf_counter()
    ensure_dir(out_dir)

    print(f"\n{'='*60}")
    print(f"BASELINE E3 - {ticker}")
    print(f"{'='*60}")

    # 1. CARGAR DATOS 5-MIN
    csv_path = raw_dir / f"{ticker}_5m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No existe {csv_path}. Descargá los datos primero con:\n"
            f"  python -m src.e3.intraday_data --tickers {ticker}"
        )

    print(f"  Cargando datos: {csv_path.name}")
    ohlcv = load_ohlcv_csv(csv_path)
    ohlcv = apply_training_window(
        ohlcv, config, granularity="intraday", use_latest=use_latest_data, ticker=ticker,
    )
    print(f"  {len(ohlcv)} barras de 5min ({ohlcv.index[0]} → {ohlcv.index[-1]})")

    # Retornos de 1-barra para el backtest
    bar_ret_series = pd.Series(
        np.log(ohlcv["close"] / ohlcv["close"].shift(1)).fillna(0).to_numpy(dtype=float),
        index=ohlcv.index,
    )

    # 2. CALCULAR FEATURES
    e3_cfg = config.get("strategies", {}).get("e3_intraday", {})
    print("Calculando features (10 indicadores)...")
    df_feat = compute_intraday_features(ohlcv)

    # 3. CREAR TARGET
    horizon_bars = int(e3_cfg.get("horizon_bars", 6))
    print(f"Creando target (horizonte={horizon_bars} barras = {horizon_bars * 5} min)...")
    target_series = make_target_return(ohlcv, horizon_bars=horizon_bars)

    # 4. CREAR SECUENCIAS (N, lookback_bars, n_features)
    lookback_bars = int(e3_cfg.get("lookback_bars", 96))
    print(f"Creando secuencias (lookback={lookback_bars} barras = {lookback_bars * 5} min)...")

    X, y, timestamps, feature_names = make_sequences(
        features=df_feat,
        target=target_series,
        lookback_bars=lookback_bars,
    )

    print(f"Forma de datos: X={X.shape}, y={y.shape}")
    print(f"Features ({len(feature_names)}): {', '.join(feature_names)}")

    # 5. SPLIT TEMPORAL SIMPLE (70 / 15 / 15) — sin shuffle, preserva orden temporal
    n = len(X)
    e3_splits = e3_cfg.get("splits", {})
    train_frac = float(e3_splits.get("train_frac", 0.70))
    val_frac = float(e3_splits.get("val_frac", 0.15))

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)

    print(
        f"\nSplit temporal: train={len(train_idx)}, val={len(val_idx)}, "
        f"test={len(test_idx)} muestras"
    )
    print(f"  Train: {timestamps[train_idx[0]]} → {timestamps[train_idx[-1]]}")
    print(f"  Val:   {timestamps[val_idx[0]]} → {timestamps[val_idx[-1]]}")
    print(f"  Test:  {timestamps[test_idx[0]]} → {timestamps[test_idx[-1]]}")

    # ESTANDARIZAR FEATURES: stats calculadas sólo desde train (no leakage)
    scaler = StandardScaler()
    X_train_2d = X[train_idx].reshape(-1, X.shape[-1])
    scaler.fit(X_train_2d)

    def _scale(idx):
        return scaler.transform(
            X[idx].reshape(-1, X.shape[-1])
        ).reshape(X[idx].shape)

    X_train_s = _scale(train_idx)
    X_val_s = _scale(val_idx)
    X_test_s = _scale(test_idx)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Z-SCORE TARGET: consistente con el LSTM que vendrá después
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-12
    y_train_z = (y_train - mean_y) / std_y
    y_val_z = (y_val - mean_y) / std_y

    # 6. ENTRENAR RIDGE (alpha desde config, default 100.0 para intradía)
    baseline_cfg = e3_cfg.get("baseline", {})
    ridge_alpha = float(baseline_cfg.get("alpha", 100.0))
    print(f"\nEntrenando Ridge Regression (alpha={ridge_alpha})...")
    model = IntradayRidgeBaseline(alpha=ridge_alpha, seed=42)
    train_result = model.fit(X_train_s, y_train_z, X_val_s, y_val_z)
    print(f"Train R²={train_result.train_score:.4f}, Val R²={train_result.val_score:.4f}")
    print(f"Features efectivas (post-flatten): {train_result.n_features_effective}")

    # 7. PREDICCIONES EN TEST (des-escalar a retornos originales)
    y_pred_z = model.predict(X_test_s)
    y_pred_test = y_pred_z * std_y + mean_y

    # 8. MÉTRICAS ML
    test_metrics = {
        "mae": mae(y_test, y_pred_test),
        "rmse": rmse(y_test, y_pred_test),
        "directional_accuracy": directional_accuracy(y_test, y_pred_test),
        "ic": information_coefficient(y_test, y_pred_test),
    }
    print(
        f"\nMétricas en test:"
        f"\n  MAE={test_metrics['mae']:.6f}"
        f"\n  RMSE={test_metrics['rmse']:.6f}"
        f"\n  Directional Accuracy={test_metrics['directional_accuracy']:.3f}"
        f"\n  IC={test_metrics['ic']:.3f}"
    )

    # Guardar predicciones
    test_timestamps = timestamps[test_idx]
    pred_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "y_true": y_test,
        "y_pred": y_pred_test,
    })
    pred_df.to_csv(out_dir / f"{ticker}_baseline_predictions.csv", index=False)

    # 9. BACKTEST INTRADIARIO
    thresholds = e3_cfg.get("thresholds", {})
    tau_buy = float(thresholds.get("tau_buy", 0.001))
    tau_sell = float(thresholds.get("tau_sell", 0.001))
    round_trip_bps = float(config.get("costs", {}).get("intraday_round_trip_bps", 20.0))
    backtest_cfg = e3_cfg.get("backtest", {})
    allow_short = bool(backtest_cfg.get("allow_short", True))
    execution_delay = int(backtest_cfg.get("execution_delay_bars", 1))
    max_position = float(backtest_cfg.get("max_position", 1.0))

    print(
        f"\nEjecutando backtest (tau_buy={tau_buy}, tau_sell={tau_sell}, "
        f"allow_short={allow_short}, costs={round_trip_bps}bps)..."
    )

    test_timestamps_idx = pd.DatetimeIndex(pred_df["timestamp"].values)
    bar_returns_test = bar_ret_series.reindex(test_timestamps_idx).fillna(0).to_numpy(dtype=float)

    backtest_df = backtest_intraday_signals(
        timestamps=test_timestamps_idx,
        bar_returns=bar_returns_test,
        pred_forward_returns=y_pred_test,
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        execution_delay_bars=execution_delay,
        allow_short=allow_short,
        max_position=max_position,
    )

    backtest_df.to_csv(out_dir / f"{ticker}_baseline_backtest.csv")
    bt_summary = summarize_backtest_intraday(backtest_df)

    print(f"\nBacktest Summary:")
    print(f"  Total Return:  {bt_summary['total_return']:.2%}")
    print(f"  Sharpe:        {bt_summary['sharpe']:.3f}")
    print(f"  Max DD:        {bt_summary['max_drawdown']:.2%}")
    print(f"  Profit Factor: {bt_summary['profit_factor']:.2f}")
    print(f"  Time in Mkt:   {bt_summary['time_in_market']:.1%}")

    # 10. SUMMARY COMPLETO
    duration = round(time.perf_counter() - start_time, 2)
    summary = {
        "ticker": ticker,
        "model": "Ridge_Baseline",
        "train_r2": train_result.train_score,
        "val_r2": train_result.val_score,
        **{f"ml_{k}": v for k, v in test_metrics.items()},
        **{f"bt_{k}": v for k, v in bt_summary.items()},
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "lookback_bars": lookback_bars,
        "horizon_bars": horizon_bars,
        "n_features": len(feature_names),
        "n_features_effective": train_result.n_features_effective,
        "ridge_alpha": ridge_alpha,
        "timing_train_seconds": duration,
    }

    print(f"\nDuración total: {duration:.2f}s")

    pd.DataFrame([summary]).to_csv(out_dir / f"{ticker}_baseline_summary.csv", index=False)

    # 11. LIFECYCLE REGISTRY
    try:
        from ..lifecycle.registry import ModelRegistry
        from ..lifecycle.guardrails import log_candidate_metrics

        root = project_root()
        registry_path = root / "models" / "registry.json"

        log_candidate_metrics(
            metrics=summary,
            strategy="e3",
            ticker=ticker,
            run_dir=out_dir,
            variant="e3_baseline",
            log_path=root / "models" / "metrics_log.jsonl",
        )

        registry = ModelRegistry(registry_path)
        registry.register_baseline(
            strategy="e3",
            ticker=ticker,
            run_dir=str(out_dir.relative_to(root)),
            metrics=summary,
            variant="e3_baseline",
        )
        print(f"✓ Registered {ticker} as e3 baseline in lifecycle registry")
    except Exception as exc:
        print(f"⚠️  Lifecycle registration skipped: {exc}")

    # 12. MLFLOW TRACKING
    if mlflow_enabled and mlflow is not None:
        try:
            run_name = f"E3Baseline_{ticker}_{timestamp}" if timestamp else f"E3Baseline_{ticker}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("strategy", "e3_baseline")
                mlflow.log_param("ticker", ticker)
                mlflow.log_param("model_type", "Ridge")
                if timestamp:
                    mlflow.log_param("timestamp", timestamp)

                mlflow.log_params({
                    "lookback_bars": lookback_bars,
                    "horizon_bars": horizon_bars,
                    "n_features": len(feature_names),
                    "n_features_effective": train_result.n_features_effective,
                    "ridge_alpha": ridge_alpha,
                    "tau_buy": tau_buy,
                    "tau_sell": tau_sell,
                    "round_trip_bps": round_trip_bps,
                    "allow_short": allow_short,
                    "train_frac": train_frac,
                    "val_frac": val_frac,
                })

                metrics_to_log = {
                    k: float(v)
                    for k, v in summary.items()
                    if k not in {"ticker", "model"}
                    and isinstance(v, (int, float, np.number))
                }
                mlflow.log_metrics(metrics_to_log)

                for artifact_path in [
                    out_dir / f"{ticker}_baseline_predictions.csv",
                    out_dir / f"{ticker}_baseline_backtest.csv",
                    out_dir / f"{ticker}_baseline_summary.csv",
                ]:
                    if artifact_path.exists():
                        mlflow.log_artifact(str(artifact_path))

            print(f"✓ Run guardado en MLflow: {run_name}")
        except Exception as exc:
            print(f"⚠️  Error guardando en MLflow: {exc}")

    return summary


def _setup_mlflow(root: Path, experiment_name: str) -> tuple[bool, object | None]:
    """Intenta activar MLflow (remoto → local SQLite fallback)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    timeout = float(os.getenv("MLFLOW_REMOTE_CHECK_TIMEOUT_SECONDS", "1.5"))
    local_dir = root / "runs" / "mlflow_local"
    ensure_dir(local_dir)
    local_db = local_dir / "mlflow.db"
    local_artifacts = local_dir / "artifacts"
    ensure_dir(local_artifacts)
    local_uri = os.getenv("MLFLOW_LOCAL_TRACKING_URI", "").strip() or f"sqlite:///{local_db}"

    def _reachable(uri: str) -> bool:
        from urllib.parse import urlparse
        parsed = urlparse(uri)
        if parsed.scheme not in {"http", "https"}:
            return True
        host = parsed.hostname
        if not host:
            return True
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    try:
        import mlflow as _mlflow

        if tracking_uri and _reachable(tracking_uri):
            try:
                _mlflow.set_tracking_uri(tracking_uri)
                _mlflow.set_experiment(experiment_name)
                print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
                return True, _mlflow
            except Exception as exc:
                print(f"⚠️  MLflow remoto falló: {exc}")

        try:
            _mlflow.set_tracking_uri(local_uri)
            if _mlflow.get_experiment_by_name(experiment_name) is None:
                _mlflow.create_experiment(
                    experiment_name,
                    artifact_location=local_artifacts.resolve().as_uri(),
                )
            _mlflow.set_experiment(experiment_name)
            label = "local fallback" if tracking_uri else "tracking local"
            print(f"✓ MLflow habilitado ({label}): {local_db}")
            return True, _mlflow
        except Exception as exc:
            print(f"⚠️  MLflow local falló: {exc}")

    except Exception as exc:
        print(f"⚠️  MLflow no disponible: {exc}")

    return False, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3 Baseline - Ridge Regression (datos 5-min intradiarios)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Tickers separados por coma (ej: SPY,AAPL). Si se omite usa universe.tickers_by_strategy.e3_intraday.",
    )
    parser.add_argument(
        "--config",
        default="src/config/base.yaml",
        help="Archivo de configuración YAML",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/intraday",
        help="Directorio con datos 5-min raw",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directorio de salida (auto: runs/e3_baseline/<timestamp>)",
    )
    parser.add_argument(
        "--use-latest-data",
        action="store_true",
        help=(
            "Entrenar con el rango extendido hasta hoy "
            "(ignora data.training_window.intraday.end del config; start se preserva). "
            "Nota: este script no descarga datos; usá src.e3.intraday_data aparte."
        ),
    )
    args = parser.parse_args()

    root = project_root()
    config = load_yaml(root / args.config)

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = (
            config.get("universe", {})
            .get("tickers_by_strategy", {})
            .get("e3_intraday", [])
        )

    if not tickers:
        raise ValueError(
            "No se especificaron tickers. Usa --tickers SPY o define "
            "universe.tickers_by_strategy.e3_intraday en base.yaml."
        )

    print(f"Entrenando baseline E3 para {len(tickers)} ticker(s): {', '.join(tickers)}")

    raw_dir = root / args.raw_dir
    ensure_dir(raw_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = Path(args.output_dir) if args.output_dir else (
        root / "runs" / "e3_baseline" / timestamp
    )
    ensure_dir(out_base)

    import yaml
    with open(out_base / "config_used.yaml", "w") as f:
        yaml.dump(config, f)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E3_Baseline")
    mlflow_enabled, mlflow = _setup_mlflow(root, experiment_name)

    # Entrenar un baseline por ticker
    summaries: list[dict] = []
    for ticker in tickers:
        try:
            summary = run_baseline_for_ticker(
                config=config,
                ticker=ticker,
                raw_dir=raw_dir,
                out_dir=out_base / ticker,
                mlflow_enabled=mlflow_enabled,
                mlflow=mlflow,
                timestamp=timestamp,
                use_latest_data=args.use_latest_data,
            )
            summaries.append(summary)
            print(f"✓ {ticker} completado\n")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}\n")

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(out_base / "baseline_summary_all.csv", index=False)
        df_summary.to_csv(out_base / "summary_all.csv", index=False)

        print(f"\n{'='*60}")
        print("RESULTADOS BASELINE E3")
        print(f"{'='*60}")
        cols = ["ticker", "ml_ic", "ml_directional_accuracy", "bt_sharpe",
                "bt_max_drawdown", "bt_profit_factor"]
        print(df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))
        print(f"\n✓ Resultados guardados en: {out_base}")


if __name__ == "__main__":
    main()

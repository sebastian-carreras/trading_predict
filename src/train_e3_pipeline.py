from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.intraday import (
    backtest_intraday_signals,
    compute_max_drawdown,
    compute_profit_factor,
)
from .data.intraday_yfinance import download_ohlcv_5m, load_ohlcv_csv
from .features.intraday import compute_intraday_features, make_sequences, make_target_return
from .models.e3_lstm import LSTMRegressor
from .reporting.intraday_metrics import directional_accuracy, information_coefficient, mae, rmse
from .utils import ensure_dir, get_nested, load_yaml, project_root


def _time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("Invalid split fractions")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, n)
    return idx_train, idx_val, idx_test


def run_for_ticker(config: dict, ticker: str, raw_dir: Path, out_dir: Path) -> dict:
    print(f"  Loading config for {ticker}...")
    e3 = get_nested(config, ["strategies", "e3_intraday"], {})
    
    print(f"  Extracting parameters...")
    lookback_bars = int(e3.get("lookback_bars", 96))
    horizon_bars = int(e3.get("horizon_bars", 6))

    thresholds = e3.get("thresholds", {})
    print(f"  Thresholds type: {type(thresholds)}, value: {thresholds}")
    tau_buy = float(thresholds.get("tau_buy", 0.001))
    tau_sell = float(thresholds.get("tau_sell", 0.001))

    costs = config.get("costs", {})
    round_trip_bps = float(costs.get("intraday_round_trip_bps", 20))

    model_cfg = e3.get("model", {})
    ensemble_members = int(model_cfg.get("ensemble_members", 3))
    hidden_size = int(model_cfg.get("lstm_hidden_size", 64))
    num_layers = int(model_cfg.get("lstm_num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    lr = float(model_cfg.get("learning_rate", 1e-3))
    batch_size = int(model_cfg.get("batch_size", 256))
    max_epochs = int(model_cfg.get("max_epochs", 30))
    patience = int(model_cfg.get("early_stopping_patience", 5))
    loss = str(model_cfg.get("loss", "huber"))
    huber_delta = float(model_cfg.get("huber_delta", 1.0))

    bt_cfg = e3.get("backtest", {})
    execution_delay_bars = int(bt_cfg.get("execution_delay_bars", 1))
    allow_short = bool(bt_cfg.get("allow_short", True))
    max_position = float(bt_cfg.get("max_position", 1.0))

    csv_path = raw_dir / f"{ticker}_5m.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing intraday CSV: {csv_path}")

    ohlcv = load_ohlcv_csv(csv_path)
    features = compute_intraday_features(ohlcv)
    target = make_target_return(ohlcv, horizon_bars=horizon_bars)

    X, y, ts, feat_names = make_sequences(features, target, lookback_bars=lookback_bars)

    idx_train, idx_val, idx_test = _time_split(len(X))
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]
    ts_test = ts[idx_test]

    # Standardize features (fit on train only)
    # Flatten time dimension for scaling
    Xtr2d = X_train.reshape(-1, X_train.shape[-1])
    mean = Xtr2d.mean(axis=0)
    std = Xtr2d.std(axis=0) + 1e-12

    def scale(Xa: np.ndarray) -> np.ndarray:
        return ((Xa - mean) / std).astype(np.float32)

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)

    preds_members: list[np.ndarray] = []
    val_losses: list[float] = []

    base_seed = int(get_nested(config, ["project", "seed"], 42))

    print(f"\nTraining ensemble of {ensemble_members} LSTM models for {ticker}...")
    for m in range(ensemble_members):
        print(f"\n  Model {m+1}/{ensemble_members} (seed={base_seed + 1000 * m}):")
        seed = base_seed + 1000 * m
        model = LSTMRegressor(
            input_size=X_train_s.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            seed=seed,
        )
        res = model.fit(
            X_train_s,
            y_train,
            X_val_s,
            y_val,
            learning_rate=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=patience,
            loss=loss,
            huber_delta=huber_delta,
            verbose=True,
        )
        val_losses.append(res.best_val_loss)
        preds_members.append(model.predict(X_test_s))
        print(f"  Model {m+1} final: val_loss={res.best_val_loss:.6f}, epochs={res.epochs_ran}")

    print(f"\nEnsemble training complete. Averaging {len(preds_members)} predictions...")

    y_pred = np.mean(np.stack(preds_members, axis=0), axis=0)

    # ML metrics
    ml = {
        "mae": mae(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "ic": information_coefficient(y_test, y_pred),
        "directional_accuracy": directional_accuracy(y_test, y_pred),
        "val_loss_mean": float(np.mean(val_losses)),
    }

    # Backtest on test period
    # 1-bar log return aligned to the same timestamps index
    close_series = ohlcv["close"].astype("float64")
    log_close = pd.Series(np.log(close_series.to_numpy()), index=close_series.index)
    bar_ret = log_close.diff().reindex(ts_test).fillna(0.0).to_numpy(dtype=np.float32)

    bt = backtest_intraday_signals(
        timestamps=ts_test,
        bar_returns=bar_ret,
        pred_forward_returns=y_pred.astype(np.float32),
        tau_buy=tau_buy,
        tau_sell=tau_sell,
        round_trip_bps=round_trip_bps,
        execution_delay_bars=execution_delay_bars,
        allow_short=allow_short,
        max_position=max_position,
    )

    trading = {
        "profit_factor": compute_profit_factor(bt["net_ret"].to_numpy(dtype=np.float32)),
        "max_drawdown": compute_max_drawdown(bt["equity"].to_numpy(dtype=np.float32)),
        "turnover_mean": float(bt["turnover"].mean()),
        "time_in_market": float((bt["pos"].abs() > 0).mean()),
    }

    # Persist artifacts
    ensure_dir(out_dir)
    bt.to_csv(out_dir / f"{ticker}_backtest.csv")

    preds_df = pd.DataFrame(
        {"y_true": y_test, "y_pred": y_pred},
        index=ts_test,
    )
    preds_df.to_csv(out_dir / f"{ticker}_predictions.csv")

    meta = {
        "ticker": ticker,
        "n_samples": int(len(X)),
        "n_test": int(len(X_test)),
        "lookback_bars": lookback_bars,
        "horizon_bars": horizon_bars,
        "tau_buy": tau_buy,
        "tau_sell": tau_sell,
        "round_trip_bps": round_trip_bps,
        "ensemble_members": ensemble_members,
        "feature_count": int(len(feat_names)),
    }

    summary = {**meta, **{f"ml_{k}": v for k, v in ml.items()}, **{f"tr_{k}": v for k, v in trading.items()}}
    (pd.Series(summary)).to_csv(out_dir / f"{ticker}_summary.csv")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 intraday pipeline (5m, LSTM ensemble)")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path to YAML config (relative to trading_predict root is allowed)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["download", "run"],
        default="run",
        help="download: fetch CSVs; run: train+backtest using existing CSVs",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers override (otherwise uses universe.tickers_by_strategy.e3_intraday)",
    )
    args = parser.parse_args()

    root = project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    config = load_yaml(cfg_path)

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        tickers = list(get_nested(config, ["universe", "tickers_by_strategy", "e3_intraday"], []))

    if not tickers:
        raise ValueError("No tickers provided and config has no universe.tickers_by_strategy.e3_intraday")

    e3 = get_nested(config, ["strategies", "e3_intraday"], {})
    data_cfg = e3.get("data", {})

    raw_dir = root / "data" / "raw" / "intraday"
    out_base = root / "runs" / "e3_intraday" / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_base)

    if args.mode == "download":
        period = str(data_cfg.get("period", "60d"))
        interval = str(data_cfg.get("interval", "5m"))
        written = download_ohlcv_5m(tickers, out_dir=raw_dir, period=period, interval=interval)
        print(f"Downloaded {len(written)} files to {raw_dir}")
        return

    # MLflow setup (optional)
    mlflow_enabled = False
    mlflow = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "E3_Intraday")
    if tracking_uri:
        try:
            import mlflow as _mlflow  # type: ignore

            _mlflow.set_tracking_uri(tracking_uri)
            _mlflow.set_experiment(experiment_name)
            mlflow = _mlflow
            mlflow_enabled = True
            print(f"✓ MLflow habilitado: {tracking_uri} (experiment={experiment_name})")
        except Exception as exc:
            print(f"⚠️  MLflow no disponible, continuando sin tracking: {exc}")
            mlflow_enabled = False

    summaries: list[dict] = []
    for ticker in tickers:
        ticker_out = out_base / ticker
        try:
            if mlflow_enabled and mlflow is not None:
                timestamp = out_base.name
                with mlflow.start_run(run_name=f"E3_{ticker}_{timestamp}"):
                    mlflow.log_param("strategy", "e3_intraday")
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("model_type", "LSTM_Ensemble")
                    mlflow.log_param("timestamp", timestamp)

                    model_cfg = e3.get("model", {})
                    mlflow.log_params(
                        {
                            "lookback_bars": int(e3.get("lookback_bars", 96)),
                            "horizon_bars": int(e3.get("horizon_bars", 6)),
                            "ensemble_members": int(model_cfg.get("ensemble_members", 3)),
                            "lstm_hidden_size": int(model_cfg.get("lstm_hidden_size", 64)),
                            "lstm_num_layers": int(model_cfg.get("lstm_num_layers", 2)),
                            "dropout": float(model_cfg.get("dropout", 0.2)),
                            "learning_rate": float(model_cfg.get("learning_rate", 1e-3)),
                            "batch_size": int(model_cfg.get("batch_size", 256)),
                            "max_epochs": int(model_cfg.get("max_epochs", 30)),
                            "early_stopping_patience": int(model_cfg.get("early_stopping_patience", 5)),
                            "loss": str(model_cfg.get("loss", "huber")),
                            "huber_delta": float(model_cfg.get("huber_delta", 1.0)),
                            "tau_buy": float(thresholds.get("tau_buy", 0.001)),
                            "tau_sell": float(thresholds.get("tau_sell", 0.001)),
                            "round_trip_bps": float(costs.get("intraday_round_trip_bps", 20)),
                            "seed": int(config.get("project", {}).get("seed", 42)),
                        }
                    )

                    summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out)

                    metrics: dict[str, float] = {}
                    for k, v in summary.items():
                        if not (k.startswith("ml_") or k.startswith("tr_")):
                            continue
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
                    if metrics:
                        mlflow.log_metrics(metrics)

                    # Log artifacts
                    for fname in [
                        f"{ticker}_predictions.csv",
                        f"{ticker}_backtest.csv",
                        f"{ticker}_summary.csv",
                    ]:
                        p = ticker_out / fname
                        if p.exists():
                            if "pred" in fname:
                                artifact_path = "predictions"
                            elif "backtest" in fname:
                                artifact_path = "backtest"
                            else:
                                artifact_path = "artifacts"
                            mlflow.log_artifact(str(p), artifact_path=artifact_path)
            else:
                summary = run_for_ticker(config, ticker=ticker, raw_dir=raw_dir, out_dir=ticker_out)
            summaries.append(summary)
            print(f"Done {ticker}: PF={summary['tr_profit_factor']:.3f} DD={summary['tr_max_drawdown']:.3%} IC={summary['ml_ic']:.3f}")
        except Exception as exc:
            print(f"✗ Error en {ticker}: {exc}")

    pd.DataFrame(summaries).to_csv(out_base / "summary_all.csv", index=False)
    print(f"Wrote run outputs to {out_base}")


if __name__ == "__main__":
    main()

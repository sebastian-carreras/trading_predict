#!/usr/bin/env python3
"""
Script para comparar el impacto de la normalización de targets.
Compara runs antes y después de normalizar targets (y).

Uso:
    python scripts/compare_normalization_impact.py \
        --run_old runs/e1_conservative/20260106_213335 \
        --run_new runs/e1_conservative/<timestamp_nuevo>
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def analyze_predictions(predictions_csv: Path) -> dict:
    """Analiza archivo de predicciones y retorna estadísticas."""
    df = pd.read_csv(predictions_csv)
    
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    
    bias = y_pred.mean() - y_true.mean()
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # Señales de trading
    buy_signals = (y_pred > 0.06).sum()
    sell_signals = (y_pred < 0.00).sum()
    
    return {
        "y_true_mean": y_true.mean(),
        "y_pred_mean": y_pred.mean(),
        "y_pred_std": y_pred.std(),
        "y_pred_min": y_pred.min(),
        "y_pred_max": y_pred.max(),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "ic": ic,
        "dir_acc": dir_acc,
        "n_samples": len(df),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "buy_pct": buy_signals / len(df) * 100,
    }


def compare_runs(run_old: Path, run_new: Path) -> pd.DataFrame:
    """Compara dos runs y retorna DataFrame con métricas."""
    
    # Encontrar todos los tickers
    summaries_old = sorted(run_old.glob("*_summary.csv"))
    tickers = [s.stem.replace("_summary", "") for s in summaries_old]
    
    results = []
    
    for ticker in tickers:
        # Archivos old run
        pred_old = run_old / f"{ticker}_predictions.csv"
        summ_old = run_old / f"{ticker}_summary.csv"
        
        # Archivos new run (puede estar en subdirectorio)
        pred_new = run_new / ticker / f"{ticker}_predictions.csv"
        summ_new = run_new / ticker / f"{ticker}_summary.csv"
        
        if not pred_new.exists():
            pred_new = run_new / f"{ticker}_predictions.csv"
            summ_new = run_new / f"{ticker}_summary.csv"
        
        if not pred_old.exists() or not pred_new.exists():
            print(f"⚠️  Saltando {ticker} (archivos no encontrados)")
            continue
        
        # Analizar predicciones
        stats_old = analyze_predictions(pred_old)
        stats_new = analyze_predictions(pred_new)
        
        # Cargar summaries para backtesting
        df_old = pd.read_csv(summ_old, index_col=0)
        df_new = pd.read_csv(summ_new, index_col=0)
        
        results.append({
            "ticker": ticker,
            # Bias
            "bias_old": stats_old["bias"],
            "bias_new": stats_new["bias"],
            "bias_improvement": stats_old["bias"] - stats_new["bias"],
            # IC
            "ic_old": stats_old["ic"],
            "ic_new": stats_new["ic"],
            "ic_delta": stats_new["ic"] - stats_old["ic"],
            # Directional Accuracy
            "dir_acc_old": stats_old["dir_acc"],
            "dir_acc_new": stats_new["dir_acc"],
            "dir_acc_delta": stats_new["dir_acc"] - stats_old["dir_acc"],
            # MAE
            "mae_old": stats_old["mae"],
            "mae_new": stats_new["mae"],
            "mae_delta": stats_new["mae"] - stats_old["mae"],
            # Señales BUY
            "buy_signals_old": stats_old["buy_signals"],
            "buy_signals_new": stats_new["buy_signals"],
            "buy_pct_old": stats_old["buy_pct"],
            "buy_pct_new": stats_new["buy_pct"],
            # Backtesting
            "bt_trades_old": int(df_old.loc["bt_num_trades", "0"]),
            "bt_trades_new": int(df_new.loc["bt_num_trades", "0"]),
            "bt_sharpe_old": float(df_old.loc["bt_sharpe", "0"]),
            "bt_sharpe_new": float(df_new.loc["bt_sharpe", "0"]),
        })
    
    return pd.DataFrame(results)


def print_comparison(df: pd.DataFrame) -> None:
    """Imprime comparación formateada."""
    
    print("\n" + "="*80)
    print("COMPARACIÓN: ANTES vs DESPUÉS DE NORMALIZAR TARGETS")
    print("="*80)
    
    print(f"\nTotal tickers analizados: {len(df)}")
    
    # Resumen de bias
    print("\n📊 BIAS (predicción - realidad):")
    print(f"  Bias promedio ANTES: {df['bias_old'].mean():+.4f} ({df['bias_old'].mean()*100:+.2f}%)")
    print(f"  Bias promedio DESPUÉS: {df['bias_new'].mean():+.4f} ({df['bias_new'].mean()*100:+.2f}%)")
    print(f"  Mejora promedio: {df['bias_improvement'].mean():.4f} ({df['bias_improvement'].mean()*100:.2f}%)")
    
    # IC
    print("\n📈 INFORMATION COEFFICIENT (IC):")
    print(f"  IC promedio ANTES: {df['ic_old'].mean():.4f}")
    print(f"  IC promedio DESPUÉS: {df['ic_new'].mean():.4f}")
    print(f"  Delta promedio: {df['ic_delta'].mean():+.4f}")
    
    # Directional Accuracy
    print("\n🎯 DIRECTIONAL ACCURACY:")
    print(f"  Dir Acc promedio ANTES: {df['dir_acc_old'].mean():.4f} ({df['dir_acc_old'].mean()*100:.1f}%)")
    print(f"  Dir Acc promedio DESPUÉS: {df['dir_acc_new'].mean():.4f} ({df['dir_acc_new'].mean()*100:.1f}%)")
    print(f"  Delta promedio: {df['dir_acc_delta'].mean():+.4f} ({df['dir_acc_delta'].mean()*100:+.1f}%)")
    
    # Señales BUY
    print("\n💼 SEÑALES DE TRADING (tau_buy=0.06):")
    print(f"  Señales BUY promedio ANTES: {df['buy_signals_old'].mean():.1f} ({df['buy_pct_old'].mean():.1f}%)")
    print(f"  Señales BUY promedio DESPUÉS: {df['buy_signals_new'].mean():.1f} ({df['buy_pct_new'].mean():.1f}%)")
    
    # Backtesting
    print("\n📊 BACKTESTING:")
    tickers_with_trades_old = (df['bt_trades_old'] > 0).sum()
    tickers_with_trades_new = (df['bt_trades_new'] > 0).sum()
    print(f"  Tickers con trades ANTES: {tickers_with_trades_old}/{len(df)}")
    print(f"  Tickers con trades DESPUÉS: {tickers_with_trades_new}/{len(df)}")
    
    if tickers_with_trades_old > 0:
        df_with_trades_old = df[df['bt_trades_old'] > 0]
        print(f"  Sharpe promedio ANTES: {df_with_trades_old['bt_sharpe_old'].mean():.4f}")
    
    if tickers_with_trades_new > 0:
        df_with_trades_new = df[df['bt_trades_new'] > 0]
        print(f"  Sharpe promedio DESPUÉS: {df_with_trades_new['bt_sharpe_new'].mean():.4f}")
    
    # Top mejoras
    print("\n🏆 TOP 5 MEJORAS EN IC:")
    top_ic = df.nlargest(5, 'ic_delta')[['ticker', 'ic_old', 'ic_new', 'ic_delta']]
    print(top_ic.to_string(index=False))
    
    print("\n🏆 TOP 5 MEJORAS EN DIRECTIONAL ACCURACY:")
    top_dir = df.nlargest(5, 'dir_acc_delta')[['ticker', 'dir_acc_old', 'dir_acc_new', 'dir_acc_delta']]
    print(top_dir.to_string(index=False))
    
    print("\n🏆 TOP 5 REDUCCIONES EN BIAS (más cercano a cero):")
    df['bias_abs_improvement'] = df['bias_old'].abs() - df['bias_new'].abs()
    top_bias = df.nlargest(5, 'bias_abs_improvement')[['ticker', 'bias_old', 'bias_new', 'bias_improvement']]
    print(top_bias.to_string(index=False))
    
    # Detalle por ticker
    print("\n" + "="*80)
    print("DETALLE POR TICKER")
    print("="*80)
    
    for _, row in df.iterrows():
        print(f"\n📊 {row['ticker']}")
        print(f"  Bias: {row['bias_old']:+.4f} → {row['bias_new']:+.4f} (Δ {row['bias_improvement']:+.4f})")
        print(f"  IC: {row['ic_old']:.4f} → {row['ic_new']:.4f} (Δ {row['ic_delta']:+.4f})")
        print(f"  Dir Acc: {row['dir_acc_old']:.3f} → {row['dir_acc_new']:.3f} (Δ {row['dir_acc_delta']:+.3f})")
        print(f"  BUY signals: {row['buy_signals_old']:.0f} → {row['buy_signals_new']:.0f}")
        print(f"  BT trades: {row['bt_trades_old']:.0f} → {row['bt_trades_new']:.0f}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparar impacto de normalización de targets"
    )
    parser.add_argument(
        "--run_old",
        type=str,
        required=True,
        help="Path al run ANTES de normalizar (e.g., runs/e1_conservative/20260106_213335)"
    )
    parser.add_argument(
        "--run_new",
        type=str,
        required=True,
        help="Path al run DESPUÉS de normalizar (e.g., runs/e1_conservative/20260106_220000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path para guardar CSV con comparación (opcional)"
    )
    
    args = parser.parse_args()
    
    run_old = Path(args.run_old)
    run_new = Path(args.run_new)
    
    if not run_old.exists():
        print(f"❌ Run old no encontrado: {run_old}")
        return
    
    if not run_new.exists():
        print(f"❌ Run new no encontrado: {run_new}")
        return
    
    # Comparar
    df_comparison = compare_runs(run_old, run_new)
    
    # Imprimir
    print_comparison(df_comparison)
    
    # Guardar si se especificó output
    if args.output:
        output_path = Path(args.output)
        df_comparison.to_csv(output_path, index=False)
        print(f"\n✅ Comparación guardada en: {output_path}")


if __name__ == "__main__":
    main()

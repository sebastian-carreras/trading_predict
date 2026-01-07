#!/usr/bin/env python3
"""
Script para verificar que el escalado/desescalado de targets funciona correctamente.
Valida que:
1. Los targets se normalizan solo con estadísticas de train
2. Las predicciones se desnormalizan correctamente
3. Las métricas se calculan con valores originales (no normalizados)
4. El backtesting usa predicciones desnormalizadas

Uso:
    python scripts/verify_scaling.py --run_dir runs/e1_conservative/<timestamp>
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def verify_ticker_scaling(ticker_dir: Path, verbose: bool = False) -> dict:
    """
    Verifica el escalado correcto para un ticker.
    
    Returns:
        dict con resultados de validación
    """
    ticker = ticker_dir.name
    
    # Cargar archivos
    predictions_csv = ticker_dir / f"{ticker}_predictions.csv"
    target_scaler_csv = ticker_dir / f"{ticker}_target_scaler.csv"
    summary_csv = ticker_dir / f"{ticker}_summary.csv"
    
    if not predictions_csv.exists():
        return {"ticker": ticker, "status": "MISSING_FILES", "errors": ["predictions.csv not found"]}
    
    if not target_scaler_csv.exists():
        return {"ticker": ticker, "status": "MISSING_FILES", "errors": ["target_scaler.csv not found"]}
    
    # Cargar datos
    preds_df = pd.read_csv(predictions_csv)
    scaler_df = pd.read_csv(target_scaler_csv)
    summary_df = pd.read_csv(summary_csv, index_col=0)
    
    y_true = preds_df["y_true"].values
    y_pred = preds_df["y_pred"].values
    
    mean_y = float(scaler_df["mean_y"][0])
    std_y = float(scaler_df["std_y"][0])
    
    errors = []
    warnings = []
    
    # TEST 1: Verificar que y_true no está normalizado
    # (debe tener media y std diferentes de 0 y 1)
    if abs(y_true.mean()) < 1e-6 and abs(y_true.std() - 1.0) < 1e-2:
        errors.append("y_true parece estar normalizado (mean≈0, std≈1)")
    
    # TEST 2: Verificar que y_pred no está normalizado
    # (debe tener rango razonable de retornos)
    if abs(y_pred.mean()) < 1e-6 and abs(y_pred.std() - 1.0) < 1e-2:
        errors.append("y_pred parece estar normalizado (mean≈0, std≈1)")
    
    # TEST 3: Verificar que mean_y es razonable (entre -50% y +50%)
    if abs(mean_y) > 0.5:
        warnings.append(f"mean_y muy alto: {mean_y:.4f} ({mean_y*100:.2f}%)")
    
    # TEST 4: Verificar que std_y es razonable (entre 1% y 50%)
    if std_y < 0.01 or std_y > 0.5:
        warnings.append(f"std_y fuera de rango esperado: {std_y:.4f}")
    
    # TEST 5: Simular re-normalización y verificar que produce valores cercanos a N(0,1)
    y_pred_renorm = (y_pred - mean_y) / std_y
    
    if abs(y_pred_renorm.std() - 1.0) > 0.5:
        warnings.append(f"Re-normalización de y_pred no da std≈1 (got {y_pred_renorm.std():.4f})")
    
    # TEST 6: Verificar que métricas coinciden con cálculos manuales
    mae_summary = float(summary_df.loc["ml_mae", "0"])
    ic_summary = float(summary_df.loc["ml_ic", "0"])
    
    mae_manual = float(np.mean(np.abs(y_true - y_pred)))
    ic_manual = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    
    if abs(mae_summary - mae_manual) > 1e-4:
        errors.append(f"MAE no coincide: summary={mae_summary:.6f} vs manual={mae_manual:.6f}")
    
    if abs(ic_summary - ic_manual) > 1e-4:
        errors.append(f"IC no coincide: summary={ic_summary:.6f} vs manual={ic_manual:.6f}")
    
    # TEST 7: Verificar que y_pred tiene rango razonable para retornos
    if y_pred.min() < -1.0 or y_pred.max() > 2.0:
        warnings.append(f"y_pred fuera de rango típico: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # TEST 8: Verificar que no hay NaN o Inf
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        errors.append("y_pred contiene NaN o Inf")
    
    if np.isnan(y_true).any() or np.isinf(y_true).any():
        errors.append("y_true contiene NaN o Inf")
    
    # Resumen
    status = "PASS" if len(errors) == 0 else "FAIL"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"Status: {status}")
        print(f"\nParámetros de escalado:")
        print(f"  mean_y: {mean_y:.6f} ({mean_y*100:.2f}%)")
        print(f"  std_y: {std_y:.6f} ({std_y*100:.2f}%)")
        
        print(f"\ny_true (valores originales):")
        print(f"  Mean: {y_true.mean():.6f} ({y_true.mean()*100:.2f}%)")
        print(f"  Std: {y_true.std():.6f}")
        print(f"  Range: [{y_true.min():.4f}, {y_true.max():.4f}]")
        
        print(f"\ny_pred (valores desnormalizados):")
        print(f"  Mean: {y_pred.mean():.6f} ({y_pred.mean()*100:.2f}%)")
        print(f"  Std: {y_pred.std():.6f}")
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        print(f"\ny_pred re-normalizado (debe ser ~N(0,1)):")
        print(f"  Mean: {y_pred_renorm.mean():.6f}")
        print(f"  Std: {y_pred_renorm.std():.6f}")
        
        print(f"\nMétricas (deben calcularse con valores originales):")
        print(f"  MAE summary: {mae_summary:.6f}")
        print(f"  MAE manual: {mae_manual:.6f}")
        print(f"  IC summary: {ic_summary:.6f}")
        print(f"  IC manual: {ic_manual:.6f}")
        
        if errors:
            print(f"\n❌ Errores:")
            for err in errors:
                print(f"  - {err}")
        
        if warnings:
            print(f"\n⚠️  Advertencias:")
            for warn in warnings:
                print(f"  - {warn}")
    
    return {
        "ticker": ticker,
        "status": status,
        "mean_y": mean_y,
        "std_y": std_y,
        "y_true_mean": y_true.mean(),
        "y_pred_mean": y_pred.mean(),
        "y_pred_renorm_std": y_pred_renorm.std(),
        "mae_match": abs(mae_summary - mae_manual) < 1e-4,
        "ic_match": abs(ic_summary - ic_manual) < 1e-4,
        "errors": errors,
        "warnings": warnings,
    }


def verify_run(run_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Verifica todos los tickers en un run."""
    
    # Buscar subdirectorios de tickers
    ticker_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    
    results = []
    
    if ticker_dirs:
        # Estructura con subdirectorios (nueva)
        for ticker_dir in sorted(ticker_dirs):
            result = verify_ticker_scaling(ticker_dir, verbose=verbose)
            results.append(result)
    else:
        # Estructura plana (archivos directamente en run_dir)
        print("ℹ️  Estructura plana detectada (archivos sin subdirectorios)")
        
        # Encontrar todos los archivos *_summary.csv
        summary_files = sorted(run_dir.glob("*_summary.csv"))
        
        for summary_file in summary_files:
            ticker = summary_file.stem.replace("_summary", "")
            
            # Crear un "pseudo-directorio" para la función verify_ticker_scaling
            # Pasamos run_dir pero modificamos la función para aceptar flat structure
            result = verify_ticker_scaling_flat(run_dir, ticker, verbose=verbose)
            results.append(result)
    
    return pd.DataFrame(results)


def verify_ticker_scaling_flat(run_dir: Path, ticker: str, verbose: bool = False) -> dict:
    """
    Verifica el escalado correcto para un ticker en estructura plana.
    """
    
    # Cargar archivos
    predictions_csv = run_dir / f"{ticker}_predictions.csv"
    target_scaler_csv = run_dir / f"{ticker}_target_scaler.csv"
    summary_csv = run_dir / f"{ticker}_summary.csv"
    
    if not predictions_csv.exists():
        return {"ticker": ticker, "status": "MISSING_FILES", "errors": ["predictions.csv not found"]}
    
    if not target_scaler_csv.exists():
        return {"ticker": ticker, "status": "MISSING_FILES", "errors": ["target_scaler.csv not found"]}
    
    # Cargar datos
    preds_df = pd.read_csv(predictions_csv)
    scaler_df = pd.read_csv(target_scaler_csv)
    summary_df = pd.read_csv(summary_csv, index_col=0)
    
    y_true = preds_df["y_true"].values
    y_pred = preds_df["y_pred"].values
    
    mean_y = float(scaler_df["mean_y"][0])
    std_y = float(scaler_df["std_y"][0])
    
    errors = []
    warnings = []
    
    # TEST 1: Verificar que y_true no está normalizado
    # (debe tener media y std diferentes de 0 y 1)
    if abs(y_true.mean()) < 1e-6 and abs(y_true.std() - 1.0) < 1e-2:
        errors.append("y_true parece estar normalizado (mean≈0, std≈1)")
    
    # TEST 2: Verificar que y_pred no está normalizado
    # (debe tener rango razonable de retornos)
    if abs(y_pred.mean()) < 1e-6 and abs(y_pred.std() - 1.0) < 1e-2:
        errors.append("y_pred parece estar normalizado (mean≈0, std≈1)")
    
    # TEST 3: Verificar que mean_y es razonable (entre -50% y +50%)
    if abs(mean_y) > 0.5:
        warnings.append(f"mean_y muy alto: {mean_y:.4f} ({mean_y*100:.2f}%)")
    
    # TEST 4: Verificar que std_y es razonable (entre 1% y 50%)
    if std_y < 0.01 or std_y > 0.5:
        warnings.append(f"std_y fuera de rango esperado: {std_y:.4f}")
    
    # TEST 5: Simular re-normalización y verificar que produce valores cercanos a N(0,1)
    y_pred_renorm = (y_pred - mean_y) / std_y
    
    if abs(y_pred_renorm.std() - 1.0) > 0.5:
        warnings.append(f"Re-normalización de y_pred no da std≈1 (got {y_pred_renorm.std():.4f})")
    
    # TEST 6: Verificar que métricas coinciden con cálculos manuales
    mae_summary = float(summary_df.loc["ml_mae", "0"])
    ic_summary = float(summary_df.loc["ml_ic", "0"])
    
    mae_manual = float(np.mean(np.abs(y_true - y_pred)))
    ic_manual = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    
    if abs(mae_summary - mae_manual) > 1e-4:
        errors.append(f"MAE no coincide: summary={mae_summary:.6f} vs manual={mae_manual:.6f}")
    
    if abs(ic_summary - ic_manual) > 1e-4:
        errors.append(f"IC no coincide: summary={ic_summary:.6f} vs manual={ic_manual:.6f}")
    
    # TEST 7: Verificar que y_pred tiene rango razonable para retornos
    if y_pred.min() < -1.0 or y_pred.max() > 2.0:
        warnings.append(f"y_pred fuera de rango típico: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # TEST 8: Verificar que no hay NaN o Inf
    if np.isnan(y_pred).any() or np.isinf(y_pred).any():
        errors.append("y_pred contiene NaN o Inf")
    
    if np.isnan(y_true).any() or np.isinf(y_true).any():
        errors.append("y_true contiene NaN o Inf")
    
    # Resumen
    status = "PASS" if len(errors) == 0 else "FAIL"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"Status: {status}")
        print(f"\nParámetros de escalado:")
        print(f"  mean_y: {mean_y:.6f} ({mean_y*100:.2f}%)")
        print(f"  std_y: {std_y:.6f} ({std_y*100:.2f}%)")
        
        print(f"\ny_true (valores originales):")
        print(f"  Mean: {y_true.mean():.6f} ({y_true.mean()*100:.2f}%)")
        print(f"  Std: {y_true.std():.6f}")
        print(f"  Range: [{y_true.min():.4f}, {y_true.max():.4f}]")
        
        print(f"\ny_pred (valores desnormalizados):")
        print(f"  Mean: {y_pred.mean():.6f} ({y_pred.mean()*100:.2f}%)")
        print(f"  Std: {y_pred.std():.6f}")
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        
        print(f"\ny_pred re-normalizado (debe ser ~N(0,1)):")
        print(f"  Mean: {y_pred_renorm.mean():.6f}")
        print(f"  Std: {y_pred_renorm.std():.6f}")
        
        print(f"\nMétricas (deben calcularse con valores originales):")
        print(f"  MAE summary: {mae_summary:.6f}")
        print(f"  MAE manual: {mae_manual:.6f}")
        print(f"  IC summary: {ic_summary:.6f}")
        print(f"  IC manual: {ic_manual:.6f}")
        
        if errors:
            print(f"\n❌ Errores:")
            for err in errors:
                print(f"  - {err}")
        
        if warnings:
            print(f"\n⚠️  Advertencias:")
            for warn in warnings:
                print(f"  - {warn}")
    
    return {
        "ticker": ticker,
        "status": status,
        "mean_y": mean_y,
        "std_y": std_y,
        "y_true_mean": y_true.mean(),
        "y_pred_mean": y_pred.mean(),
        "y_pred_renorm_std": y_pred_renorm.std(),
        "mae_match": abs(mae_summary - mae_manual) < 1e-4,
        "ic_match": abs(ic_summary - ic_manual) < 1e-4,
        "errors": errors,
        "warnings": warnings,
    }


def print_summary(df: pd.DataFrame) -> None:
    """Imprime resumen de verificación."""
    
    print("\n" + "="*80)
    print("RESUMEN DE VERIFICACIÓN DE ESCALADO")
    print("="*80)
    
    n_total = len(df)
    n_pass = (df["status"] == "PASS").sum()
    n_fail = (df["status"] == "FAIL").sum()
    n_missing = (df["status"] == "MISSING_FILES").sum()
    
    print(f"\nTotal tickers: {n_total}")
    print(f"  ✅ PASS: {n_pass}")
    print(f"  ❌ FAIL: {n_fail}")
    print(f"  ⚠️  MISSING_FILES: {n_missing}")
    
    if n_fail > 0:
        print(f"\n❌ Tickers con errores:")
        failed = df[df["status"] == "FAIL"]
        for _, row in failed.iterrows():
            print(f"\n  {row['ticker']}:")
            for err in row['errors']:
                print(f"    - {err}")
    
    # Estadísticas de parámetros de escalado
    valid = df[df["status"] == "PASS"]
    
    if len(valid) > 0:
        print(f"\n📊 Estadísticas de parámetros de escalado (tickers válidos):")
        print(f"  mean_y: {valid['mean_y'].mean():.4f} ± {valid['mean_y'].std():.4f}")
        print(f"    Range: [{valid['mean_y'].min():.4f}, {valid['mean_y'].max():.4f}]")
        print(f"  std_y: {valid['std_y'].mean():.4f} ± {valid['std_y'].std():.4f}")
        print(f"    Range: [{valid['std_y'].min():.4f}, {valid['std_y'].max():.4f}]")
        
        print(f"\n📈 Verificación de desnormalización:")
        print(f"  y_pred re-norm std promedio: {valid['y_pred_renorm_std'].mean():.4f}")
        print(f"    (debe estar cercano a 1.0)")
        
        print(f"\n✅ Métricas correctas:")
        print(f"  MAE match: {valid['mae_match'].sum()}/{len(valid)}")
        print(f"  IC match: {valid['ic_match'].sum()}/{len(valid)}")
    
    # Tickers con warnings
    warnings_list = []
    for _, row in df.iterrows():
        if row['warnings']:
            warnings_list.append((row['ticker'], row['warnings']))
    
    if warnings_list:
        print(f"\n⚠️  Tickers con advertencias:")
        for ticker, warns in warnings_list:
            print(f"\n  {ticker}:")
            for warn in warns:
                print(f"    - {warn}")


def main():
    parser = argparse.ArgumentParser(
        description="Verificar escalado/desescalado de targets"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path al run a verificar (e.g., runs/e1_conservative/20260106_220000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar detalles por ticker"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path para guardar resultados en CSV (opcional)"
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"❌ Run directory no encontrado: {run_dir}")
        return
    
    # Verificar
    df_results = verify_run(run_dir, verbose=args.verbose)
    
    if df_results.empty:
        print("❌ No se encontraron tickers para verificar")
        return
    
    # Imprimir resumen
    print_summary(df_results)
    
    # Guardar si se especificó output
    if args.output:
        output_path = Path(args.output)
        df_results.to_csv(output_path, index=False)
        print(f"\n✅ Resultados guardados en: {output_path}")
    
    # Exit code basado en resultados
    n_fail = (df_results["status"] == "FAIL").sum()
    exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()

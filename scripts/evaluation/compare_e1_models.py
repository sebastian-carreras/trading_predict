"""
Script de comparación entre modelos E1: GRU vs Regresión Lineal Baseline.

Compara las siguientes métricas:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directional Accuracy
- IC (Information Coefficient - Spearman)
- Métricas de trading (Sharpe, CAGR, Max Drawdown, etc.)

Uso:
    python scripts/compare_e1_models.py
    python scripts/compare_e1_models.py --gru-run runs/e1_conservative/20260115_120000
    python scripts/compare_e1_models.py --baseline-run runs/e1_baseline/20260115_130000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np


def load_latest_run(runs_dir: Path) -> Path | None:
    """Encuentra el run más reciente en un directorio."""
    if not runs_dir.exists():
        return None
    
    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    return runs[-1] if runs else None


def load_model_summary(run_dir: Path, model_name: str) -> pd.DataFrame | None:
    """Carga el summary de un modelo."""
    # Buscar archivo summary
    summary_files = [
        run_dir / "summary_all.csv",
        run_dir / f"{model_name}_summary_all.csv",
        run_dir / "baseline_summary_all.csv",
    ]
    
    for summary_file in summary_files:
        if summary_file.exists():
            return pd.read_csv(summary_file)
    
    return None


def compute_comparison_metrics(df_baseline: pd.DataFrame, df_gru: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de comparación entre baseline y GRU.
    
    Returns:
        DataFrame con comparación lado a lado
    """
    # Métricas a comparar
    metrics_config = [
        # ML Metrics
        ("MAE", "ml_mae", "lower_better", "Menor error absoluto promedio"),
        ("RMSE", "ml_rmse", "lower_better", "Menor error cuadrático medio"),
        ("Dir. Accuracy", "ml_directional_accuracy", "higher_better", "% predicciones correctas de dirección"),
        ("IC (Spearman)", "ml_ic", "higher_better", "Correlación Spearman (>0.05 significativo)"),
        
        # Trading Metrics
        ("Sharpe Ratio", "bt_sharpe", "higher_better", "Retorno ajustado por riesgo"),
        ("Sortino Ratio", "bt_sortino", "higher_better", "Retorno ajustado por downside risk"),
        ("CAGR", "bt_cagr", "higher_better", "Retorno anualizado compuesto"),
        ("Max Drawdown", "bt_max_drawdown", "lower_better", "Peor caída desde máximo"),
        ("Calmar Ratio", "bt_calmar", "higher_better", "CAGR / Max Drawdown"),
        ("Profit Factor", "bt_profit_factor", "higher_better", "Ganancias brutas / Pérdidas brutas"),
        ("Hit Rate", "bt_hit_rate", "higher_better", "% operaciones ganadoras"),
    ]
    
    comparison_data = []
    
    for metric_name, col_name, direction, description in metrics_config:
        baseline_val = df_baseline[col_name].mean() if col_name in df_baseline.columns else np.nan
        gru_val = df_gru[col_name].mean() if col_name in df_gru.columns else np.nan
        
        diff = gru_val - baseline_val
        
        # Determinar si GRU es mejor
        if direction == "higher_better":
            gru_better = diff > 0
            improvement_pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
        else:  # lower_better
            gru_better = diff < 0
            improvement_pct = (-diff / baseline_val * 100) if baseline_val != 0 else 0
        
        comparison_data.append({
            "Métrica": metric_name,
            "Baseline": baseline_val,
            "GRU": gru_val,
            "Diferencia": diff,
            "Mejora %": improvement_pct,
            "GRU Mejor": "✓" if gru_better else "✗",
            "Descripción": description,
        })
    
    return pd.DataFrame(comparison_data)


def print_comparison_table(df_comparison: pd.DataFrame) -> None:
    """Imprime tabla de comparación formateada."""
    print("\n" + "="*100)
    print("COMPARACIÓN: Regresión Lineal Baseline vs GRU")
    print("="*100)
    
    # Separar métricas ML y Trading
    ml_metrics = ["MAE", "RMSE", "Dir. Accuracy", "IC (Spearman)"]
    
    print("\n📊 MÉTRICAS ML (Offline)")
    print("-"*100)
    df_ml = df_comparison[df_comparison["Métrica"].isin(ml_metrics)].copy()
    print_formatted_table(df_ml)
    
    print("\n💰 MÉTRICAS TRADING (Online - Backtest)")
    print("-"*100)
    df_trading = df_comparison[~df_comparison["Métrica"].isin(ml_metrics)].copy()
    print_formatted_table(df_trading)
    
    # Resumen
    print("\n" + "="*100)
    print("RESUMEN")
    print("="*100)
    
    total_metrics = len(df_comparison)
    gru_wins = (df_comparison["GRU Mejor"] == "✓").sum()
    baseline_wins = total_metrics - gru_wins
    
    print(f"Total métricas comparadas: {total_metrics}")
    print(f"GRU superior en: {gru_wins} métricas ({gru_wins/total_metrics*100:.1f}%)")
    print(f"Baseline superior en: {baseline_wins} métricas ({baseline_wins/total_metrics*100:.1f}%)")
    
    # Mejoras más significativas
    print("\n Top 3 mejoras del GRU:")
    top_improvements = df_comparison.nlargest(3, "Mejora %")[["Métrica", "Mejora %", "GRU Mejor"]]
    for idx, row in top_improvements.iterrows():
        status = "✓" if row["GRU Mejor"] == "✓" else "✗"
        print(f"  {status} {row['Métrica']}: {row['Mejora %']:+.2f}%")


def print_formatted_table(df: pd.DataFrame) -> None:
    """Imprime DataFrame con formato mejorado."""
    for idx, row in df.iterrows():
        metric = row["Métrica"]
        baseline = row["Baseline"]
        gru = row["GRU"]
        diff = row["Diferencia"]
        mejora = row["Mejora %"]
        mejor = row["GRU Mejor"]
        
        print(f"{metric:<20} | Base: {baseline:>10.4f} | GRU: {gru:>10.4f} | "
              f"Δ: {diff:>+10.4f} | {mejora:>+7.2f}% {mejor}")


def save_comparison_report(df_comparison: pd.DataFrame, output_path: Path) -> None:
    """Guarda reporte de comparación."""
    # CSV
    df_comparison.to_csv(output_path.with_suffix(".csv"), index=False)
    
    # Markdown
    with open(output_path.with_suffix(".md"), "w") as f:
        f.write("# Comparación E1: Baseline vs GRU\n\n")
        f.write("## Métricas ML\n\n")
        
        ml_metrics = ["MAE", "RMSE", "Dir. Accuracy", "IC (Spearman)"]
        df_ml = df_comparison[df_comparison["Métrica"].isin(ml_metrics)]
        f.write(df_ml.to_markdown(index=False))
        
        f.write("\n\n## Métricas Trading\n\n")
        df_trading = df_comparison[~df_comparison["Métrica"].isin(ml_metrics)]
        f.write(df_trading.to_markdown(index=False))
        
        # Resumen
        total = len(df_comparison)
        gru_wins = (df_comparison["GRU Mejor"] == "✓").sum()
        
        f.write(f"\n\n## Resumen\n\n")
        f.write(f"- **Total métricas**: {total}\n")
        f.write(f"- **GRU superior**: {gru_wins} ({gru_wins/total*100:.1f}%)\n")
        f.write(f"- **Baseline superior**: {total - gru_wins} ({(total-gru_wins)/total*100:.1f}%)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara modelos E1: Baseline vs GRU"
    )
    parser.add_argument(
        "--baseline-run",
        type=str,
        default=None,
        help="Path al run del baseline (auto-detect si None)",
    )
    parser.add_argument(
        "--gru-run",
        type=str,
        default=None,
        help="Path al run del GRU (auto-detect si None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/e1_model_comparison",
        help="Path de salida para el reporte",
    )
    
    args = parser.parse_args()
    
    # Detectar project root
    current_file = Path(__file__).resolve()
    # Asumir que estamos en scripts/ o src/
    if current_file.parent.name == "scripts":
        root = current_file.parent.parent
    else:
        root = Path.cwd()
    
    # Auto-detectar runs si no se especifican
    if args.baseline_run is None:
        baseline_runs_dir = root / "runs" / "e1_baseline"
        baseline_run_path = load_latest_run(baseline_runs_dir)
        if baseline_run_path is None:
            print("❌ No se encontraron runs del baseline en runs/e1_baseline/")
            print("   Ejecuta primero: python -m src.train_e1_baseline --tickers AAPL")
            sys.exit(1)
    else:
        baseline_run_path = Path(args.baseline_run)
    
    if args.gru_run is None:
        gru_runs_dir = root / "runs" / "e1_conservative"
        gru_run_path = load_latest_run(gru_runs_dir)
        if gru_run_path is None:
            print("❌ No se encontraron runs del GRU en runs/e1_conservative/")
            print("   Ejecuta primero: python -m src.train_e1_pipeline --tickers AAPL")
            sys.exit(1)
    else:
        gru_run_path = Path(args.gru_run)
    
    print(f"\n📁 Cargando resultados...")
    print(f"   Baseline: {baseline_run_path}")
    print(f"   GRU:      {gru_run_path}")
    
    # Cargar summaries
    df_baseline = load_model_summary(baseline_run_path, "baseline")
    df_gru = load_model_summary(gru_run_path, "gru")
    
    if df_baseline is None:
        print(f"❌ No se encontró summary del baseline en {baseline_run_path}")
        sys.exit(1)
    
    if df_gru is None:
        print(f"❌ No se encontró summary del GRU en {gru_run_path}")
        sys.exit(1)
    
    print(f"   ✓ Baseline: {len(df_baseline)} tickers")
    print(f"   ✓ GRU: {len(df_gru)} tickers")
    
    # Comparar
    df_comparison = compute_comparison_metrics(df_baseline, df_gru)
    
    # Mostrar resultados
    print_comparison_table(df_comparison)
    
    # Guardar reporte
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_comparison_report(df_comparison, output_path)
    
    print(f"\n✅ Reporte guardado en:")
    print(f"   📄 {output_path}.csv")
    print(f"   📄 {output_path}.md")


if __name__ == "__main__":
    main()

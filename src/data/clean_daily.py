"""
Limpieza y validación de datos OHLCV después de descarga.

Funciones:
- Detectar valores nulos/vacíos por feature
- Reportar estadísticas de calidad de datos
- Aplicar limpieza básica (forward fill, interpolación, o eliminación)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def diagnose_data_quality(df: pd.DataFrame, ticker: str) -> dict:
    """Diagnostica calidad de datos OHLCV.

    Args:
        df: DataFrame con OHLCV
        ticker: Nombre del ticker (para reporte)

    Returns:
        Dict con estadísticas de calidad
    """
    report = {
        "ticker": ticker,
        "total_rows": len(df),
        "null_counts": {},
        "null_percentages": {},
        "features_with_nulls": [],
        "zero_volume_days": 0,
        "duplicate_timestamps": 0,
        "data_gaps_days": [],
    }

    # 1. Conteo de nulos por columna
    for col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            report["null_counts"][col] = int(null_count)
            report["null_percentages"][col] = round(100 * null_count / len(df), 2)
            report["features_with_nulls"].append(col)

    # 2. Días con volumen cero (posible problema de datos)
    if "volume" in df.columns:
        report["zero_volume_days"] = int((df["volume"] == 0).sum())

    # 3. Timestamps duplicados
    if "timestamp" in df.columns:
        report["duplicate_timestamps"] = int(df["timestamp"].duplicated().sum())

    # 4. Gaps en serie temporal (días faltantes vs días de mercado esperados)
    if "timestamp" in df.columns and len(df) > 1:
        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff()
        # Gaps > 4 días (considerando fines de semana + feriados normales)
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=4)]
        if len(large_gaps) > 0:
            report["data_gaps_days"] = [
                {
                    "date": str(df_sorted.loc[idx, "timestamp"]),
                    "gap_days": int(gap.days)
                }
                for idx, gap in large_gaps.items()
            ]

    return report


def clean_ohlcv_data(
    df: pd.DataFrame,
    strategy: Literal["forward_fill", "interpolate", "drop"] = "forward_fill",
    max_consecutive_nulls: int = 5,
    remove_zero_volume: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Limpia datos OHLCV aplicando estrategia seleccionada.

    Args:
        df: DataFrame con OHLCV
        strategy: Estrategia de limpieza
            - "forward_fill": Propagar último valor válido (conservador, recomendado)
            - "interpolate": Interpolar linealmente (útil para gaps pequeños)
            - "drop": Eliminar filas con nulos (puede perder muchos datos)
        max_consecutive_nulls: Máximo de nulos consecutivos permitidos antes de marcar como problemático
        remove_zero_volume: Si True, elimina días con volumen=0 (recomendado, evita ruido)
        verbose: Si True, imprime advertencias

    Returns:
        DataFrame limpio
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Eliminar duplicados de timestamp
    if "timestamp" in df_clean.columns:
        duplicates = df_clean["timestamp"].duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=["timestamp"], keep="first")
            if verbose:
                print(f"  ⚠️  Eliminados {duplicates} timestamps duplicados")

    # 2. Ordenar por timestamp
    if "timestamp" in df_clean.columns:
        df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    # 3. Eliminar días con volumen=0 (sin trading real)
    if remove_zero_volume and "volume" in df_clean.columns:
        zero_volume_mask = df_clean["volume"] == 0
        zero_volume_count = zero_volume_mask.sum()
        
        if zero_volume_count > 0:
            df_clean = df_clean[~zero_volume_mask].reset_index(drop=True)
            if verbose:
                pct_removed = 100 * zero_volume_count / initial_rows
                print(f"  🗑️  Eliminados {zero_volume_count} días con volumen=0 ({pct_removed:.1f}%) - sin trading real")

    # 4. Detectar columnas con nulos
    null_cols = [col for col in df_clean.columns if df_clean[col].isna().any()]

    if not null_cols:
        if verbose and not remove_zero_volume:
            print(f"  ✓ Sin valores nulos detectados")
        return df_clean

    # 5. Aplicar estrategia de limpieza
    for col in null_cols:
        null_count = df_clean[col].isna().sum()
        null_pct = 100 * null_count / len(df_clean)

        if verbose:
            print(f"  🔧 {col}: {null_count} nulos ({null_pct:.1f}%)", end=" → ")

        # Verificar nulos consecutivos
        null_mask = df_clean[col].isna()
        consecutive_nulls = null_mask.astype(int).groupby((~null_mask).cumsum()).sum()
        max_consecutive = consecutive_nulls.max() if len(consecutive_nulls) > 0 else 0

        if max_consecutive > max_consecutive_nulls:
            if verbose:
                print(f"⚠️  {max_consecutive} nulos consecutivos (> {max_consecutive_nulls} límite)")

        # Aplicar estrategia
        if strategy == "forward_fill":
            df_clean[col] = df_clean[col].fillna(method="ffill")
            # Si quedan nulos al inicio (no hay valor previo), usar backward fill
            df_clean[col] = df_clean[col].fillna(method="bfill")
            if verbose:
                remaining = df_clean[col].isna().sum()
                if remaining == 0:
                    print("✓ Forward fill")
                else:
                    print(f"⚠️  {remaining} nulos restantes después de forward fill")

        elif strategy == "interpolate":
            df_clean[col] = df_clean[col].interpolate(method="linear")
            # Nulos al inicio/final (no se pueden interpolar)
            df_clean[col] = df_clean[col].fillna(method="ffill").fillna(method="bfill")
            if verbose:
                print("✓ Interpolación lineal")

        elif strategy == "drop":
            # No hacemos nada aquí, se eliminan filas al final
            if verbose:
                print("Marcado para eliminación")

    # 6. Si strategy="drop", eliminar filas con cualquier nulo
    if strategy == "drop":
        df_clean = df_clean.dropna()
        dropped = initial_rows - len(df_clean)
        if verbose and dropped > 0:
            print(f"  ⚠️  Eliminadas {dropped} filas con nulos ({100*dropped/initial_rows:.1f}%)")

    # 7. Validación final
    remaining_nulls = df_clean.isna().sum().sum()
    if remaining_nulls > 0 and verbose:
        print(f"  ⚠️  ADVERTENCIA: {remaining_nulls} nulos restantes después de limpieza")
        print(f"     Columnas afectadas: {[c for c in df_clean.columns if df_clean[c].isna().any()]}")

    final_rows = len(df_clean)
    if verbose and final_rows < initial_rows:
        print(f"  📉 Filas: {initial_rows} → {final_rows} ({100*(initial_rows-final_rows)/initial_rows:.1f}% reducción)")

    return df_clean


def process_daily_data_with_cleaning(
    raw_dir: Path,
    clean_dir: Path,
    strategy: Literal["forward_fill", "interpolate", "drop"] = "forward_fill",
    min_days: int = 252,
    remove_zero_volume: bool = True,
    verbose: bool = True,
    tickers: Iterable[str] | None = None,
) -> dict[str, dict]:
    """Procesa todos los archivos CSV del directorio raw aplicando limpieza.

    Args:
        raw_dir: Directorio con CSVs raw
        clean_dir: Directorio de salida para CSVs limpios
        strategy: Estrategia de limpieza
        min_days: Mínimo de días requeridos después de limpieza (252 = 1 año trading)
        remove_zero_volume: Si True, elimina días con volumen=0
        verbose: Si True, imprime reporte detallado

    Returns:
        Dict con reportes de calidad por ticker
    """
    from ..utils import ensure_dir

    ensure_dir(clean_dir)
    csv_files = list(raw_dir.glob("*_daily.csv"))

    if tickers:
        ticker_set = {t.strip() for t in tickers if t and t.strip()}
        csv_files = [
            path for path in csv_files
            if path.stem.replace("_daily", "") in ticker_set
        ]

    if not csv_files:
        if tickers:
            print(f"⚠️  No se encontraron CSV para los tickers solicitados en {raw_dir}")
        else:
            print(f"⚠️  No se encontraron archivos CSV en {raw_dir}")
        return {}

    reports = {}

    print(f"\n{'='*80}")
    print(f"LIMPIEZA DE DATOS - Estrategia: {strategy.upper()}")
    print(f"{'='*80}\n")

    for csv_path in csv_files:
        ticker = csv_path.stem.replace("_daily", "")

        if verbose:
            print(f"📊 {ticker}")

        # Cargar datos
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)

        # Diagnóstico pre-limpieza
        report = diagnose_data_quality(df, ticker)
        reports[ticker] = report

        # Mostrar problemas encontrados
        if verbose:
            if report["features_with_nulls"]:
                print(f"  ⚠️  Nulos detectados en: {', '.join(report['features_with_nulls'])}")
                for feat in report["features_with_nulls"]:
                    print(f"     - {feat}: {report['null_counts'][feat]} ({report['null_percentages'][feat]}%)")
            
            if report["zero_volume_days"] > 0:
                print(f"  ⚠️  {report['zero_volume_days']} días con volumen=0")
            
            if report["duplicate_timestamps"] > 0:
                print(f"  ⚠️  {report['duplicate_timestamps']} timestamps duplicados")
            
            if report["data_gaps_days"]:
                print(f"  ⚠️  {len(report['data_gaps_days'])} gaps grandes en serie temporal (>{4} días)")

        # Aplicar limpieza
        df_clean = clean_ohlcv_data(
            df, 
            strategy=strategy, 
            remove_zero_volume=remove_zero_volume,
            verbose=verbose
        )

        # Validación post-limpieza
        if len(df_clean) < min_days:
            print(f"  ❌ {ticker} descartado: solo {len(df_clean)} días después de limpieza (< {min_days} requerido)")
            reports[ticker]["status"] = "rejected"
            continue

        # Guardar datos limpios
        out_path = clean_dir / f"{ticker}_daily.csv"
        df_clean.to_csv(out_path, index=False)
        reports[ticker]["status"] = "cleaned"
        reports[ticker]["rows_after_cleaning"] = len(df_clean)

        if verbose:
            print(f"  ✓ Guardado: {out_path.name} ({len(df_clean)} días)\n")

    # Resumen global
    print(f"\n{'='*80}")
    print("RESUMEN DE LIMPIEZA")
    print(f"{'='*80}")
    
    total_tickers = len(reports)
    cleaned = sum(1 for r in reports.values() if r.get("status") == "cleaned")
    rejected = sum(1 for r in reports.values() if r.get("status") == "rejected")
    
    print(f"Total tickers procesados: {total_tickers}")
    print(f"  ✓ Limpiados: {cleaned}")
    print(f"  ❌ Rechazados: {rejected}")
    
    # Tickers con más problemas
    tickers_with_issues = [
        (ticker, len(r["features_with_nulls"])) 
        for ticker, r in reports.items() 
        if r["features_with_nulls"]
    ]
    if tickers_with_issues:
        tickers_with_issues.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTickers con más features problemáticas:")
        for ticker, num_features in tickers_with_issues[:5]:
            print(f"  - {ticker}: {num_features} features con nulos")

    print(f"\n✓ Datos limpios guardados en: {clean_dir}\n")

    return reports


def main() -> None:
    """Entrypoint: limpia datos descargados.

    Uso:
        python -m src.data.clean_daily                    # Forward fill (recomendado)
        python -m src.data.clean_daily --interpolate      # Interpolación lineal
        python -m src.data.clean_daily --drop             # Eliminar filas con nulos
    """
    import sys
    from ..utils import project_root

    root = project_root()
    raw_dir = root / "data" / "raw" / "daily"
    clean_dir = root / "data" / "clean"

    # Determinar estrategia desde argumentos
    if "--interpolate" in sys.argv:
        strategy = "interpolate"
    elif "--drop" in sys.argv:
        strategy = "drop"
    else:
        strategy = "forward_fill"

    print(f"Directorio raw: {raw_dir}")
    print(f"Directorio clean: {clean_dir}")
    print(f"Estrategia: {strategy}\n")

    reports = process_daily_data_with_cleaning(
        raw_dir=raw_dir,
        clean_dir=clean_dir,
        strategy=strategy,
        min_days=252,
        verbose=True,
    )

    # Guardar reporte JSON
    import json
    report_path = clean_dir / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    
    print(f"📄 Reporte de calidad guardado en: {report_path}")


if __name__ == "__main__":
    main()

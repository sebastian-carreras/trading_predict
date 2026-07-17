"""Monitoreo de drift de features: PSI + KS de dos muestras.

Compara la distribución de referencia (con la que se entrenó/promovió el
champion) contra la distribución actual (datos recientes en producción) para
detectar **data drift**. Dos señales complementarias por feature:

- **PSI (Population Stability Index):** cuánto se movió la masa de la
  distribución entre bins definidos por la referencia. Umbrales de industria:
  < 0.10 estable · 0.10–0.25 drift moderado · >= 0.25 drift significativo.
- **KS (Kolmogorov-Smirnov, dos muestras):** máxima distancia entre las CDF
  empíricas + p-value. p < 0.05 → las distribuciones difieren.

El reporte se puede loguear a JSONL (misma convención que
``guardrails.log_candidate_metrics``) para calibración e historización.

Nota: esto detecta *data drift* (cambia la distribución de las features). El
*concept drift* (cambia la relación feature→target) se monitorea mirando las
métricas en vivo del backtest/registry, no acá.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Umbrales estándar de industria para PSI.
PSI_STABLE = 0.10  # < esto: sin drift relevante
PSI_MODERATE = 0.25  # entre PSI_STABLE y esto: drift moderado; >=: significativo

# Nivel de significancia para el test KS.
KS_ALPHA = 0.05


@dataclass
class FeatureDrift:
    """Resultado de drift para una feature."""

    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    severity: str  # "estable" | "moderado" | "significativo"
    n_ref: int
    n_cur: int

    @property
    def drifted(self) -> bool:
        """True si PSI o KS marcan drift (moderado/significativo o KS < alpha)."""
        return self.severity != "estable" or self.ks_pvalue < KS_ALPHA


def _clean(values: Iterable[float]) -> np.ndarray:
    """A array 1-D float64, descartando NaN/Inf."""
    arr = np.asarray(list(values), dtype="float64").ravel()
    return arr[np.isfinite(arr)]


def population_stability_index(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI entre una distribución de referencia y una actual (feature única).

    Los bordes de bin se definen con **cuantiles de la referencia** (robusto a
    features sesgadas/financieras). PSI = sum((cur% - ref%) * ln(cur%/ref%)).

    Args:
        reference: valores de referencia (train/promoción del champion).
        current: valores actuales (producción).
        bins: cantidad de bins cuantílicos (default 10).
        epsilon: piso para proporciones vacías (evita ln(0) / división por 0).

    Returns:
        PSI >= 0. 0.0 si no hay datos suficientes o la referencia es constante.
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size == 0 or cur.size == 0:
        return 0.0

    # Bordes por cuantiles de la referencia; -inf/+inf en las colas para
    # capturar valores actuales fuera del rango histórico.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # Referencia (casi) constante: no hay distribución que comparar.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_frac = np.clip(ref_counts / ref.size, epsilon, None)
    cur_frac = np.clip(cur_counts / cur.size, epsilon, None)

    psi = np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac))
    return float(psi)


def _classify(psi: float) -> str:
    if psi < PSI_STABLE:
        return "estable"
    if psi < PSI_MODERATE:
        return "moderado"
    return "significativo"


def feature_drift(
    feature: str,
    reference: Iterable[float],
    current: Iterable[float],
    *,
    bins: int = 10,
) -> FeatureDrift:
    """Calcula PSI + KS para una feature y clasifica la severidad."""
    ref = _clean(reference)
    cur = _clean(current)

    psi = population_stability_index(ref, cur, bins=bins)

    if ref.size >= 2 and cur.size >= 2:
        ks = ks_2samp(ref, cur)
        ks_stat, ks_pvalue = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_pvalue = 0.0, 1.0

    return FeatureDrift(
        feature=feature,
        psi=round(psi, 6),
        ks_stat=round(ks_stat, 6),
        ks_pvalue=round(ks_pvalue, 6),
        severity=_classify(psi),
        n_ref=int(ref.size),
        n_cur=int(cur.size),
    )


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    bins: int = 10,
) -> list[FeatureDrift]:
    """Reporte de drift feature-por-feature entre dos DataFrames.

    Args:
        reference: features de referencia (una columna por feature).
        current: features actuales (mismas columnas).
        feature_cols: subconjunto a evaluar; por defecto la intersección
            numérica de columnas de ambos DataFrames.
        bins: bins cuantílicos para PSI.

    Returns:
        Lista de FeatureDrift, ordenada de mayor a menor PSI.
    """
    if feature_cols is None:
        common = [c for c in reference.columns if c in current.columns]
        feature_cols = [
            c for c in common if pd.api.types.is_numeric_dtype(reference[c])
        ]

    results = [
        feature_drift(col, reference[col], current[col], bins=bins)
        for col in feature_cols
    ]
    results.sort(key=lambda r: r.psi, reverse=True)
    return results


def summarize(report: list[FeatureDrift]) -> dict[str, Any]:
    """Resumen accionable de un reporte: conteos y máximos."""
    drifted = [r for r in report if r.drifted]
    worst = max(report, key=lambda r: r.psi) if report else None
    return {
        "n_features": len(report),
        "n_drifted": len(drifted),
        "n_significativo": sum(r.severity == "significativo" for r in report),
        "n_moderado": sum(r.severity == "moderado" for r in report),
        "worst_feature": worst.feature if worst else None,
        "worst_psi": worst.psi if worst else 0.0,
        "drifted_features": [r.feature for r in drifted],
    }


def log_drift_report(
    report: list[FeatureDrift],
    strategy: str,
    ticker: str,
    *,
    reference_tag: str,
    log_path: str | Path | None = None,
) -> Path:
    """Agrega una entrada al JSONL de drift (una fila por corrida de monitoreo).

    Convención de path por defecto: ``models/drift_log.jsonl`` en la raíz del
    repo (igual criterio que ``guardrails.metrics_log.jsonl``).

    Returns:
        Path al archivo de log.
    """
    if log_path is None:
        # src/lifecycle/drift.py -> parents[2] == raíz del repo
        log_path = Path(__file__).resolve().parents[2] / "models" / "drift_log.jsonl"

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "ticker": ticker,
        "reference_tag": reference_tag,
        **summarize(report),
        "features": [asdict(r) for r in report],
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    return log_path

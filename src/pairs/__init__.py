"""
Módulo de pairs trading para estrategia E4.

Componentes:
- select_pairs: Selección de pares cointegrados
- build_spread: Construcción de spreads y cálculo de z-scores
- ou_process: Modelado de procesos Ornstein-Uhlenbeck
- knn_confirm: Confirmación k-NN (opcional)
"""

from .select_pairs import find_cointegrated_pairs, test_pair_cointegration
from .build_spread import build_spread, calculate_hedge_ratio, calculate_zscore
from .ou_process import estimate_ou_parameters, calculate_half_life

__all__ = [
    "find_cointegrated_pairs",
    "test_pair_cointegration",
    "build_spread",
    "calculate_hedge_ratio",
    "calculate_zscore",
    "estimate_ou_parameters",
    "calculate_half_life",
]

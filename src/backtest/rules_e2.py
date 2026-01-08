"""
Reglas de trading para E2 (Estrategia Moderada).

Lógica de entrada/salida:
- Entrada:
    - Predicción del modelo > tau_buy (2.5%)
    - Filtros (opcionales):
        - RSI entre 35 y 70
        - MACD histograma > 0 (momentum positivo)
        - Volume z-score > 0 (participación)
- Salida:
    - Predicción del modelo < tau_sell (0.00%)
    - Tocar Stop Loss o Take Profit
    - Time stop (20 días)

Esta lógica se aplica día a día en el backtest.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def apply_e2_rules(
    current_date: pd.Timestamp,
    model_pred: float,
    current_nav: float,
    position: dict | None,
    market_data: pd.Series,
    config: dict,
) -> dict | None:
    """Aplica reglas E2 para determinar una acción de trading.

    Args:
        current_date: Fecha actual
        model_pred: Retorno predicho para H=20 días
        current_nav: Valor actual del portafolio (cash + holdings)
        position: Diccionario con posición actual o None
                  {'ticker': str, 'entry_price': float, 'shares': float, 'entry_date': ts}
        market_data: Series con features del día (precio, RSI, MACD...)
        config: Configuración de estrategia (thresholds, filtros)

    Returns:
        Acción a tomar:
        - {'action': 'BUY', 'size': float}  (size en $)
        - {'action': 'SELL', 'reason': str}
        - None (hold/wait)
    """
    thresholds = config.get("thresholds", {})
    filters = config.get("filters", {})
    risk = config.get("risk", {})

    tau_buy = thresholds.get("tau_buy", 0.025)
    tau_sell = thresholds.get("tau_sell", 0.00)

    # Precios
    close = market_data.get("close")
    if pd.isna(close):
        return None

    # --- Lógica de Salida (SELL) ---
    if position is not None:
        entry_price = position["entry_price"]
        entry_date = position["entry_date"]
        
        # 1. Retorno actual del trade
        ret_trade = (close / entry_price) - 1.0
        
        # Stop Loss
        sl_pct = risk.get("stop_loss_pct", 0.07)
        if ret_trade < -sl_pct:
            return {"action": "SELL", "reason": "stop_loss"}
            
        # Take Profit
        tp_pct = risk.get("take_profit_pct", 0.10)
        if ret_trade > tp_pct:
            return {"action": "SELL", "reason": "take_profit"}
            
        # Time Stop
        time_limit = risk.get("time_stop_days", 20)
        days_held = (current_date - entry_date).days
        if days_held > time_limit:
            return {"action": "SELL", "reason": "time_stop"}
            
        # Salida por señal del modelo (predicción bajista)
        if model_pred < tau_sell:
             return {"action": "SELL", "reason": "model_signal"}

        # Si no hay señal de venta, mantener (HOLD)
        return None

    # --- Lógica de Entrada (BUY) ---
    if position is None:
        # 1. Señal principal del modelo
        if model_pred > tau_buy:
            
            # 2. Filtros de confirmación (si están habilitados)
            
            # Filtro RSI (evitar sobrecompra extrema o sobreventa extrema)
            rsi = market_data.get("rsi_14", 50)
            rsi_min = filters.get("rsi14_min", 35)
            rsi_max = filters.get("rsi14_max", 70)
            if not (rsi_min <= rsi <= rsi_max):
                return None # Filtrado por RSI
                
            # Filtro MACD (momentum positivo)
            if filters.get("macd_confirmation", True):
                macd_hist = market_data.get("macd_hist", 0)
                if macd_hist <= 0:
                     return None # Filtrado por MACD (momentum no acompaña)

            # Filtro Volumen (participación)
            if filters.get("volume_zscore_min", 0) > -999: # solo si configurado
                vol_min = filters.get("volume_zscore_min", 0)
                vol_z = market_data.get("volume_zscore_20", 0)
                # Nota: En README dice vol_zscore > 0, aquí lo hacemos configurable
                if vol_z < vol_min:
                    return None # Filtrado por Volumen

            # Si pasa todos los filtros -> COMPRAR
            # Tamaño de posición: simplificado a 100% equity disponible (menos cash buffer)
            # En portafolio real sería allocation ponderado.
            return {"action": "BUY"}
            
    return None

"""
Reglas de trading para estrategia E4 (pairs trading).

Implementa:
- Señales de entrada basadas en z-score (±2.0)
- Señales de salida (z-score cruza 0)
- Stop-loss (z-score > ±3.0)
- Time-stop (días sin reversión)
- Gestión dollar-neutral
- Backtest de pares
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_pair_signals(
    zscore: pd.Series,
    spread: pd.Series,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.25,
    stop_z: float = 3.0,
    time_stop_days: int = 20,
    use_knn: bool = False,
    knn_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Genera señales de trading para pairs trading basadas en z-score.
    
    Lógica:
    - Entrada: |z-score| >= entry_z
    - Salida: |z-score| <= exit_z (convergencia)
    - Stop: |z-score| >= stop_z (breakdown)
    - Time-stop: days_held >= time_stop_days sin reversión
    
    Args:
        zscore: Serie de z-score del spread
        spread: Serie del spread
        entry_z: Umbral de z-score para entrada (e.g., 2.0)
        exit_z: Umbral de z-score para salida (e.g., 0.25)
        stop_z: Umbral de z-score para stop-loss (e.g., 3.0)
        time_stop_days: Días máximos sin reversión
        use_knn: Si True, usar k-NN para confirmar señales
        knn_signals: DataFrame con señales k-NN (si use_knn=True)
    
    Returns:
        DataFrame con señales:
        - signal: 1 (long spread), -1 (short spread), 0 (flat)
        - position: Posición actual (mantenida hasta salida)
        - entry_reason: Razón de entrada
        - exit_reason: Razón de salida
        - days_held: Días en posición
    """
    
    # Alinear
    df = pd.DataFrame({"zscore": zscore, "spread": spread}).dropna()
    
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)
    positions = np.zeros(n, dtype=np.int8)
    entry_reasons = [""] * n
    exit_reasons = [""] * n
    days_held = np.zeros(n, dtype=np.int32)
    
    current_pos = 0
    days_in_pos = 0
    entry_zscore = 0.0
    
    for i in range(n):
        z = df["zscore"].iloc[i]
        
        # Actualizar días en posición
        if current_pos != 0:
            days_in_pos += 1
        
        # Evaluar salidas primero (si estamos en posición)
        if current_pos != 0:
            exit_now = False
            reason = ""
            
            # 1. Exit normal: z-score cruza hacia 0
            if abs(z) <= exit_z:
                exit_now = True
                reason = "convergence"
            
            # 2. Stop-loss: z-score se aleja más (breakdown)
            elif abs(z) >= stop_z:
                exit_now = True
                reason = "stop_loss"
            
            # 3. Time-stop: demasiado tiempo sin reversión
            elif days_in_pos >= time_stop_days:
                exit_now = True
                reason = "time_stop"
            
            # 4. Reversión de z-score (cruzó la media)
            # Si entramos en +z y ahora z < 0, salir
            # Si entramos en -z y ahora z > 0, salir
            elif (current_pos > 0 and z < -exit_z) or (current_pos < 0 and z > exit_z):
                exit_now = True
                reason = "mean_reversion"
            
            if exit_now:
                exit_reasons[i] = reason
                days_held[i] = days_in_pos  # Guardar ANTES de resetear
                current_pos = 0
                days_in_pos = 0
        
        # Evaluar entradas (si estamos flat)
        if current_pos == 0:
            entry_now = False
            reason = ""
            
            # Condición 1: Z-score alto (spread anormalmente alto)
            if z >= entry_z:
                # Short spread (short A, long B)
                # Apostar a que spread baja
                current_pos = -1
                entry_now = True
                reason = "z_high"
                entry_zscore = z
            
            # Condición 2: Z-score bajo (spread anormalmente bajo)
            elif z <= -entry_z:
                # Long spread (long A, short B)
                # Apostar a que spread sube
                current_pos = 1
                entry_now = True
                reason = "z_low"
                entry_zscore = z
            
            # Confirmación k-NN (opcional)
            if entry_now and use_knn and knn_signals is not None:
                idx = df.index[i]
                if idx in knn_signals.index:
                    knn_signal = knn_signals.loc[idx, "signal"]
                    knn_conf = knn_signals.loc[idx, "confidence"]
                    
                    # Verificar que k-NN coincide con señal z-score
                    # k-NN signal: dirección esperada del spread
                    # Si z-score alto (short spread), queremos knn_signal < 0 (spread baja)
                    # Si z-score bajo (long spread), queremos knn_signal > 0 (spread sube)
                    
                    knn_signal_val = int(knn_signal) if not pd.isna(knn_signal) else 0
                    knn_conf_val = float(knn_conf) if not pd.isna(knn_conf) else 0.0
                    
                    if current_pos == -1 and knn_signal_val >= 0:
                        # k-NN no confirma (espera spread suba/flat, pero queremos baje)
                        if knn_conf_val > 0.5:  # Solo rechazar si k-NN tiene alta confianza
                            current_pos = 0
                            entry_now = False
                            reason = ""
                    
                    elif current_pos == 1 and knn_signal_val <= 0:
                        # k-NN no confirma (espera spread baje/flat, pero queremos suba)
                        if knn_conf_val > 0.5:
                            current_pos = 0
                            entry_now = False
                            reason = ""
            
            if entry_now:
                days_in_pos = 0
                entry_reasons[i] = reason
        
        # Registrar estado
        signals[i] = current_pos
        positions[i] = current_pos
        if exit_reasons[i] == "":  # Solo actualizar si no es exit
            days_held[i] = days_in_pos
    
    # Construir DataFrame
    df_signals = pd.DataFrame({
        "signal": signals,
        "position": positions,
        "entry_reason": entry_reasons,
        "exit_reason": exit_reasons,
        "days_held": days_held,
    }, index=df.index)
    
    return df_signals


def backtest_pair_strategy(
    price_a: pd.Series,
    price_b: pd.Series,
    signals: pd.DataFrame,
    *,
    hedge_ratio: pd.Series | float | None = None,
    round_trip_bps: float = 20.0,  # 2 activos x 10 bps = 20 bps
    initial_capital: float = 100000.0,
    position_size: float = 0.25,  # 25% del capital por par
) -> pd.DataFrame:
    """
    Backtest de estrategia pairs trading con gestión dollar-neutral.
    
    Args:
        price_a: Precios del activo A
        price_b: Precios del activo B
        signals: DataFrame con señales (position column)
        hedge_ratio: Hedge ratio (β) para calcular posiciones. Si None, usa 1.0.
        round_trip_bps: Costos de transacción total (20 bps para 2 activos)
        initial_capital: Capital inicial
        position_size: Tamaño de posición como % de capital
    
    Returns:
        DataFrame con backtest:
        - position: Posición en spread (1/-1/0)
        - shares_a: Acciones del activo A
        - shares_b: Acciones del activo B
        - value_a: Valor de posición A
        - value_b: Valor de posición B
        - net_exposure: Exposición neta (debe ser ~0)
        - pnl: PnL diario
        - equity: Curva de equity
        - costs: Costos de transacción
    """
    
    # Alinear datos
    df = pd.DataFrame({
        "price_a": price_a,
        "price_b": price_b,
    }).dropna()
    
    # Alinear con señales
    df["position"] = signals["position"].reindex(df.index, fill_value=0)
    
    # Preparar hedge ratio
    if hedge_ratio is None:
        # Default: beta = 1.0 (igual peso)
        df["beta"] = 1.0
    elif isinstance(hedge_ratio, (int, float)):
        df["beta"] = float(hedge_ratio)
    else:
        # Series: alinear con índice
        df["beta"] = hedge_ratio.reindex(df.index, method="ffill").fillna(1.0)
    
    n = len(df)
    
    # Arrays de resultados
    shares_a = np.zeros(n)
    shares_b = np.zeros(n)
    value_a = np.zeros(n)
    value_b = np.zeros(n)
    net_exposure = np.zeros(n)
    pnl_daily = np.zeros(n)
    costs_daily = np.zeros(n)
    equity = np.zeros(n)
    
    # Estado inicial
    current_shares_a = 0.0
    current_shares_b = 0.0
    cash = initial_capital
    
    for i in range(n):
        pos = df["position"].iloc[i]
        pa = df["price_a"].iloc[i]
        pb = df["price_b"].iloc[i]
        
        # Calcular PnL de posición existente
        if i > 0:
            pa_prev = df["price_a"].iloc[i-1]
            pb_prev = df["price_b"].iloc[i-1]
            
            pnl_a = current_shares_a * (pa - pa_prev)
            pnl_b = current_shares_b * (pb - pb_prev)
            pnl_daily[i] = pnl_a + pnl_b
            cash += pnl_daily[i]
        
        # Detectar cambio de posición
        prev_pos = df["position"].iloc[i-1] if i > 0 else 0
        
        if pos != prev_pos:
            # Cerrar posición anterior (si existe)
            if prev_pos != 0:
                # Vender posiciones
                cash += current_shares_a * pa
                cash += current_shares_b * pb
                
                # Costos de cierre
                cost = (
                    abs(current_shares_a * pa) + abs(current_shares_b * pb)
                ) * (round_trip_bps / 20000.0)  # Dividir por 2 para salida
                costs_daily[i] += cost
                cash -= cost
                
                current_shares_a = 0.0
                current_shares_b = 0.0
            
            # Abrir nueva posición (si no es flat)
            if pos != 0:
                # Dollar-neutral pairs trading
                # Objetivo: Invertir igual cantidad en cada lado
                # |value_A| = |value_B| = dollar_amount / 2
                # 
                # Esto garantiza net_exposure ≈ 0
                # El hedge ratio β se usa para construir el spread y señales,
                # pero NO para determinar el tamaño de las posiciones
                
                dollar_amount = initial_capital * position_size
                half_amount = dollar_amount / 2.0
                
                if pos == 1:
                    # Long spread: Long A, Short B
                    # Invertir mitad en cada lado
                    current_shares_a = half_amount / pa
                    current_shares_b = -half_amount / pb
                elif pos == -1:
                    # Short spread: Short A, Long B  
                    current_shares_a = -half_amount / pa
                    current_shares_b = half_amount / pb
                
                # Ajustar cash
                cash -= current_shares_a * pa
                cash -= current_shares_b * pb
                
                # Costos de apertura
                cost = (
                    abs(current_shares_a * pa) + abs(current_shares_b * pb)
                ) * (round_trip_bps / 20000.0)  # Dividir por 2 para entrada
                costs_daily[i] += cost
                cash -= cost
        
        # Registrar estado
        shares_a[i] = current_shares_a
        shares_b[i] = current_shares_b
        value_a[i] = current_shares_a * pa
        value_b[i] = current_shares_b * pb
        net_exposure[i] = value_a[i] + value_b[i]
        
        # Equity = cash + valor de posiciones
        equity[i] = cash + value_a[i] + value_b[i]
    
    # Construir DataFrame
    df_backtest = pd.DataFrame({
        "position": df["position"],
        "shares_a": shares_a,
        "shares_b": shares_b,
        "value_a": value_a,
        "value_b": value_b,
        "net_exposure": net_exposure,
        "pnl": pnl_daily,
        "costs": costs_daily,
        "equity": equity,
    }, index=df.index)
    
    return df_backtest


def summarize_pair_backtest(
    backtest: pd.DataFrame,
    signals: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
) -> Dict:
    """
    Resume resultados del backtest de un par.
    
    Args:
        backtest: DataFrame con resultados de backtest
        signals: DataFrame con señales
        ticker_a: Ticker del activo A
        ticker_b: Ticker del activo B
    
    Returns:
        Dict con métricas:
        - total_return: Retorno total
        - cagr: CAGR anualizado
        - sharpe: Sharpe ratio
        - max_drawdown: Máximo drawdown
        - num_trades: Número de trades
        - win_rate: Tasa de acierto
        - avg_holding_days: Días promedio por trade
        - net_exposure_mean: Exposición neta promedio (debe ser ~0)
    """
    
    equity = backtest["equity"]
    initial_capital = equity.iloc[0]
    final_capital = equity.iloc[-1]
    
    # Retorno total
    total_return = (final_capital / initial_capital) - 1.0
    
    # CAGR
    n_days = len(equity)
    years = n_days / 252.0
    if years > 0:
        cagr = (final_capital / initial_capital) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0
    
    # Sharpe ratio
    daily_returns = backtest["pnl"] / equity.shift(1)
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(daily_returns) > 1:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Número de trades
    position_changes = backtest["position"].diff().fillna(0)
    num_trades = (position_changes != 0).sum() // 2  # Dividir por 2 (entrada+salida)
    
    # Win rate
    exits = signals[signals["exit_reason"] != ""].copy()
    if len(exits) > 0:
        winning_exits = exits[
            (exits["exit_reason"] == "convergence") |
            (exits["exit_reason"] == "mean_reversion")
        ]
        win_rate = len(winning_exits) / len(exits)
    else:
        win_rate = 0.0
    
    # Días promedio por trade
    if len(exits) > 0:
        avg_holding_days = exits["days_held"].mean()
    else:
        avg_holding_days = 0.0
    
    # Exposición neta promedio (market-neutral check)
    # Debe estar cerca de 0 para estrategia market-neutral
    net_exposure_mean = backtest["net_exposure"].mean()
    
    return {
        "pair": f"{ticker_a}-{ticker_b}",
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "num_trades": int(num_trades),
        "win_rate": win_rate,
        "avg_holding_days": avg_holding_days,
        "net_exposure_mean": net_exposure_mean,
    }

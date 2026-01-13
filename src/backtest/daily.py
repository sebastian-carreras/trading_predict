"""
Backtesting engine para estrategias diarias (E1, E2).

Simula trading basado en predicciones de retorno multi-día con:
- Señales de compra/venta basadas en umbrales
- Costos de transacción realistas
- Gestión de posición (long/flat/short)
- Cálculo de métricas de performance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_daily_signals(
    timestamps: pd.DatetimeIndex,
    close_prices: np.ndarray,
    pred_returns: np.ndarray,
    *,
    tau_buy: float = 0.02,
    tau_sell: float = 0.02,
    round_trip_bps: float = 10.0,
    holding_period_days: int = 90,
    allow_short: bool = False,
    max_position: float = 1.0,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    """
    Backtest de estrategia diaria con predicciones de retorno.
    
    Parámetros
    ----------
    timestamps : pd.DatetimeIndex
        Timestamps de cada predicción (alineados con test set)
    close_prices : np.ndarray
        Precios de cierre históricos
    pred_returns : np.ndarray
        Predicciones de retorno del modelo (e.g., retorno a 90 días)
    tau_buy : float
        Umbral de predicción para señal de compra (e.g., 0.02 = +2%)
    tau_sell : float
        Umbral de predicción para señal de venta (e.g., 0.02 = -2%)
    round_trip_bps : float
        Costos de transacción en bps (10 bps = 0.1%)
    holding_period_days : int
        Período de holding antes de re-evaluar posición (e.g., 90 días)
    allow_short : bool
        Permitir posiciones cortas
    max_position : float
        Tamaño máximo de posición (1.0 = 100% capital)
    initial_capital : float
        Capital inicial en USD
        
    Retorna
    -------
    pd.DataFrame
        Backtest con columnas:
        - pos: posición (1=long, 0=flat, -1=short)
        - signal: señal generada (1=buy, -1=sell, 0=hold)
        - gross_ret: retorno bruto diario
        - costs: costos de transacción
        - net_ret: retorno neto (gross - costs)
        - equity: curva de equity
        - turnover: turnover diario
    """
    
    n = len(timestamps)
    if not (len(close_prices) == len(pred_returns) == n):
        raise ValueError("Inputs must have same length")
    
    # Calcular retornos diarios logarítmicos
    log_prices = np.log(close_prices)
    daily_returns = np.diff(log_prices)
    daily_returns = np.concatenate([[0.0], daily_returns])
    
    # Generar señales de trading
    signal = np.zeros(n, dtype=np.float32)
    signal[pred_returns >= tau_buy] = 1.0  # Comprar
    if allow_short:
        signal[pred_returns <= -tau_sell] = -1.0  # Vender en corto
    else:
        signal[pred_returns <= -tau_sell] = 0.0  # Cerrar posición (flat)
    
    # Gestión de posición con holding period
    pos = np.zeros(n, dtype=np.float32)
    days_held = 0
    current_pos = 0.0
    
    for i in range(n):
        if days_held >= holding_period_days or current_pos == 0:
            # Re-evaluar posición después del holding period o si estamos flat
            current_pos = signal[i] * max_position
            days_held = 0
        
        pos[i] = current_pos
        days_held += 1
    
    # Calcular PnL
    # Retorno bruto: posición actual * retorno del día siguiente
    gross_ret = np.zeros(n, dtype=np.float32)
    gross_ret[:-1] = pos[:-1] * daily_returns[1:]
    
    # Costos de transacción
    position_change = np.abs(np.diff(pos))
    position_change = np.concatenate([[0.0], position_change])
    
    # Costo = bps * tamaño del cambio
    # round_trip_bps incluye entrada + salida, dividir por 2 para cada trade
    costs = (position_change * (round_trip_bps / 10000.0) / 2.0).astype(np.float32)
    
    # Retorno neto
    net_ret = gross_ret - costs
    
    # Equity curve (en log space para composición)
    equity_log = np.cumsum(net_ret)
    equity = initial_capital * np.exp(equity_log)
    
    # Turnover (cambio de posición como % del capital)
    turnover = position_change
    
    # Construir DataFrame
    df = pd.DataFrame({
        'pos': pos,
        'signal': signal,
        'gross_ret': gross_ret,
        'costs': costs,
        'net_ret': net_ret,
        'equity': equity,
        'turnover': turnover,
    }, index=timestamps)
    
    return df


def compute_sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio anualizado.
    
    Parámetros
    ----------
    returns : np.ndarray
        Retornos diarios (log returns)
    periods_per_year : int
        Días de trading por año (252 para diario)
        
    Retorna
    -------
    float
        Sharpe ratio anualizado
    """
    if len(returns) == 0:
        return 0.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret == 0:
        return 0.0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    return float(sharpe)


def compute_max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum drawdown de la curva de equity.
    
    Parámetros
    ----------
    equity : np.ndarray
        Curva de equity ($)
        
    Retorna
    -------
    float
        Max drawdown (positivo, e.g., 0.15 = -15%)
    """
    if len(equity) == 0:
        return 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    
    max_dd = float(np.max(drawdown))
    return max_dd


def compute_calmar_ratio(returns: np.ndarray, equity: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calmar ratio: retorno anualizado / max drawdown.
    
    Parámetros
    ----------
    returns : np.ndarray
        Retornos diarios
    equity : np.ndarray
        Curva de equity
    periods_per_year : int
        Períodos por año
        
    Retorna
    -------
    float
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # CAGR (Compound Annual Growth Rate)
    total_return = (equity[-1] / equity[0]) - 1
    years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Max drawdown
    max_dd = compute_max_drawdown(equity)
    
    if max_dd == 0:
        return 0.0
    
    calmar = cagr / max_dd
    return float(calmar)


def compute_profit_factor(returns: np.ndarray) -> float:
    """
    Profit factor: suma de retornos positivos / abs(suma de retornos negativos).
    
    Parámetros
    ----------
    returns : np.ndarray
        Retornos diarios
        
    Retorna
    -------
    float
        Profit factor (>1 = profitable)
    """
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    
    pf = gains / losses
    return float(pf)


def compute_sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    *,
    risk_free_rate: float = 0.0,
) -> float:
    """Sortino ratio: retorno excedente / downside deviation.

    Similar a Sharpe pero usando solo volatilidad negativa.
    """
    if len(returns) == 0:
        return 0.0

    # excess returns per period
    rf_per_period = risk_free_rate / periods_per_year if periods_per_year > 0 else 0.0
    excess = returns - rf_per_period

    downside = excess[excess < 0]
    if len(downside) == 0:
        # Sin retornos negativos: ratio muy alto (capado por callers si hace falta)
        return float("inf")

    downside_std = float(np.std(downside))
    if downside_std <= 0:
        return 0.0

    mean_excess = float(np.mean(excess))
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def compute_win_rate(returns: np.ndarray) -> float:
    """
    Win rate: porcentaje de días con retorno positivo.
    
    Parámetros
    ----------
    returns : np.ndarray
        Retornos diarios
        
    Retorna
    -------
    float
        Win rate (0.0 a 1.0)
    """
    if len(returns) == 0:
        return 0.0
    
    # Solo considerar días con posición (returns != 0)
    active_returns = returns[returns != 0]
    
    if len(active_returns) == 0:
        return 0.0
    
    wins = (active_returns > 0).sum()
    win_rate = wins / len(active_returns)
    
    return float(win_rate)


def summarize_backtest(bt: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Calcula métricas resumen del backtest.
    
    Parámetros
    ----------
    bt : pd.DataFrame
        Output de backtest_daily_signals
    periods_per_year : int
        Períodos por año (252 para diario)
        
    Retorna
    -------
    dict
        Métricas de performance
    """
    net_ret = bt['net_ret'].to_numpy()
    equity = bt['equity'].to_numpy()
    
    # Retorno total
    total_return = (equity[-1] / equity[0]) - 1
    
    # CAGR
    years = len(net_ret) / periods_per_year
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    # Métricas de riesgo
    sharpe = compute_sharpe_ratio(net_ret, periods_per_year)
    sortino = compute_sortino_ratio(net_ret, periods_per_year)
    max_dd = compute_max_drawdown(equity)
    calmar = compute_calmar_ratio(net_ret, equity, periods_per_year)
    
    # Métricas de trading
    profit_factor = compute_profit_factor(net_ret)
    win_rate = compute_win_rate(net_ret)
    
    # Turnover y costos
    avg_turnover = bt['turnover'].mean()
    total_costs = bt['costs'].sum()
    
    # Número de trades (cambios de posición)
    num_trades = (bt['turnover'] > 0).sum()
    
    # Time in market (% de días con posición != 0)
    time_in_market = (bt['pos'].abs() > 0).mean()
    
    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar': float(calmar),
        'profit_factor': float(profit_factor),
        'win_rate': float(win_rate),
        'hit_rate': float(win_rate),
        'num_trades': int(num_trades),
        'avg_turnover': float(avg_turnover),
        'total_costs': float(total_costs),
        'time_in_market': float(time_in_market),
    }

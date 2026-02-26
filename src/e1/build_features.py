"""
Construcción de features para E1 (Estrategia Conservadora).

Según especificación del documento:
- Precio/retorno core: log-retornos, retornos rolling, volatilidad, ATR, volume z-score
- Tendencia largo plazo: SMA(50/200), EMA(50), MACD, Bollinger, ADX
- Target: retorno acumulado a H=90 días
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_e1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features para E1 (conservadora, largo plazo).

    Args:
        df: DataFrame con OHLCV diario (index=timestamp, cols: open/high/low/close/volume)

    Returns:
        DataFrame con features (mismo índice que df)
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    # ===== Precio/retorno (core) =====
    # Estrategia: Capturar momentum y mean-reversion en diferentes horizontes temporales
    
    log_close = np.log(close)
    # 
    # ret_1d: Retorno logarítmico diario
    # Propósito: Captura la velocidad de cambio del precio día a día
    # Estrategia: Detectar días de fuerte movimiento (posibles señales de continuación o reversión)
    # out["ret_1d"] = log_close.diff()

    # Retornos rolling (momentum multi-horizonte)
    # OPTIMIZADO PARA LARGO PLAZO: Eliminamos ret_5d (ruido para horizon=90 días)
    
    # ret_1w: Retorno semanal (5 días hábiles)
    # Propósito: Momentum semanal (escala más alineada con horizon=90 días)
    # Estrategia: Captura tendencias semanales sin ruido diario. Si >0 → semana alcista, si <0 → bajista
    out["ret_1w"] = log_close.diff(5)
    
    # ret_4w: Retorno mensual (~20 días hábiles = 4 semanas)
    # Propósito: Momentum mensual (escala crítica para horizon=90 días = 13 semanas)
    # Estrategia: Acciones con ret_4w alto tienden a continuar subiendo próximos 90 días (momentum persistente)
    out["ret_4w"] = log_close.diff(20)
    
    # ret_13w: Retorno trimestral (~60 días hábiles = 13 semanas)
    # Propósito: Momentum trimestral (perfectamente alineado con horizon=90 días)
    # Estrategia: Detecta tendencias de largo plazo. Valores extremos indican sobrecompra/sobreventa a escala trimestral
    out["ret_13w"] = log_close.diff(60)

    # Volatilidad realizada (medida de riesgo)
    # Estrategia: Alta volatilidad = mayor incertidumbre → ajustar predicciones y tamaño de posición
    # OPTIMIZADO PARA LARGO PLAZO: Una sola medida de volatilidad mensual (más estable)
    
    # vol_4w: Volatilidad (std de retornos) últimos 20 días (~4 semanas)
    # Propósito: Medir riesgo mensual (más relevante para horizon=90 días que volatilidad diaria)
    # Estrategia: 
    #   - vol_4w alta → Mayor riesgo/oportunidad. Predicciones menos confiables, reducir tamaño de posición
    #   - vol_4w baja → Mercado estable. Predicciones más confiables, posible squeeze (breakout inminente)
    #   - Cambios en vol_4w indican cambios de régimen (calma → tormenta o viceversa)
    out["vol_4w"] = log_close.diff().rolling(20).std()
    
    # vol_regime: Cambio de volatilidad (vol_4w actual vs vol_4w hace 4 semanas)
    # Propósito: Detectar expansión o contracción de volatilidad
    # Estrategia:
    #   - vol_regime > 0.5 → Volatilidad EXPANSIÓN (aumento >50%) → Mayor incertidumbre, ajustar posiciones
    #   - vol_regime < -0.3 → Volatilidad CONTRACCIÓN (reducción >30%) → Mercado calmándose, possible squeeze
    #   - vol_regime ~ 0 → Volatilidad estable
    # IMPORTANTE: Cambios bruscos de volatilidad predicen reversiones de tendencia
    vol_4w_lag = out["vol_4w"].shift(20)
    out["vol_regime"] = (out["vol_4w"] / (vol_4w_lag + 1e-12)) - 1.0

    # Rango intradía y ATR (Average True Range)
    # Estrategia: Medir la amplitud real de movimiento (más robusto que solo high-low)
    
    # ELIMINADA POR SER POCO ÚTIL PARA LARGO PLAZO:
    # range_pct: Rango intradía normalizado (high-low) / close
    # Propósito: Detectar días de alta volatilidad intradía
    # Estrategia: Días con range_pct alto → indecisión del mercado o fuerte disputa entre compradores/vendedores
    # range_pct alto (>3-5%) puede indicar reversión inminente o alta volatilidad
    # range_pct bajo (<1%) indica mercado tranquilo
    # range_pct es útil para ajustar stops y evaluar riesgo intradía
    # out["range_pct"] = (high - low) / close
    
    # True Range: Máximo entre (high-low), (high-close_prev), (low-close_prev)
    # Propósito: Captura gaps y movimientos que el simple high-low no detecta
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    
    # atr_14: Average True Range normalizado (promedio de TR últimos 14 días / close)
    # Propósito: Volatilidad "real" considerando gaps
    # Estrategia: ATR alto → mayor riesgo, necesitas stops más amplios. ATR bajo → mercado tranquilo, posible breakout
    out["atr_14"] = tr.rolling(14).mean() / close

    # Volumen z-score (anomalías en volumen de trading)
    # Estrategia: Volumen anormal indica convicción de mercado (confirmación de tendencia o reversión)
    
    # vol_zscore_60: Z-score del volumen vs últimos 60 días
    # Propósito: Detectar días con volumen anormalmente alto o bajo
    # Estrategia: 
    #   - vol_zscore > 2 → Volumen excepcional (ej. earnings, noticias) → confirma movimiento de precio
    #   - vol_zscore < -1 → Volumen muy bajo → movimiento de precio poco confiable (puede revertir)
    #   - vol_zscore ~ 0 → Volumen normal, sin señal especial
    vol_roll = volume.rolling(60)
    out["vol_zscore_60"] = (volume - vol_roll.mean()) / (vol_roll.std() + 1e-12)

    # ===== Tendencia (largo plazo) =====
    # Estrategia: Identificar tendencia primaria y posición relativa del precio
    
    # sma_50: Simple Moving Average de 50 días
    # Propósito: Tendencia de medio plazo (2-3 meses)
    # Estrategia: Si close > sma_50 → tendencia alcista de medio plazo. Golden cross: sma_50 > sma_200 → bull market
    out["sma_50"] = close.rolling(50).mean()
    
    # sma_200: Simple Moving Average de 200 días
    # Propósito: Tendencia de largo plazo (filtro fundamental)
    # Estrategia: Si close > sma_200 → mercado alcista estructural. Muchos traders NO compran si close < sma_200
    out["sma_200"] = close.rolling(200).mean()
    
    # sma50_sma200_ratio: Relación entre SMA(50) y SMA(200)
    # Propósito: Detectar Golden Cross / Death Cross y medir fuerza de tendencia
    # Estrategia:
    #   - ratio > 0 (SMA50 > SMA200) → GOLDEN CROSS activo → tendencia alcista de largo plazo confirmada
    #   - ratio < 0 (SMA50 < SMA200) → DEATH CROSS activo → tendencia bajista de largo plazo confirmada
    #   - ratio cerca de 0 → SMAs convergiendo → posible cruce inminente (alta importancia)
    #   - ratio muy positivo (ej. +0.05 = +5%) → tendencia alcista MUY fuerte
    #   - ratio muy negativo (ej. -0.05 = -5%) → tendencia bajista MUY fuerte
    # IMPORTANTE: Esta feature captura la señal técnica más seguida por inversores institucionales
    out["sma50_sma200_ratio"] = (out["sma_50"] / out["sma_200"]) - 1.0
    
    # close_sma200_dist: Distancia relativa entre precio actual y SMA(200)
    # Propósito: Medir cuán "estirado" está el precio respecto a su tendencia de largo plazo
    # Estrategia:
    #   - Muy positivo (ej. +0.2 = +20%) → posible sobrecompra, candidato a corrección
    #   - Muy negativo (ej. -0.2 = -20%) → posible sobreventa, candidato a rebote
    #   - Cerca de 0 → precio en línea con tendencia de largo plazo (zona "justa")
    out["close_sma200_dist"] = (close / out["sma_200"]) - 1.0

    # Eliminado por ser redundante con sma50 y sma50_sma200_ratio
    # ema_50: Exponential Moving Average de 50 días
    # Propósito: Similar a SMA(50) pero reacciona más rápido a cambios recientes
    # Estrategia: Usar junto con SMA(50) para detectar aceleración/desaceleración de tendencia
    # out["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence) - Indicador de momentum
    # Estrategia: Detectar cambios en fuerza, dirección, momentum y duración de tendencia
    
    ema12 = close.ewm(span=12, adjust=False).mean()  # EMA rápida (corto plazo)
    ema26 = close.ewm(span=26, adjust=False).mean()  # EMA lenta (largo plazo)
    
    # macd_line: Diferencia entre EMA(12) y EMA(26)
    # Propósito: Medir momentum (qué tan rápido se mueve el precio)
    # Estrategia:
    #   - macd_line > 0 → EMA corto > EMA largo → momentum alcista
    #   - macd_line < 0 → EMA corto < EMA largo → momentum bajista
    #   - macd_line creciente → momentum acelerándose
    macd_line = ema12 - ema26
    # Eliminado por ser redundante con macd_hist
    # out["macd_line"] = macd_line
    
    # macd_signal: EMA(9) de la MACD line (línea de señal)
    # Propósito: Suavizar la MACD para generar señales más confiables
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    # Eliminado por ser redundante con macd_hist
    # out["macd_signal"] = macd_signal
    
    # macd_hist: Histograma MACD (diferencia entre MACD line y signal)
    # Propósito: Medir divergencia entre momentum actual y su tendencia
    # Estrategia:
    #   - macd_hist > 0 y creciente → momentum fuerte y acelerándose → comprar
    #   - macd_hist < 0 y decreciente → momentum débil y desacelerándose → vender
    #   - Cruce de macd_line con macd_signal → señal clásica de trading
    out["macd_hist"] = macd_line - macd_signal

    # Bollinger Bands (20, 2) - Bandas de volatilidad
    # Estrategia: Identificar condiciones de sobrecompra/sobreventa y volatilidad
    
    sma20 = close.rolling(20).mean()  # Media de 20 días
    std20 = close.rolling(20).std()   # Desviación estándar de 20 días
    bb_upper = sma20 + 2 * std20      # Banda superior (95% de valores están dentro)
    bb_lower = sma20 - 2 * std20      # Banda inferior
    
    # bb_pct_b: %B de Bollinger (posición relativa del precio en las bandas)
    # Propósito: Normalizar la posición del precio respecto a las bandas
    # Estrategia:
    #   - bb_pct_b > 1.0 → Precio SOBRE banda superior → sobrecompra extrema (posible reversión bajista)
    #   - bb_pct_b > 0.8 → Precio cerca de banda superior → zona de sobrecompra
    #   - bb_pct_b ~ 0.5 → Precio en la media (neutral)
    #   - bb_pct_b < 0.2 → Precio cerca de banda inferior → zona de sobreventa
    #   - bb_pct_b < 0.0 → Precio BAJO banda inferior → sobreventa extrema (posible reversión alcista)
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-12)
    
    # bb_bandwidth: Ancho de las bandas normalizado
    # Propósito: Medir expansión/contracción de volatilidad
    # Estrategia:
    #   - bb_bandwidth alto → Alta volatilidad, bandas anchas → mercado activo
    #   - bb_bandwidth bajo → Baja volatilidad, bandas estrechas → "squeeze" → posible breakout inminente
    #   - Bollinger Squeeze: cuando bb_bandwidth está en mínimos históricos → alta probabilidad de movimiento fuerte
    out["bb_bandwidth"] = (bb_upper - bb_lower) / sma20

    # ADX (Average Directional Index) - Fuerza de tendencia
    # Estrategia: Medir FUERZA de la tendencia (no dirección). Complementa MACD/SMA.
    # Nota: Esta es una versión simplificada. Cálculo completo requiere +DI/-DI normalizados.
    
    high_diff = high.diff()
    low_diff = -low.diff()
    
    # Directional Movement: Mide movimiento direccional (arriba vs abajo)
    plus_dm = pd.Series(0.0, index=df.index)   # Movimiento alcista
    minus_dm = pd.Series(0.0, index=df.index)  # Movimiento bajista
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

    atr = tr.rolling(14).mean()
    # +DI: Indicador direccional positivo (fuerza alcista)
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-12))
    # -DI: Indicador direccional negativo (fuerza bajista)
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-12))
    
    # DX: Directional Index (divergencia entre +DI y -DI)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12))
    
    # adx_14: Promedio del DX (fuerza de tendencia)
    # Propósito: Medir qué tan fuerte es la tendencia actual (alcista o bajista)
    # Estrategia:
    #   - adx_14 < 20 → Tendencia DÉBIL o ausente → mercado lateral (rango) → evitar estrategias de momentum
    #   - adx_14 entre 20-40 → Tendencia MODERADA → tendencia presente pero no extrema
    #   - adx_14 > 40 → Tendencia FUERTE → alta convicción del mercado → seguir la tendencia (momentum funciona)
    #   - adx_14 > 60 → Tendencia MUY FUERTE → posible agotamiento (cuidado con reversión)
    # IMPORTANTE: ADX NO dice si la tendencia es alcista o bajista, solo qué tan fuerte es
    out["adx_14"] = dx.rolling(14).mean()

    return out


def make_target_e1(df: pd.DataFrame, horizon_days: int = 90) -> pd.Series:
    """Construye target: retorno acumulado a horizonte H.

    Args:
        df: DataFrame con columna 'close'
        horizon_days: Horizonte de predicción (90 días por defecto)

    Returns:
        Series con target (log-retorno acumulado forward)
    """
    close = df["close"].astype("float64")
    fwd_close = close.shift(-horizon_days)
    target = np.log(fwd_close / close)
    target.name = f"target_ret_{horizon_days}d"
    return target

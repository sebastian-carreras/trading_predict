"""
Construcción de features para E2 (Estrategia Moderada).

Según especificación del documento:
- Precio/retorno core: log-retornos, retornos rolling, volatilidad, ATR, volume z-score
- Tendencia medio plazo: SMA(20/50), EMA(20), MACD, Bollinger, ADX, RSI
- Target: retorno acumulado a H=20 días
- Enfoque: Mayor sensibilidad a momentum de corto plazo vs E1
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_e2_features(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula features para E2 (moderada, medio plazo).

    Args:
        df: DataFrame con OHLCV diario (index=timestamp, cols: open/high/low/close/volume)
        benchmark_df: DataFrame del benchmark (SPY) con mismo formato

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
    # ret_1d: Retorno logarítmico diario
    # Propósito: Captura la velocidad de cambio del precio día a día
    # Estrategia: Detectar días de fuerte movimiento (posibles señales de continuación o reversión)
    out["ret_1d"] = log_close.diff()

    # Retornos rolling (momentum multi-horizonte)
    # ret_5d: Retorno acumulado última semana (5 días)
    # Propósito: Momentum de muy corto plazo
    # Estrategia: Si >0 → tendencia alcista reciente, si <0 → bajista reciente
    out["ret_5d"] = out["ret_1d"].rolling(5).sum()
    
    # ret_20d: Retorno acumulado último mes (~20 días hábiles)
    # Propósito: Momentum de corto-medio plazo
    # Estrategia: Captura tendencias mensuales. Acciones con ret_20d alto tienden a continuar subiendo (momentum)
    out["ret_20d"] = out["ret_1d"].rolling(20).sum()
    
    # ret_60d: Retorno acumulado últimos 3 meses (~60 días hábiles)
    # Propósito: Momentum de medio plazo
    # Estrategia: Detecta tendencias trimestrales. Valores extremos pueden indicar sobrecompra/sobreventa
    out["ret_60d"] = out["ret_1d"].rolling(60).sum()

    # Volatilidad realizada (medida de riesgo)
    # Estrategia: Alta volatilidad = mayor incertidumbre → ajustar predicciones y tamaño de posición
    
    # vol_20d: Volatilidad (std de retornos) últimos 20 días
    # Propósito: Medir riesgo de corto plazo
    # Estrategia: Alta volatilidad reciente → mayor probabilidad de reversión (mean-reversion)
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    
    # vol_60d: Volatilidad (std de retornos) últimos 60 días
    # Propósito: Medir riesgo de medio plazo (baseline de volatilidad)
    # Estrategia: Si vol_20d >> vol_60d → régimen de alta volatilidad transitorio (oportunidad o peligro)
    out["vol_60d"] = out["ret_1d"].rolling(60).std()

    # Rango intradía y ATR (Average True Range)
    # Estrategia: Medir la amplitud real de movimiento (más robusto que solo high-low)
    
    # range_pct: Rango intradía normalizado (high-low) / close
    # Propósito: Detectar días de alta volatilidad intradía
    # Estrategia: Días con range_pct alto → indecisión del mercado o fuerte disputa entre compradores/vendedores
    out["range_pct"] = (high - low) / close
    
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

    # ema_50: Exponential Moving Average de 50 días
    # Propósito: Similar a SMA(50) pero reacciona más rápido a cambios recientes
    # Estrategia: Usar junto con SMA(50) para detectar aceleración/desaceleración de tendencia
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()

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
    out["macd_line"] = macd_line
    
    # macd_signal: EMA(9) de la MACD line (línea de señal)
    # Propósito: Suavizar la MACD para generar señales más confiables
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_signal"] = macd_signal
    
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

    # ===== RSI (Relative Strength Index) =====
    # Estrategia: Identificar condiciones de sobrecompra/sobreventa (más útil para E2 de medio plazo)
    
    # Calcular cambios de precio
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # RSI(14): Media móvil exponencial de ganancias y pérdidas
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    
    # rsi_14: Relative Strength Index (0-100)
    # Propósito: Medir momentum y detectar extremos de precio
    # Estrategia:
    #   - rsi_14 > 70 → SOBRECOMPRA → alta probabilidad de corrección bajista (vender o evitar comprar)
    #   - rsi_14 > 80 → SOBRECOMPRA EXTREMA → reversión inminente
    #   - rsi_14 entre 40-60 → Zona NEUTRAL → sin señal clara
    #   - rsi_14 < 30 → SOBREVENTA → alta probabilidad de rebote alcista (comprar oportunidad)
    #   - rsi_14 < 20 → SOBREVENTA EXTREMA → reversión inminente
    # IMPORTANTE: RSI funciona mejor en mercados laterales (rango). En tendencias fuertes puede dar señales falsas.
    out["rsi_14"] = 100 - (100 / (1 + rs))

    return out


def make_target_e2(df: pd.DataFrame, horizon_days: int = 20) -> pd.Series:
    """Construye target: retorno acumulado a horizonte H.

    Args:
        df: DataFrame con columna 'close'
        horizon_days: Horizonte de predicción (20 días por defecto para E2)

    Returns:
        Series con target (log-retorno acumulado forward)
    """
    close = df["close"].astype("float64")
    fwd_close = close.shift(-horizon_days)
    target = np.log(fwd_close / close)
    target.name = f"target_ret_{horizon_days}d"
    return target

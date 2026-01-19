#!/usr/bin/env python
"""
Script de trading en vivo con E1 Simple + API IOL (ambiente de prueba).

Flujo:
1. Autenticación con IOL usando credenciales del .env
2. Cargar modelo E1 Simple entrenado
3. Descargar datos recientes del ticker
4. Calcular features y hacer predicción
5. Si decision_score ≥ 0.70 (BUY) → ejecutar orden en IOL

Uso:
    python scripts/e1_simple_iol_live_trade.py --ticker AAPL --model runs/e1_simple/20260119_120000/AAPL/AAPL_model.pth

Variables de entorno necesarias (.env):
    IOL_USERNAME=tu_usuario
    IOL_PASSWORD=tu_password
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.e1_gru import GRURegressor
from src.features.build_features_e1 import compute_e1_features
from src.utils import load_yaml


class IOLClient:
    """Cliente para interactuar con API de Invertir Online (ambiente de prueba)."""
    
    BASE_URL_TEST = "https://api.invertironline.com"  # Ambiente de prueba
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def authenticate(self) -> bool:
        """
        Autenticación con IOL.
        
        POST https://api.invertironline.com/token
        Content-Type: application/x-www-form-urlencoded
        
        grant_type=password&username=xxx&password=yyy
        
        Returns:
            True si autenticación exitosa
        """
        url = f"{self.BASE_URL_TEST}/token"
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password
        }
        
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get("access_token")
            self.refresh_token = token_data.get("refresh_token")
            expires_in = token_data.get("expires_in", 3600)  # segundos
            
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            print(f"✓ Autenticación exitosa. Token expira en {expires_in}s")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error en autenticación: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Respuesta: {e.response.text}")
            return False
    
    def get_auth_headers(self) -> dict:
        """Retorna headers con token de autenticación."""
        if not self.access_token:
            raise RuntimeError("No autenticado. Llamar authenticate() primero.")
        
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def get_account_info(self) -> dict:
        """
        Obtiene información de la cuenta.
        GET https://api.invertironline.com/api/v2/portafolio/argentina
        """
        url = f"{self.BASE_URL_TEST}/api/v2/portafolio/argentina"
        
        try:
            response = requests.get(url, headers=self.get_auth_headers())
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error obteniendo info de cuenta: {e}")
            return {}
    
    def place_market_order(
        self,
        ticker: str,
        quantity: int,
        side: str = "compra"  # "compra" o "venta"
    ) -> dict:
        """
        Ejecuta orden a mercado en IOL.
        
        POST https://api.invertironline.com/api/v2/operar/Comprar
        {
            "mercado": "bCBA",
            "simbolo": "GGAL",
            "cantidad": 1,
            "precio": 123.45,
            "plazo": "t2",
            "validez": "dia"
        }
        
        Args:
            ticker: Símbolo del activo (ej: GGAL, YPFD)
            quantity: Cantidad de acciones
            side: "compra" o "venta"
        
        Returns:
            Respuesta de la API
        """
        # Para ambiente de prueba, usar mercado bCBA (Bolsa argentina)
        endpoint = "Comprar" if side == "compra" else "Vender"
        url = f"{self.BASE_URL_TEST}/api/v2/operar/{endpoint}"
        
        payload = {
            "mercado": "bCBA",
            "simbolo": ticker,
            "cantidad": quantity,
            "plazo": "t2",
            "validez": "dia"
        }
        
        try:
            response = requests.post(
                url,
                headers=self.get_auth_headers(),
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ Orden ejecutada: {side.upper()} {quantity} {ticker}")
            print(f"  Detalles: {result}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error ejecutando orden: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Respuesta: {e.response.text}")
            return {}


def load_e1_simple_model(model_path: Path) -> tuple[GRURegressor, dict, dict]:
    """
    Carga modelo E1 Simple entrenado.
    
    Returns:
        (model, scaler_X, scaler_y)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    print(f"Cargando modelo: {model_path.name}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extraer configuración
    model_kwargs = checkpoint['model_kwargs']
    
    # Recrear modelo
    model = GRURegressor(**model_kwargs)
    model.model.load_state_dict(checkpoint['state_dict'])
    model.model.eval()
    
    # Scalers
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    print(f"✓ Modelo cargado: {checkpoint['ticker']} - {checkpoint['strategy']}")
    print(f"  Lookback: {checkpoint['lookback_days']}d | Horizon: {checkpoint['horizon_days']}d")
    print(f"  Features: {len(checkpoint['feature_names'])}")
    
    return model, scaler_X, scaler_y


def download_recent_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """Descarga datos recientes de yfinance."""
    import yfinance as yf
    
    print(f"Descargando {days} días de {ticker}...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{days}d")
    
    if df.empty:
        raise ValueError(f"No se pudieron descargar datos para {ticker}")
    
    # Renombrar columnas a minúsculas
    df.columns = df.columns.str.lower()
    df.index.name = 'timestamp'
    
    print(f"✓ Descargados {len(df)} días (desde {df.index[0].date()} hasta {df.index[-1].date()})")
    
    return df


def predict_with_model(
    model: GRURegressor,
    ohlcv: pd.DataFrame,
    scaler_X: dict,
    scaler_y: dict,
    lookback: int,
    benchmark_df: pd.DataFrame = None
) -> tuple[float, pd.DataFrame]:
    """
    Hace predicción usando modelo E1 Simple.
    
    Returns:
        (predicted_return, features_df)
    """
    # Calcular features
    features = compute_e1_features(ohlcv, benchmark_df)
    
    # Necesitamos al menos lookback días
    if len(features) < lookback:
        raise ValueError(f"Insuficientes datos. Necesitamos {lookback} días, tenemos {len(features)}")
    
    # Tomar últimos lookback días
    recent_features = features.iloc[-lookback:]
    
    # Convertir a numpy
    X = recent_features.values.astype(np.float32)
    
    # Normalizar usando scaler del entrenamiento
    mean_X = np.array(scaler_X['mean'])
    std_X = np.array(scaler_X['std'])
    X_scaled = (X - mean_X) / std_X
    
    # Reshape para modelo: (1, lookback, features)
    X_input = X_scaled.reshape(1, lookback, -1)
    X_tensor = torch.FloatTensor(X_input)
    
    # Predecir
    with torch.no_grad():
        y_pred_scaled = model.predict(X_tensor)
    
    # Desnormalizar
    pred_return = float(y_pred_scaled[0] * scaler_y['std'] + scaler_y['mean'])
    
    print(f"✓ Predicción: retorno esperado = {pred_return:+.2%}")
    
    return pred_return, features


def compute_simple_decision_score(
    pred_return: float,
    config: dict
) -> tuple[float, str]:
    """
    Calcula decision score basado solo en la predicción.
    
    Nota: En producción, deberías calcular IC, Sharpe, etc. con datos out-of-sample.
    Aquí usamos solo el retorno predicho vs tau_buy como proxy.
    
    Returns:
        (decision_score, signal)
    """
    e1_simple = config.get("strategies", {}).get("e1_simple", {})
    tau_buy = float(e1_simple.get("thresholds", {}).get("tau_buy", 0.06))
    
    decision_cfg = config.get("decision", {})
    threshold = float(decision_cfg.get("threshold", 0.70))
    
    # Score simplificado: si pred_return > tau_buy, asumimos que el modelo
    # tiene confianza suficiente (esto es una aproximación)
    # En un sistema real, necesitarías métricas recientes de performance
    
    if pred_return >= tau_buy:
        # Mapear pred_return a [0, 1] de forma heurística
        # Si pred = tau_buy → 0.7, si pred > 2*tau_buy → 1.0
        score = min(0.70 + (pred_return - tau_buy) / tau_buy * 0.30, 1.0)
        signal = "buy" if score >= threshold else "hold"
    else:
        score = max(0.0, 0.70 * (pred_return / tau_buy))
        signal = "hold"
    
    print(f"✓ Decision score: {score:.3f} (threshold: {threshold:.2f}) → {signal.upper()}")
    
    return score, signal


def main():
    parser = argparse.ArgumentParser(description="Trading en vivo con E1 Simple + IOL")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker a operar (ej: GGAL, YPFD para IOL Argentina)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path al modelo .pth (ej: runs/e1_simple/.../AAPL_model.pth)"
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Cantidad de acciones a comprar si BUY (default: 1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path al config YAML"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular orden sin ejecutar en IOL"
    )
    
    args = parser.parse_args()
    
    # Cargar .env
    load_dotenv()
    
    # Credenciales IOL
    iol_username = os.getenv("IOL_USERNAME")
    iol_password = os.getenv("IOL_PASSWORD")
    
    if not iol_username or not iol_password:
        print("❌ Error: Faltan credenciales IOL en .env")
        print("   Agrega: IOL_USERNAME=tu_usuario")
        print("           IOL_PASSWORD=tu_password")
        sys.exit(1)
    
    print("=" * 70)
    print("E1 Simple - Trading en vivo con API IOL (ambiente de prueba)")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Modelo: {args.model}")
    print(f"Cantidad: {args.quantity}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    print()
    
    # 1. Cargar configuración
    root = Path(__file__).parent.parent
    config = load_yaml(root / args.config)
    
    # 2. Autenticación IOL
    print("1️⃣  Autenticando con IOL...")
    iol = IOLClient(iol_username, iol_password)
    
    if not iol.authenticate():
        print("❌ Autenticación fallida. Verifica credenciales en .env")
        sys.exit(1)
    
    # Verificar cuenta
    account_info = iol.get_account_info()
    if account_info:
        print(f"✓ Cuenta conectada")
    print()
    
    # 3. Cargar modelo
    print("2️⃣  Cargando modelo E1 Simple...")
    model_path = Path(args.model)
    model, scaler_X, scaler_y = load_e1_simple_model(model_path)
    
    # Extraer lookback del checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    lookback_days = checkpoint['lookback_days']
    print()
    
    # 4. Descargar datos recientes
    print("3️⃣  Descargando datos recientes...")
    ohlcv = download_recent_data(args.ticker, days=365)
    
    # Descargar benchmark si está disponible
    benchmark_ticker = config.get("universe", {}).get("benchmark", "SPY")
    try:
        benchmark_df = download_recent_data(benchmark_ticker, days=365)
    except:
        print(f"⚠️  No se pudo descargar benchmark {benchmark_ticker}, continuando sin él")
        benchmark_df = None
    print()
    
    # 5. Hacer predicción
    print("4️⃣  Calculando predicción...")
    pred_return, features = predict_with_model(
        model, ohlcv, scaler_X, scaler_y, lookback_days, benchmark_df
    )
    print()
    
    # 6. Calcular decision score
    print("5️⃣  Evaluando decisión...")
    decision_score, signal = compute_simple_decision_score(pred_return, config)
    print()
    
    # 7. Ejecutar orden si BUY
    print("6️⃣  Ejecutando acción...")
    
    if signal == "buy":
        print(f"🟢 Señal: COMPRAR {args.quantity} {args.ticker}")
        
        if args.dry_run:
            print("   [DRY RUN] Orden simulada, no se ejecuta en IOL")
        else:
            print("   Ejecutando orden en IOL...")
            result = iol.place_market_order(
                ticker=args.ticker,
                quantity=args.quantity,
                side="compra"
            )
            
            if result:
                print("✓ Orden ejecutada exitosamente")
            else:
                print("❌ Error al ejecutar orden")
    else:
        print(f"🔴 Señal: HOLD (no se ejecuta orden)")
        print(f"   Razón: decision_score ({decision_score:.3f}) < threshold (0.70)")
    
    print()
    print("=" * 70)
    print("✓ Proceso completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

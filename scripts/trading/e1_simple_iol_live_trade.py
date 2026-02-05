#!/usr/bin/env python
"""
Script de trading en vivo con E1 Simple + API IOL (ambiente de prueba).

Flujo:
1. Autenticación con IOL usando credenciales del .env
2. Cargar modelo E1 Simple entrenado
3. Descargar datos recientes del ticker
4. Calcular features y hacer predicción
5. Si pred_return ≥ tau_buy → ejecutar orden en IOL

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.e1_gru import GRURegressor
from src.features.build_features_e1 import compute_e1_features
from src.utils import load_yaml


class IOLClient:
    """Cliente para interactuar con API de Invertir Online (ambiente de prueba)."""
    
    BASE_URL_PROD = "https://api.invertironline.com"
    BASE_URL_TEST = "https://api.invertironline.com"  # Sandbox
    
    def __init__(self, username: str, password: str, use_production: bool = False):
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.base_url = self.BASE_URL_PROD if use_production else self.BASE_URL_TEST
        self.environment = "PRODUCCIÓN" if use_production else "PRUEBA"
    
    def authenticate(self) -> bool:
        """
        Autenticación con IOL.
        
        POST https://api.invertironline.com/token
        Content-Type: application/x-www-form-urlencoded
        
        grant_type=password&username=xxx&password=yyy
        
        Returns:
            True si autenticación exitosa
        """
        url = f"{self.base_url}/token"
        
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
        side: str = "compra",  # "compra" o "venta"
        market: str = "bCBA"
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
            market: Mercado IOL (bCBA, aNYS, etc.)
        
        Returns:
            Respuesta de la API
        """
        # Convertir ticker de formato yfinance (GGAL.BA) a formato IOL (GGAL)
        iol_ticker = ticker.replace('.BA', '')
        
        # Endpoint según operación
        endpoint = "Comprar" if side == "compra" else "Vender"
        url = f"{self.base_url}/api/v2/operar/{endpoint}"
        
        # Payload según documentación oficial IOL
        # https://api.invertironline.com/Help#!/Operar/Operar_Comprar
        payload = {
            "mercado": market,
            "simbolo": iol_ticker,
            "cantidad": quantity,
            "precio": 0,  # 0 = orden a mercado
            "plazo": "t2",  # t0, t1, t2
            "validez": None,  # None = día, o fecha ISO
            "tipoOrden": "precioMercado"  # precioMercado o precioLimite
        }
        
        try:
            response = requests.post(
                url,
                headers=self.get_auth_headers(),
                json=payload
            )
            
            # Imprimir detalles de la solicitud para debug
            print(f"   DEBUG - Request payload: {payload}")
            print(f"   DEBUG - Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ Orden ejecutada: {side.upper()} {quantity} {ticker}")
            print(f"  Detalles: {result}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error ejecutando orden: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Status code: {e.response.status_code}")
                print(f"   Headers: {dict(e.response.headers)}")
                try:
                    error_json = e.response.json()
                    print(f"   Respuesta JSON: {error_json}")
                except:
                    text = e.response.text
                    print(f"   Respuesta texto ({len(text)} chars): {text[:500]}")
            
            # Agregar sugerencias según el error
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 400:
                    print(f"\n   💡 Posibles causas:")
                    print(f"      - El símbolo '{iol_ticker}' no existe en mercado '{market}'")
                    print(f"      - El ambiente de prueba no soporta este instrumento")
                    print(f"      - Usa tickers argentinos válidos: GGAL, YPFD, BBAR, etc.")

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
    feature_names: list,
    benchmark_df: pd.DataFrame = None
) -> tuple[float, pd.DataFrame]:
    """
    Hace predicción usando modelo E1 Simple.
    
    Returns:
        (predicted_return, features_df)
    """
    # Calcular features
    features = compute_e1_features(ohlcv, benchmark_df)
    
    # Verificar que tengamos las features correctas
    if len(features.columns) != len(feature_names):
        print(f"⚠️  Warning: Calculadas {len(features.columns)} features, modelo espera {len(feature_names)}")
        print(f"   Features calculadas: {list(features.columns)[:5]}...")
        print(f"   Features modelo: {feature_names[:5]}...")
    
    # Reordenar y filtrar solo las features del modelo
    try:
        features = features[feature_names]
    except KeyError as e:
        print(f"❌ Error: Falta feature {e}")
        print(f"   Features disponibles: {list(features.columns)}")
        raise
    
    # Necesitamos al menos lookback días
    if len(features) < lookback:
        raise ValueError(f"Insuficientes datos. Necesitamos {lookback} días, tenemos {len(features)}")
    
    # Eliminar NaN/Inf
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0)
    
    # Tomar últimos lookback días
    recent_features = features.iloc[-lookback:]
    
    # Convertir a numpy
    X = recent_features.values.astype(np.float32)
    
    # Normalizar usando scaler del entrenamiento
    mean_X = np.array(scaler_X['mean'])
    std_X = np.array(scaler_X['std'])
    
    # Verificar dimensiones
    if X.shape[1] != len(mean_X):
        raise ValueError(f"Mismatch de features: datos={X.shape[1]}, scaler={len(mean_X)}")
    
    X_scaled = (X - mean_X) / std_X
    
    # Verificar NaN en datos normalizados
    if np.isnan(X_scaled).any():
        print(f"⚠️  Warning: NaN detectados en features normalizadas, reemplazando con 0")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reshape para modelo: (1, lookback, features)
    X_input = X_scaled.reshape(1, lookback, -1)
    X_tensor = torch.FloatTensor(X_input)
    
    # Predecir
    with torch.no_grad():
        y_pred_scaled = model.predict(X_tensor)
    
    # Desnormalizar
    pred_return = float(y_pred_scaled[0] * scaler_y['std'] + scaler_y['mean'])
    
    # Sanitizar predicción
    if not np.isfinite(pred_return):
        print(f"⚠️  Warning: Predicción no finita ({pred_return}), usando 0.0")
        pred_return = 0.0
    
    print(f"✓ Predicción: retorno esperado = {pred_return:+.2%}")
    
    return pred_return, features


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
        "--market",
        type=str,
        default="bCBA",
        help="Mercado IOL (bCBA, aNYS, etc.)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular orden sin ejecutar en IOL"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Usar ambiente de PRODUCCIÓN (por defecto usa sandbox/prueba)"
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
    
    ambiente = "PRODUCCIÓN ⚠️" if args.production else "PRUEBA"
    print("=" * 70)
    print(f"E1 Simple - Trading en vivo con API IOL (ambiente: {ambiente})")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Modelo: {args.model}")
    print(f"Cantidad: {args.quantity}")
    print(f"Mercado: {args.market}")
    print(f"Dry run: {args.dry_run}")
    print(f"Ambiente: {ambiente}")
    print("=" * 70)
    print()
    
    # 1. Cargar configuración
    root = Path(__file__).parent.parent
    config = load_yaml(root / args.config)
    
    # 2. Autenticación IOL
    print(f"1️⃣  Autenticando con IOL (ambiente: {('PRODUCCIÓN' if args.production else 'PRUEBA')})...")
    iol = IOLClient(iol_username, iol_password, use_production=args.production)
    
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
    
    # Extraer lookback y feature_names del checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    lookback_days = checkpoint['lookback_days']
    feature_names = checkpoint.get('feature_names', None)
    
    if feature_names is None:
        print("⚠️  Warning: Modelo no tiene feature_names guardados")
    
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
        model, ohlcv, scaler_X, scaler_y, lookback_days, feature_names, benchmark_df
    )
    
    # Guardar predicción en MLflow para evaluación continua
    if not args.dry_run:
        try:
            import mlflow
            
            project_root = Path(__file__).parent.parent
            mlflow_uri = "sqlite:///" + str(project_root / "runs" / "mlflow_local" / "mlflow.db")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("E1_Simple_Production_Tracking")
            
            with mlflow.start_run(run_name=f"prediction_{args.ticker}_{datetime.now().strftime('%Y%m%d')}"):
                mlflow.log_param("ticker", args.ticker)
                mlflow.log_param("prediction_date", datetime.now().strftime("%Y-%m-%d"))
                mlflow.log_param("horizon_days", 90)
                mlflow.log_param("model_path", str(model_path))
                mlflow.log_param("status", "pending")
                mlflow.log_metric("predicted_return", pred_return)
                
                print(f"  ✓ Predicción guardada en MLflow para evaluación futura")
        except Exception as e:
            print(f"  ⚠️  No se pudo guardar en MLflow: {e}")
    
    print()
    
    # 6. Evaluar señal por umbral
    print("5️⃣  Evaluando decisión...")
    e1_simple = config.get("strategies", {}).get("e1_simple", {})
    tau_buy = float(e1_simple.get("thresholds", {}).get("tau_buy", 0.06))
    signal = "buy" if pred_return >= tau_buy else "hold"
    print()
    
    # 7. Ejecutar orden si BUY
    print("6️⃣  Ejecutando acción...")
    
    if signal == "buy":
        print(f"🟢 Señal: COMPRAR {args.quantity} {args.ticker}")
        
        if args.dry_run:
            print("   [DRY RUN] Orden simulada, no se ejecuta en IOL")
        else:
            print(f"   Ejecutando orden en IOL (mercado: {args.market})...")
            result = iol.place_market_order(
                ticker=args.ticker,
                quantity=args.quantity,
                side="compra",
                market=args.market
            )
            
            if result:
                print("✓ Orden ejecutada exitosamente")
            else:
                print("❌ Error al ejecutar orden")
    else:
        print(f"🔴 Señal: HOLD (no se ejecuta orden)")
        print(f"   Razón: pred_return ({pred_return:+.2%}) < tau_buy ({tau_buy:.2%})")
    
    print()
    print("=" * 70)
    print("✓ Proceso completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

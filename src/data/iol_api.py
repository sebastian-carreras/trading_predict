"""
Cliente para API de InvertirOnline (IOL) - Backup para Yahoo Finance.

La API de IOL permite obtener datos históricos de acciones argentinas.
Requiere autenticación con usuario/contraseña para obtener un token.

Documentación: https://api.invertironline.com/
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv no instalado, usar variables de entorno del sistema


class IOLClient:
    """Cliente para API de InvertirOnline.
    
    Autenticación:
        1. Requiere usuario y contraseña de IOL
        2. Genera un bearer token válido por 15 minutos
        3. Usa refresh token para renovar automáticamente
    
    Variables de entorno requeridas:
        - IOL_USERNAME: Usuario de IOL
        - IOL_PASSWORD: Contraseña de IOL
    """
    
    BASE_URL = "https://api.invertironline.com"
    
    def __init__(self, username: str | None = None, password: str | None = None):
        """Inicializa cliente IOL.
        
        Args:
            username: Usuario IOL (si None, usa variable IOL_USERNAME)
            password: Contraseña IOL (si None, usa variable IOL_PASSWORD)
        """
        self.username = username or os.getenv("IOL_USERNAME")
        self.password = password or os.getenv("IOL_PASSWORD")
        
        if not self.username or not self.password:
            raise ValueError(
                "Se requieren credenciales IOL. Define IOL_USERNAME y IOL_PASSWORD "
                "como variables de entorno o pasa username/password al constructor."
            )
        
        self.bearer_token: str | None = None
        self.refresh_token: str | None = None
        self.token_expires_at: datetime | None = None
    
    def _authenticate(self) -> None:
        """Autentica y obtiene tokens de acceso."""
        url = f"{self.BASE_URL}/token"
        payload = {
            "username": self.username,
            "password": self.password,
            "grant_type": "password"
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        self.bearer_token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        
        # Token válido por 15 minutos (usar 14 para margen de seguridad)
        self.token_expires_at = datetime.now() + timedelta(minutes=14)
    
    def _ensure_authenticated(self) -> None:
        """Asegura que hay un token válido, renovando si es necesario."""
        if not self.bearer_token or not self.token_expires_at:
            self._authenticate()
            return
        
        # Renovar si expira en menos de 1 minuto
        if datetime.now() >= self.token_expires_at - timedelta(minutes=1):
            if self.refresh_token:
                self._refresh_authentication()
            else:
                self._authenticate()
    
    def _refresh_authentication(self) -> None:
        """Renueva el token usando refresh_token."""
        url = f"{self.BASE_URL}/token"
        payload = {
            "refresh_token": self.refresh_token,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        self.bearer_token = data["access_token"]
        self.token_expires_at = datetime.now() + timedelta(minutes=14)
    
    def _get_headers(self) -> dict[str, str]:
        """Retorna headers con autenticación."""
        self._ensure_authenticated()
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
    
    def get_historical_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        mercado: str = "bCBA",
        ajustada: str = "sinAjustar"  # Cambiado a "sinAjustar" por defecto (compatible con bonos)
    ) -> pd.DataFrame:
        """Obtiene datos históricos OHLCV de un ticker.
        
        Args:
            ticker: Símbolo del ticker (ej: "GGAL", "YPF")
            start_date: Fecha inicio formato YYYY-MM-DD
            end_date: Fecha fin formato YYYY-MM-DD
            mercado: Mercado (bCBA=Buenos Aires, nYSE=New York)
            ajustada: "ajustada" o "sinAjustar"
        
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        # IOL usa símbolos sin .BA
        ticker_clean = ticker.replace(".BA", "")
        
        # Formato de fechas: YYYY-MM-DD
        url = (
            f"{self.BASE_URL}/api/v2/{mercado}/Titulos/{ticker_clean}/"
            f"Cotizacion/seriehistorica/{start_date}/{end_date}/{ajustada}"
        )
        
        response = requests.get(url, headers=self._get_headers(), timeout=30)
        
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame(data)
        
        # Mapear nombres de columnas IOL a formato estándar
        # IOL devuelve: fechaHora, apertura, maximo, minimo, ultimoPrecio, volumenNominal
        column_map = {
            "fechaHora": "timestamp",
            "apertura": "open",
            "maximo": "high",
            "minimo": "low",
            "ultimoPrecio": "close",  # Precio de cierre
            "volumenNominal": "volume"  # Volumen nominal
        }
        
        # Renombrar columnas que existan
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Asegurar que timestamp es datetime
        if "timestamp" in df.columns:
            # Usar format='ISO8601' para manejar timestamps con y sin microsegundos
            df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601', utc=True)
            df = df.sort_values("timestamp")
        
        # Crear columna adj_close (igual a close si es ajustada)
        if "close" in df.columns and "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        
        # Seleccionar solo columnas relevantes
        columns = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
        df = df[[col for col in columns if col in df.columns]]
        
        return df


def download_iol_daily(
    ticker: str,
    out_dir: Path,
    years: int = 10,
    username: str | None = None,
    password: str | None = None
) -> Path | None:
    """Descarga datos históricos de IOL para un ticker argentino.
    
    Args:
        ticker: Símbolo del ticker (con o sin .BA)
        out_dir: Directorio de salida
        years: Años de historia a descargar
        username: Usuario IOL (opcional, usa env var si None)
        password: Contraseña IOL (opcional, usa env var si None)
    
    Returns:
        Path del archivo CSV creado, o None si falla
    """
    try:
        client = IOLClient(username=username, password=password)
        
        # Calcular fechas
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        print(f"  📡 Descargando {ticker} desde IOL API...")
        df = client.get_historical_data(ticker, start_date, end_date)
        
        if df.empty:
            print(f"  ⚠️  IOL: Sin datos para {ticker}")
            return None
        
        # Validación básica
        if len(df) < 252:
            print(f"  ⚠️  IOL: {ticker} tiene menos de 1 año de datos ({len(df)} días)")
        
        # Guardar CSV
        out_path = out_dir / f"{ticker}_daily.csv"
        df.to_csv(out_path, index=False)
        print(f"  ✓ IOL: {len(df)} días guardados en {out_path.name}")
        
        return out_path
    
    except Exception as e:
        print(f"  ⚠️  IOL falló para {ticker}: {e}")
        return None

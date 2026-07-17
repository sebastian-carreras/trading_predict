#!/usr/bin/env python3
"""
Script para listar precios actuales de las acciones usadas en E1 Simple.
Usa el ambiente de PRODUCCIÓN de IOL para obtener cotizaciones en tiempo real.
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import json


class IOLClient:
    """Cliente para API de IOL en producción."""

    BASE_URL = "https://api.invertironline.com"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.access_token = None

    def authenticate(self) -> bool:
        """Autenticar con IOL."""
        url = f"{self.BASE_URL}/token"

        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password
        }

        try:
            response = requests.post(url, data=data)
            response.raise_for_status()

            result = response.json()
            self.access_token = result['access_token']

            print(f"✓ Autenticación exitosa (ambiente: PRODUCCIÓN)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Error en autenticación: {e}")
            return False

    def get_headers(self) -> dict:
        """Obtener headers con token."""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def get_cotizacion(self, ticker: str, market: str = "bCBA") -> dict:
        """
        Obtener cotización de un ticker.

        Args:
            ticker: Símbolo (ej: GGAL, YPFD)
            market: Mercado (bCBA por defecto)

        Returns:
            Dict con información de cotización
        """
        # Convertir ticker de yfinance a IOL
        iol_ticker = ticker.replace('.BA', '')

        # Probar endpoint de cotización
        url = f"{self.BASE_URL}/api/v2/{market}/Titulos/{iol_ticker}/Cotizacion"

        try:
            response = requests.get(url, headers=self.get_headers())

            if response.status_code == 200:
                return response.json()
            else:
                # Probar endpoint alternativo
                url2 = f"{self.BASE_URL}/api/v2/Cotizaciones/{market}/{iol_ticker}/Punta"
                response2 = requests.get(url2, headers=self.get_headers())

                if response2.status_code == 200:
                    return response2.json()
                else:
                    return {
                        'error': True,
                        'status': response.status_code,
                        'message': f'No disponible ({response.status_code})'
                    }

        except Exception as e:
            return {
                'error': True,
                'message': str(e)
            }


def get_e1_simple_tickers(base_path: Path) -> list:
    """Obtener lista de tickers de modelos E1 Simple."""

    e1_simple_path = base_path / "runs" / "e1_simple"

    if not e1_simple_path.exists():
        print(f"⚠️  No se encontró directorio: {e1_simple_path}")
        return []

    tickers = set()

    # Buscar modelos .pth
    for model_file in e1_simple_path.rglob("*.pth"):
        # Extraer ticker del nombre (ej: AAPL_model.pth -> AAPL)
        ticker = model_file.stem.replace('_model', '')
        tickers.add(ticker)

    return sorted(list(tickers))


def format_price(value, decimals=2):
    """Formatear precio."""
    if value is None or value == 0:
        return "N/A"
    return f"${value:,.{decimals}f}"


def main():
    # Cargar .env
    load_dotenv()

    username = os.getenv("IOL_USERNAME")
    password = os.getenv("IOL_PASSWORD")

    if not username or not password:
        print("❌ Error: Faltan credenciales IOL en .env")
        sys.exit(1)

    print("=" * 80)
    print("E1 Simple - Precios Actuales de Acciones (IOL PRODUCCIÓN)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Obtener tickers de E1 Simple
    base_path = Path(__file__).parent.parent
    tickers = get_e1_simple_tickers(base_path)

    if not tickers:
        print("❌ No se encontraron modelos E1 Simple")
        sys.exit(1)

    print(f"📊 Encontrados {len(tickers)} tickers en E1 Simple:")
    print(f"   {', '.join(tickers[:10])}" + ("..." if len(tickers) > 10 else ""))
    print()

    # Autenticar con IOL
    print("🔐 Autenticando con IOL (PRODUCCIÓN)...")
    iol = IOLClient(username, password)

    if not iol.authenticate():
        print("❌ No se pudo autenticar con IOL")
        sys.exit(1)

    print()
    print("💰 Obteniendo cotizaciones...")
    print()

    # Obtener cotizaciones
    resultados = []

    for ticker in tickers:
        cotizacion = iol.get_cotizacion(ticker)

        if cotizacion.get('error'):
            resultados.append({
                'ticker': ticker,
                'status': 'ERROR',
                'message': cotizacion.get('message', 'Desconocido')
            })
        else:
            # Extraer información relevante
            resultados.append({
                'ticker': ticker,
                'status': 'OK',
                'data': cotizacion
            })

    # Mostrar resultados
    print("=" * 80)
    print(f"{'TICKER':<15} {'STATUS':<10} {'ÚLTIMO':<15} {'VARIACIÓN':<15} {'VOLUMEN':<15}")
    print("=" * 80)

    for res in resultados:
        ticker = res['ticker']
        status = res['status']

        if status == 'OK':
            data = res['data']

            # Extraer campos comunes (pueden variar según endpoint)
            ultimo = data.get('ultimoPrecio') or data.get('precio') or data.get('puntas', {}).get('precioCompra')
            variacion = data.get('variacionPorcentual') or data.get('variacion')
            volumen = data.get('volumen') or data.get('cantidadOperada')

            ultimo_str = format_price(ultimo, 2) if ultimo else "N/A"
            variacion_str = f"{variacion:+.2f}%" if variacion else "N/A"
            volumen_str = f"{volumen:,}" if volumen else "N/A"

            print(f"{ticker:<15} {status:<10} {ultimo_str:<15} {variacion_str:<15} {volumen_str:<15}")
        else:
            mensaje = res.get('message', 'Error')
            print(f"{ticker:<15} {status:<10} {mensaje}")

    print("=" * 80)
    print()

    # Guardar resultados en JSON
    output_file = base_path / "reports" / "trading" / f"iol_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(resultados, f, indent=2, default=str)

    print(f"✓ Resultados guardados en: {output_file}")
    print()


if __name__ == "__main__":
    main()

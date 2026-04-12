#!/usr/bin/env python3
"""
Consulta el portafolio actual de una cuenta de InvertirOnline.

Uso:
    python scripts/trading/iol_get_portfolio.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


BASE_URL = "https://api.invertironline.com"


def authenticate(username: str, password: str) -> str:
    """Obtiene un access token de IOL."""
    response = requests.post(
        f"{BASE_URL}/token",
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_portfolio(access_token: str, pais: str = "argentina") -> dict:
    """Obtiene el portafolio consolidado de la cuenta."""
    response = requests.get(
        f"{BASE_URL}/api/v2/portafolio/{pais}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def format_number(value: object, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}"
    return str(value)


def extract_symbol(item: dict) -> str:
    raw = item.get("simbolo") or item.get("titulo") or item.get("descripcion")
    if isinstance(raw, dict):
        return raw.get("simbolo") or raw.get("descripcion") or "N/A"
    return str(raw or "N/A")


def format_percent(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:+.2f}%"
    return str(value)


def main() -> int:
    load_dotenv()

    username = os.getenv("IOL_USERNAME")
    password = os.getenv("IOL_PASSWORD")

    if not username or not password:
        print("Faltan IOL_USERNAME/IOL_PASSWORD en el entorno.", file=sys.stderr)
        return 1

    try:
        token = authenticate(username, password)
        portfolio = get_portfolio(token)
    except requests.HTTPError as exc:
        print(f"Error HTTP consultando IOL: {exc}", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text[:1000], file=sys.stderr)
        return 2
    except requests.RequestException as exc:
        print(f"Error de red consultando IOL: {exc}", file=sys.stderr)
        return 3

    titulo_keys = ("titulo", "titulos", "activos")
    titulos = []
    for key in titulo_keys:
        value = portfolio.get(key)
        if isinstance(value, list):
            titulos = value
            break

    print("=" * 80)
    print("PORTAFOLIO IOL")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pais: {portfolio.get('pais') or 'argentina'}")
    print(f"Cuenta: {portfolio.get('numero') or portfolio.get('cuenta') or 'N/A'}")
    print(f"Total: ${format_number(portfolio.get('total'))}")
    print(f"Disponible: ${format_number(portfolio.get('disponible'))}")
    print(f"Comprometido: ${format_number(portfolio.get('comprometido'))}")
    print(f"Titulos: {len(titulos)}")
    print("=" * 80)

    if titulos:
        print(
            f"{'SIMBOLO':<12} {'CANTIDAD':>10} {'PPC':>12} {'ULTIMO':>12} "
            f"{'GAN.$':>12} {'GAN.%':>10} {'DIA %':>10}"
        )
        print("-" * 80)
        for item in titulos:
            simbolo = extract_symbol(item)
            cantidad = item.get("cantidad") or item.get("cantidadTotal") or item.get("cantNominal")
            ppc = item.get("ppc")
            ultimo = item.get("ultimoPrecio") or item.get("precioMercado")
            ganancia_dinero = item.get("gananciaDinero")
            ganancia_porcentaje = item.get("gananciaPorcentaje")
            variacion_diaria = item.get("variacionDiaria")
            print(
                f"{str(simbolo):<12} "
                f"{format_number(cantidad):>10} "
                f"{format_number(ppc):>12} "
                f"{format_number(ultimo):>12} "
                f"{format_number(ganancia_dinero):>12} "
                f"{format_percent(ganancia_porcentaje):>10} "
                f"{format_percent(variacion_diaria):>10}"
            )
    else:
        print("No se detectaron tenencias listadas en la respuesta.")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / f"iol_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(portfolio, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 80)
    print(f"JSON guardado en: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

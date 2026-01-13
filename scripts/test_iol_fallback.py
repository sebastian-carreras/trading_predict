"""
Script de prueba para verificar el fallback IOL.

Este script simula un escenario donde Yahoo Finance falla
y el sistema debe recurrir a IOL API como fallback.
"""

from pathlib import Path
import os
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download_daily import download_daily_ohlcv
from src.data.iol_api import download_iol_daily
from src.utils import project_root


def test_iol_direct():
    """Prueba descarga directa con IOL (sin fallback)."""
    
    root = project_root()
    out_dir = root / "data" / "raw" / "daily"
    
    # Verificar credenciales IOL
    username = os.getenv("IOL_USERNAME")
    password = os.getenv("IOL_PASSWORD")
    
    print("=" * 70)
    print("TEST: Descarga DIRECTA con IOL API")
    print("=" * 70)
    print(f"\n📍 Directorio de salida: {out_dir}")
    print(f"🔐 Usuario IOL: {username if username else '✗ No configurado'}")
    
    if not username or not password:
        print("\n❌ ERROR: Credenciales IOL no configuradas")
        print("\n   Para configurar:")
        print("   export IOL_USERNAME='tu_usuario'")
        print("   export IOL_PASSWORD='tu_contraseña'")
        return
    
    # Solo tickers argentinos (IOL solo tiene mercado argentino)
    test_tickers = [
        "GGAL.BA",  # Banco Galicia
        "YPFD.BA",  # YPF
        "PAMP.BA",  # Pampa Energía
    ]
    
    print(f"\n📊 Tickers de prueba: {', '.join(test_tickers)}")
    print("\n" + "=" * 70)
    print("INICIANDO DESCARGA DIRECTA DESDE IOL")
    print("=" * 70 + "\n")
    
    # Descargar directamente con IOL
    written = []
    for ticker in test_tickers:
        result = download_iol_daily(
            ticker=ticker,
            username=username,
            password=password,
            out_dir=out_dir,
            years=2,
        )
        if result:
            written.append(result)
    
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"✓ Archivos descargados: {len(written)}/{len(test_tickers)}")
    
    if written:
        print("\nArchivos creados:")
        for path in written:
            size_kb = path.stat().st_size / 1024
            print(f"  📄 {path.name} ({size_kb:.1f} KB)")
    
    success_rate = len(written) / len(test_tickers) * 100
    print(f"\n📈 Tasa de éxito: {success_rate:.0f}%")
    
    if success_rate == 100:
        print("\n✅ ÉXITO: Todos los tickers se descargaron correctamente desde IOL")
    elif success_rate > 0:
        print("\n⚠️  PARCIAL: Algunos tickers fallaron")
    else:
        print("\n❌ ERROR: Ningún ticker se pudo descargar desde IOL")
    
    print("\n" + "=" * 70)


def test_iol_fallback():
    """Prueba el sistema de fallback con IOL."""
    
    root = project_root()
    out_dir = root / "data" / "raw" / "daily"
    
    # Verificar si las credenciales IOL están configuradas
    iol_configured = bool(os.getenv("IOL_USERNAME") and os.getenv("IOL_PASSWORD"))
    
    print("=" * 70)
    print("TEST: Sistema de Fallback Yahoo Finance → IOL API")
    print("=" * 70)
    print(f"\n📍 Directorio de salida: {out_dir}")
    print(f"🔐 Credenciales IOL configuradas: {'✓ Sí' if iol_configured else '✗ No'}")
    
    if not iol_configured:
        print("\n⚠️  ADVERTENCIA: Credenciales IOL no configuradas")
        print("   El fallback no estará disponible si Yahoo Finance falla.")
        print("\n   Para configurar:")
        print("   export IOL_USERNAME='tu_usuario'")
        print("   export IOL_PASSWORD='tu_contraseña'")
        print()
    
    # Lista de tickers de prueba (mix argentinos y USA)
    test_tickers = [
        "GGAL.BA",  # Banco Galicia (Argentina)
        "YPFD.BA",  # YPF (Argentina)
        "AAPL",     # Apple (USA)
    ]
    
    print(f"\n📊 Tickers de prueba: {', '.join(test_tickers)}")
    print("\n" + "=" * 70)
    print("INICIANDO DESCARGA CON FALLBACK HABILITADO")
    print("=" * 70 + "\n")
    
    # Forzar reDescarga para ver el proceso completo
    written = download_daily_ohlcv(
        test_tickers,
        out_dir=out_dir,
        period="2y",
        skip_existing=False,  # Forzar descarga para ver el proceso
        use_iol_fallback=True,
    )
    
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"✓ Archivos descargados: {len(written)}/{len(test_tickers)}")
    
    if written:
        print("\nArchivos creados:")
        for path in written:
            size_kb = path.stat().st_size / 1024
            print(f"  📄 {path.name} ({size_kb:.1f} KB)")
    
    success_rate = len(written) / len(test_tickers) * 100
    print(f"\n📈 Tasa de éxito: {success_rate:.0f}%")
    
    if success_rate == 100:
        print("\n✅ ÉXITO: Todos los tickers se descargaron correctamente")
    elif success_rate > 0:
        print("\n⚠️  PARCIAL: Algunos tickers fallaron")
    else:
        print("\n❌ ERROR: Ningún ticker se pudo descargar")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IOL API integration")
    parser.add_argument(
        "--mode",
        choices=["direct", "fallback"],
        default="direct",
        help="Test mode: 'direct' (descarga directa IOL) o 'fallback' (Yahoo→IOL)"
    )
    args = parser.parse_args()
    
    if args.mode == "direct":
        test_iol_direct()
    else:
        test_iol_fallback()

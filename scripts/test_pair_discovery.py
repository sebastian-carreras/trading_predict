"""
Script para probar descubrimiento automático de pares cointegrados.
"""

import sys
from pathlib import Path

# Agregar src al path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.pairs.discover_pairs import discover_cointegrated_pairs, filter_best_pairs
from src.utils import load_yaml

def main():
    """Ejecutar descubrimiento de pares."""
    
    # Cargar config
    config = load_yaml(root / "src/config/base.yaml")
    e4_config = config.get("strategies", {}).get("e4_pairs", {})
    discovery_config = e4_config.get("discovery", {})
    
    # Universo de tickers
    tickers = e4_config.get("universe", [])
    
    print(f"Descubriendo pares cointegrados de {len(tickers)} tickers...")
    print(f"Parámetros:")
    print(f"  - P-value max: {discovery_config.get('pvalue_max', 0.05)}")
    print(f"  - Half-life range: [{discovery_config.get('half_life_min', 5)}, {discovery_config.get('half_life_max', 60)}] días")
    print(f"  - Correlación min: {discovery_config.get('correlation_min', 0.5)}")
    print()
    
    # Descubrir pares
    data_dir = root / "data/clean"
    
    pairs_df = discover_cointegrated_pairs(
        tickers,
        data_dir,
        pvalue_max=discovery_config.get("pvalue_max", 0.05),
        half_life_min=discovery_config.get("half_life_min", 5.0),
        half_life_max=discovery_config.get("half_life_max", 60.0),
        correlation_min=discovery_config.get("correlation_min", 0.5),
        method="engle-granger",
        verbose=True,
    )
    
    if len(pairs_df) == 0:
        print("\n❌ No se encontraron pares cointegrados")
        return
    
    print(f"\n✅ Descubiertos {len(pairs_df)} pares cointegrados:")
    print(pairs_df[['ticker_a', 'ticker_b', 'pvalue', 'ou_half_life', 'correlation', 'quality_score']].to_string(index=False))
    
    # Filtrar mejores
    best_pairs = filter_best_pairs(
        pairs_df,
        max_pairs=discovery_config.get("max_pairs", 20),
        min_quality_score=discovery_config.get("min_quality_score", 0.5),
        max_half_life=discovery_config.get("half_life_max", 60.0),
    )
    
    print(f"\n🎯 {len(best_pairs)} pares seleccionados para trading:")
    print(best_pairs[['ticker_a', 'ticker_b', 'pvalue', 'ou_half_life', 'quality_score']].to_string(index=False))
    
    # Guardar resultados
    output_dir = root / "runs/e4_pairs/discovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_csv = output_dir / "discovered_pairs_test.csv"
    pairs_df.to_csv(all_csv, index=False)
    print(f"\n💾 Todos los pares guardados en: {all_csv}")
    
    best_csv = output_dir / "selected_pairs_test.csv"
    best_pairs.to_csv(best_csv, index=False)
    print(f"💾 Pares seleccionados guardados en: {best_csv}")

if __name__ == "__main__":
    main()

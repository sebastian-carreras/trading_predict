"""
Script de ejemplo para probar la estrategia E4 con un par simple.

Uso:
    python scripts/test_e4_simple.py
"""

import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.train_e4_pipeline import main

if __name__ == "__main__":
    # Probar con un par simple de la configuración
    # Usar GGAL.BA y BMA.BA (bancos argentinos)
    pairs = [("GGAL.BA", "BMA.BA")]
    
    print("=" * 80)
    print("Testing E4 Pairs Trading Strategy")
    print("Pair: GGAL.BA - BMA.BA (Argentine Banks)")
    print("=" * 80)
    
    # Ejecutar pipeline
    main(pairs=pairs, output_tag="test_simple")
    
    print("\nTest complete! Check runs/e4_pairs/test_simple/ for results")

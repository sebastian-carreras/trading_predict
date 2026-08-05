"""Los DAGs de entrenamiento resuelven sus tickers desde base.yaml, no de listas fijas.

Contexto de la regresión que estos tests bloquean:

* ``params['tickers']`` traía una lista hardcodeada. La de E2 se escribió cuando
  ``e2_moderate`` tenía 10 tickers; el universo creció a 30 y el param nunca se
  actualizó, así que a 20 champions no se les entrenaba candidato nunca.
* El fallback de E1 leía ``universe.tickers``, clave que no existe en base.yaml
  (quedó de una versión previa). Con el param ya vacío, ese ``.get()`` devolvía
  ``[]`` y el DAG no habría descargado nada — un fallo silencioso.

Los tests leen el fuente de los DAGs en vez de importarlos porque importar un DAG
requiere Airflow instalado, que no está en el entorno de CI.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "dockerfiles" / "airflow" / "dags"

DAG_FILES = {
    "e1": DAGS / "E1" / "e1_conservative_pipeline.py",
    "e2": DAGS / "E2" / "e2_moderate_pipeline.py",
}
UNIVERSE_KEY = {"e1": "e1_conservative", "e2": "e2_moderate"}


@pytest.fixture(scope="module")
def universe():
    config = yaml.safe_load((ROOT / "src" / "config" / "base.yaml").read_text())
    return config["universe"]["tickers_by_strategy"]


def _dag_params(path: Path) -> dict:
    """Extrae el dict literal ``params={...}`` del fuente del DAG."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "params" and isinstance(kw.value, ast.Dict):
                    return ast.literal_eval(kw.value)
    raise AssertionError(f"no se encontró params={{...}} en {path.name}")


@pytest.mark.parametrize("strategy", ["e1", "e2"])
def test_tickers_param_is_empty_so_config_wins(strategy):
    """Un default no vacío hace que la rama de base.yaml sea código muerto."""
    params = _dag_params(DAG_FILES[strategy])
    assert params["tickers"].strip() == "", (
        f"{strategy}: la lista fija en params se desincroniza de base.yaml en silencio; "
        "dejala vacía y acotá corridas manuales con 'Trigger DAG w/ config'"
    )


@pytest.mark.parametrize("strategy", ["e1", "e2"])
def test_download_falls_back_to_a_non_empty_universe(strategy, universe):
    """Con el param vacío el fallback debe resolver a tickers reales, no a []."""
    source = DAG_FILES[strategy].read_text()
    key = UNIVERSE_KEY[strategy]
    assert f'"{key}"' in source, (
        f"{strategy}: el DAG debe leer universe.tickers_by_strategy.{key}"
    )
    assert len(universe[key]) > 0, f"universe.tickers_by_strategy.{key} está vacío"


@pytest.mark.parametrize("strategy", ["e1", "e2"])
def test_no_dag_reads_the_dead_universe_tickers_key(strategy):
    """``universe.tickers`` no existe en base.yaml: leerla resuelve a [] callado."""
    source = DAG_FILES[strategy].read_text()
    assert 'get("universe", {}).get("tickers", [])' not in source, (
        f"{strategy}: universe.tickers es una clave muerta; usar tickers_by_strategy"
    )


def test_universe_tickers_key_really_is_absent():
    """Si alguien la reintroduce, el test de arriba deja de tener sentido."""
    config = yaml.safe_load((ROOT / "src" / "config" / "base.yaml").read_text())
    assert "tickers" not in config["universe"], (
        "universe.tickers volvió a base.yaml: revisar los fallbacks de los DAGs"
    )

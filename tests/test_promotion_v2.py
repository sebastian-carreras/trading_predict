"""Tests de la regla de promoción v2 (compuertas + métrica primaria) y su modo sombra."""

from __future__ import annotations

import math

import pytest

from src.lifecycle.promotion import (
    compare_candidate_vs_champion,
    compare_v2,
    compute_score,
    compute_score_v2,
    evaluate_gates,
    get_v2_config,
)

_V1_WEIGHTS = {
    "bt_sharpe": 0.35,
    "ml_ic": 0.25,
    "ml_directional_accuracy": 0.20,
    "bt_calmar": 0.20,
}


def _metrics(**over) -> dict:
    base = {
        "bt_sharpe": 0.8,
        "ml_ic": 0.05,
        "ml_directional_accuracy": 0.60,
        "ml_dir_acc_edge": 0.02,
        "bt_calmar": 0.5,
        "bt_sharpe_excess": 0.30,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------
# No-regresión de la regla vieja
# ---------------------------------------------------------------
def test_compute_score_v1_no_cambio():
    """La v1 debe seguir dando exactamente lo mismo: v2 es aditiva, no la reemplaza aún."""
    m = {"bt_sharpe": 1.0, "ml_ic": 0.1, "ml_directional_accuracy": 0.6, "bt_calmar": 0.8}
    score, _ = compute_score(m, _V1_WEIGHTS)
    assert score == pytest.approx(0.35 * 1.0 + 0.25 * 0.1 + 0.20 * 0.6 + 0.20 * 0.8)


def test_v2_no_altera_la_decision_v1():
    """La sombra se calcula pero no toca should_promote ni los scores v1."""
    cand = _metrics(bt_sharpe_excess=5.0)          # v2 excelente
    champ = _metrics(bt_sharpe=0.79, bt_sharpe_excess=0.1)
    decision = compare_candidate_vs_champion(cand, champ, None)

    expected, _ = compute_score(cand, _V1_WEIGHTS)
    assert decision.candidate_score == pytest.approx(round(expected, 6))
    assert decision.would_promote_v2 is True      # la sombra opina
    assert decision.score_v2_candidate == 5.0


# ---------------------------------------------------------------
# Compuertas
# ---------------------------------------------------------------
def test_gates_pasan_con_metricas_sanas():
    passed, failures = evaluate_gates(_metrics(), get_v2_config(None)["gates"])
    assert passed and failures == []


@pytest.mark.parametrize("field", ["bt_sharpe", "ml_ic", "ml_dir_acc_edge"])
def test_cada_compuerta_rechaza_por_separado(field):
    passed, failures = evaluate_gates(_metrics(**{field: -0.01}), get_v2_config(None)["gates"])
    assert not passed
    assert any(f.startswith(field) for f in failures)


def test_gate_rechaza_el_empate():
    """Un edge de exactamente 0 (predictor 'siempre sube') no pasa: el umbral es estricto."""
    passed, failures = evaluate_gates(_metrics(ml_dir_acc_edge=0.0), get_v2_config(None)["gates"])
    assert not passed
    assert any("ml_dir_acc_edge" in f for f in failures)


def test_gate_rechaza_metrica_ausente_o_no_finita():
    gates = get_v2_config(None)["gates"]
    sin_ic = {k: v for k, v in _metrics().items() if k != "ml_ic"}
    assert not evaluate_gates(sin_ic, gates)[0]
    assert not evaluate_gates(_metrics(ml_ic=float("nan")), gates)[0]


def test_excess_alto_no_salva_compuertas_rotas():
    """El punto de las compuertas: un Sharpe excess alto con IC negativo se rechaza."""
    score, passed, failures = compute_score_v2(_metrics(ml_ic=-0.2, bt_sharpe_excess=9.9))
    assert score == 9.9          # la métrica primaria es alta...
    assert not passed            # ...pero no compite
    assert any("ml_ic" in f for f in failures)


# ---------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------
def test_promueve_si_mejora_y_pasa_compuertas():
    result = compare_v2(_metrics(bt_sharpe_excess=0.9), _metrics(bt_sharpe_excess=0.3))
    assert result["would_promote"] is True
    assert result["improvement_abs"] == pytest.approx(0.6)


def test_no_promueve_si_empata():
    """Δ=0 nunca promueve, ni con min_improvement_abs=0. Evita churn por empates."""
    result = compare_v2(_metrics(bt_sharpe_excess=0.3), _metrics(bt_sharpe_excess=0.3))
    assert result["would_promote"] is False
    assert result["improvement_abs"] == pytest.approx(0.0)


def test_no_promueve_si_empeora():
    assert compare_v2(_metrics(bt_sharpe_excess=0.1), _metrics(bt_sharpe_excess=0.3))["would_promote"] is False


def test_respeta_min_improvement_abs():
    cfg = {"v2": {"min_improvement_abs": 0.25}}
    apenas = compare_v2(_metrics(bt_sharpe_excess=0.4), _metrics(bt_sharpe_excess=0.3), cfg)
    holgado = compare_v2(_metrics(bt_sharpe_excess=0.6), _metrics(bt_sharpe_excess=0.3), cfg)
    assert apenas["would_promote"] is False       # +0.10 < 0.25
    assert holgado["would_promote"] is True       # +0.30 >= 0.25


def test_modelo_pre_v2_no_es_evaluable():
    """Sin la métrica primaria (modelos entrenados antes del cambio) no se inventa nada."""
    viejo = {"bt_sharpe": 1.0, "ml_ic": 0.1, "ml_directional_accuracy": 0.6}
    assert compare_v2(viejo, _metrics())["would_promote"] is False
    assert compare_v2(viejo, _metrics())["candidate_score"] is None

    # Candidato sano pero champion viejo: no hay contra qué comparar.
    sin_champ = compare_v2(_metrics(), viejo)
    assert sin_champ["would_promote"] is False
    assert sin_champ["champion_score"] is None
    assert "champion" in sin_champ["reason"]


def test_config_v2_hace_merge_parcial_de_gates():
    cfg = get_v2_config({"v2": {"gates": {"ml_ic": 0.05}}})
    assert cfg["gates"]["ml_ic"] == 0.05           # override
    assert cfg["gates"]["bt_sharpe"] == 0.0        # default preservado
    assert cfg["primary_metric"] == "bt_sharpe_excess"


def test_score_v2_nan_si_metrica_no_finita():
    score, _, _ = compute_score_v2(_metrics(bt_sharpe_excess=float("inf")))
    assert math.isnan(score)

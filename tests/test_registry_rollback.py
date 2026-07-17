"""Tests for ModelRegistry.restore_champion_from_retired (promotion rollback)."""

from __future__ import annotations

import json

from src.lifecycle.registry import ModelRegistry


def _registry_with_history(tmp_path):
    """A registry where a GOOD champion was retired and a WORSE one promoted."""
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register_baseline("e1", "AAA.BA", "runs/base", {"bt_sharpe": 0.5}, "e1_baseline")
    # Good model becomes champion...
    reg.register_candidate("e1", "AAA.BA", "runs/good", {"bt_sharpe": 1.6}, "e1_conservative")
    reg.promote_to_champion("e1", "AAA.BA", reason="good")
    # ...then a worse candidate is (wrongly) promoted over it.
    reg.register_candidate("e1", "AAA.BA", "runs/bad", {"bt_sharpe": 0.1}, "e1_conservative")
    reg.promote_to_champion("e1", "AAA.BA", reason="forced")
    return reg


class TestRestoreChampionFromRetired:
    def test_restores_retired_and_demotes_current(self, tmp_path) -> None:
        reg = _registry_with_history(tmp_path)
        # Sanity: the bad model is champion, the good one is retired[0].
        assert reg.get_champion("e1", "AAA.BA")["metrics"]["bt_sharpe"] == 0.1
        assert reg.data["strategies"]["e1"]["tickers"]["AAA.BA"]["retired"][0]["metrics"]["bt_sharpe"] == 1.6

        restored = reg.restore_champion_from_retired("e1", "AAA.BA")

        assert restored is not None
        assert restored["metrics"]["bt_sharpe"] == 1.6
        # Champion is now the good model again.
        assert reg.get_champion("e1", "AAA.BA")["metrics"]["bt_sharpe"] == 1.6
        # The bad model was demoted to the front of retired, tagged with the reason.
        retired = reg.data["strategies"]["e1"]["tickers"]["AAA.BA"]["retired"]
        assert retired[0]["metrics"]["bt_sharpe"] == 0.1
        assert retired[0]["reason"] == "rollback_restore"
        assert "retired_at" in retired[0]

    def test_restored_model_has_no_retirement_fields(self, tmp_path) -> None:
        reg = _registry_with_history(tmp_path)
        restored = reg.restore_champion_from_retired("e1", "AAA.BA")
        assert "retired_at" not in restored
        assert "reason" not in restored

    def test_persisted_to_disk(self, tmp_path) -> None:
        reg = _registry_with_history(tmp_path)
        reg.restore_champion_from_retired("e1", "AAA.BA")
        on_disk = json.loads((tmp_path / "registry.json").read_text())
        champ = on_disk["strategies"]["e1"]["tickers"]["AAA.BA"]["champion"]
        assert champ["metrics"]["bt_sharpe"] == 1.6

    def test_none_when_no_retired(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register_candidate("e1", "X", "runs/x", {"bt_sharpe": 1.0}, "e1_conservative")
        reg.promote_to_champion("e1", "X", reason="first")
        assert reg.restore_champion_from_retired("e1", "X") is None

    def test_none_for_unknown_ticker(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        assert reg.restore_champion_from_retired("e1", "NOPE") is None

    def test_custom_reason(self, tmp_path) -> None:
        reg = _registry_with_history(tmp_path)
        reg.restore_champion_from_retired("e1", "AAA.BA", reason="undo_20260715_force")
        retired = reg.data["strategies"]["e1"]["tickers"]["AAA.BA"]["retired"]
        assert retired[0]["reason"] == "undo_20260715_force"

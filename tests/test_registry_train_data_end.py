"""Tests for ModelRegistry.set_train_data_end (backfill for old champions)."""

from __future__ import annotations

import json

from src.lifecycle.registry import ModelRegistry


class TestSetTrainDataEnd:
    def test_sets_field_on_champion(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register_candidate("e1", "AAA.BA", "runs/good", {"bt_sharpe": 1.6}, "e1_conservative")
        reg.promote_to_champion("e1", "AAA.BA", reason="first")

        updated = reg.set_train_data_end("e1", "AAA.BA", "2026-02-17")

        assert updated is True
        assert reg.get_champion("e1", "AAA.BA")["train_data_end"] == "2026-02-17"

    def test_persisted_to_disk(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register_candidate("e1", "AAA.BA", "runs/good", {"bt_sharpe": 1.6}, "e1_conservative")
        reg.promote_to_champion("e1", "AAA.BA", reason="first")
        reg.set_train_data_end("e1", "AAA.BA", "2026-02-17")

        on_disk = json.loads((tmp_path / "registry.json").read_text())
        champ = on_disk["strategies"]["e1"]["tickers"]["AAA.BA"]["champion"]
        assert champ["train_data_end"] == "2026-02-17"

    def test_does_not_overwrite_existing_value_by_default_caller_choice(self, tmp_path) -> None:
        """The method itself always overwrites; callers decide to skip populated ones."""
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register_candidate(
            "e1", "AAA.BA", "runs/good", {"bt_sharpe": 1.6}, "e1_conservative",
            train_data_end="2026-07-15",
        )
        reg.promote_to_champion("e1", "AAA.BA", reason="first")

        reg.set_train_data_end("e1", "AAA.BA", "2026-02-17")

        assert reg.get_champion("e1", "AAA.BA")["train_data_end"] == "2026-02-17"

    def test_false_when_no_champion(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        reg.register_candidate("e1", "AAA.BA", "runs/good", {"bt_sharpe": 1.6}, "e1_conservative")
        # candidate exists but not promoted yet -> no champion
        assert reg.set_train_data_end("e1", "AAA.BA", "2026-02-17") is False

    def test_false_for_unknown_ticker(self, tmp_path) -> None:
        reg = ModelRegistry(tmp_path / "registry.json")
        assert reg.set_train_data_end("e1", "NOPE", "2026-02-17") is False

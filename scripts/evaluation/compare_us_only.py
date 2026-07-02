"""Comparación campeón vs baseline restringida a acciones de EEUU.

Replica la metodología y el estilo de figuras de compare_e1_models / compare_e2_models
(promedio simple de cada métrica entre tickers, vía _compare_common.compare_runs), pero
filtra el universo a las acciones estadounidenses de cada estrategia. El universo por
estrategia se toma de src/config/base.yaml::universe.tickers_by_strategy, de modo que solo
entran los tickers oficiales sin sufijo de mercado argentino (.BA).

No reentrena: lee los campeones/baselines ya promovidos en models/registry.json.

Salidas (en reports/conclusiones/):
    comparaciones/fig_e1_vs_baseline_us*.png  — figuras ML y trading (solo EEUU)
    comparaciones/fig_e1_vs_baseline_us.md
    comparaciones/e1_comparison_us.csv
    tablas/e1_champions_us_per_ticker.csv      — métricas del campeón por ticker
    (ídem e2)

Uso:
    python -m scripts.evaluation.compare_us_only
"""

from __future__ import annotations

from pathlib import Path

from src.utils import get_nested, load_yaml, project_root

from scripts.evaluation._compare_common import (
    compare_runs,
    load_champions_summary,
    print_comparison,
    save_comparison_figure,
    save_comparison_markdown,
    validate_improvement_signs,
)

ROOT = project_root()
OUT_COMP = ROOT / "reports" / "conclusiones" / "comparaciones"
OUT_TAB = ROOT / "reports" / "conclusiones" / "tablas"

PER_TICKER_COLS = [
    "ticker", "variant", "ml_directional_accuracy", "ml_ic", "bt_sharpe",
    "bt_sortino", "bt_cagr", "bt_max_drawdown", "bt_calmar", "bt_profit_factor",
    "bt_hit_rate", "_run_dir",
]

# (registry_key, universe_key, model_label, strategy_label, figsize_overrides)
STRATEGIES = [
    ("e1", "e1_conservative", "GRU E1", "E1 Conservative (GRU 90-day, EEUU)", None),
    ("e2", "e2_moderate", "LSTM E2", "E2 Moderate (LSTM 20-day, EEUU)", {"calidad": (8, 5.4)}),
]


def is_us(ticker: str) -> bool:
    """Acción de EEUU = sin sufijo de mercado argentino (.BA)."""
    return not ticker.endswith(".BA")


def main() -> None:
    OUT_COMP.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(ROOT / "src" / "config" / "base.yaml")

    for reg_key, univ_key, model_label, strategy_label, figsize_overrides in STRATEGIES:
        official = get_nested(cfg, ["universe", "tickers_by_strategy", univ_key], []) or []
        us_tickers = sorted({t for t in official if is_us(t)})

        df_base = load_champions_summary(reg_key, "baseline", ROOT)
        df_champ = load_champions_summary(reg_key, "champion", ROOT)
        if df_base is None or df_champ is None:
            print(f"[WARN] {reg_key}: faltan datos en el registry; se omite.")
            continue

        df_base_us = df_base[df_base["ticker"].isin(us_tickers)].copy()
        df_champ_us = df_champ[df_champ["ticker"].isin(us_tickers)].copy()

        print("\n" + "=" * 78)
        print(f"{strategy_label}")
        print("=" * 78)
        print(f"Universo oficial EEUU ({len(us_tickers)}): {us_tickers}")
        print(f"Campeones EEUU encontrados: {sorted(df_champ_us['ticker'])}")
        print(f"Baselines EEUU encontrados: {sorted(df_base_us['ticker'])}")

        # Comparación agregada (media de tickers) campeón vs baseline.
        df_comp = compare_runs(df_base_us, df_champ_us, model_label=model_label)
        validate_improvement_signs(df_comp)
        print_comparison(df_comp, model_label)

        out_png = OUT_COMP / f"fig_{reg_key}_vs_baseline_us.png"
        save_comparison_figure(
            df_comp,
            model_label=model_label,
            strategy_label=strategy_label,
            out_path=out_png,
            trading_split_per_group=True,
            figsize_overrides=figsize_overrides,
        )
        save_comparison_markdown(
            df_comp,
            model_label=model_label,
            strategy_label=strategy_label,
            out_path=OUT_COMP / f"fig_{reg_key}_vs_baseline_us.md",
        )

        out_csv = OUT_COMP / f"{reg_key}_comparison_us.csv"
        df_comp.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

        # Campeones por ticker (solo EEUU).
        cols = [c for c in PER_TICKER_COLS if c in df_champ_us.columns]
        out_per_ticker = OUT_TAB / f"{reg_key}_champions_us_per_ticker.csv"
        df_champ_us[cols].sort_values("ticker").to_csv(out_per_ticker, index=False)
        print(f"[OK] {out_per_ticker}")


if __name__ == "__main__":
    main()

"""Backfill de la regla v2 sobre las decisiones de promoción ya registradas.

Permite comparar la regla vieja (suma ponderada de 4 métricas en escala cruda) contra la
nueva (compuertas + ``bt_sharpe_excess``) sobre el histórico completo, sin esperar días de
corridas nuevas y sin tocar ningún champion.

Cómo se reconstruyen las métricas
---------------------------------
``promotion_log.jsonl`` no guarda las métricas crudas, pero sí sus **contribuciones
ponderadas** en ``detail``::

    contribution = (w / Σw) · val    →    val = contribution · Σw / w

Con los pesos de la estrategia (los mismos de ``base.yaml``) se despeja ``val``. Las dos
métricas que la v2 necesita y que no existían entonces se recomputan desde los precios:

- ``bt_sharpe_excess`` = ``bt_sharpe`` recuperado − Sharpe de buy & hold en la ventana.
- ``ml_dir_acc_edge``  = ``ml_directional_accuracy`` recuperada − tasa de subida en la ventana.

Limitaciones (declaradas, no disimuladas)
-----------------------------------------
- Solo las entradas ``fair_window`` tienen una ventana OOS bien definida y se reconstruyen
  completas. Las ``stored`` comparan métricas de ventanas distintas, así que su
  ``sharpe_excess`` no es reconstruible desde el log y quedan marcadas como parciales.
- ``insufficient_evidence`` / ``identity_skip`` / entradas viejas sin modo no tuvieron
  comparación: no hay nada que recalcular.
- La ventana se aproxima como *todas las fechas de datos limpios entre start y end*. La
  original era la intersección con las fechas predecibles por ambos modelos, así que puede
  diferir en unas pocas barras. Suficiente para comparar reglas, no para reportar métricas.
- El ticker ``AAA`` es un fixture de test (49 de las 113 promociones del log) y se excluye.

Uso:
    PYTHONPATH=. python -m scripts.evaluation.backfill_score_v2
    PYTHONPATH=. python -m scripts.evaluation.backfill_score_v2 --include-aaa
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.lifecycle.promotion import compare_v2, compute_score, get_strategy_promotion_config
from src.metrics.skill import buy_and_hold_sharpe
from src.utils import ensure_dir, get_nested, load_yaml, project_root

_EXCLUDED_TICKERS = {"AAA"}
_HORIZON_BY_STRATEGY = {"e1": 90, "e2": 20, "e3": 6}
_FIELDS = [
    "timestamp", "strategy", "ticker", "comparison_mode", "n_eval",
    "v1_should_promote", "v1_improvement_pct",
    "v2_would_promote", "v2_gates_passed", "v2_cand", "v2_champ", "v2_delta",
    "agree", "v2_reason",
]


def _load_ohlcv(ticker: str, root: Path) -> pd.DataFrame | None:
    """Carga OHLCV limpio. Los CSV de .BA mezclan timestamps con y sin fracción de
    segundo, así que `format="mixed"` es obligatorio: sin él se pierde el 87% de las filas."""
    path = root / "data" / "clean" / f"{ticker}_daily.csv"
    if not path.exists():
        path = root / "data" / "raw" / "daily" / f"{ticker}_daily.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"])


def _recover_raw_metrics(detail: dict, prefix: str, weights: dict[str, float]) -> dict[str, float]:
    """Despeja las métricas crudas de las contribuciones ponderadas guardadas en el log."""
    total = sum(weights.values())
    out: dict[str, float] = {}
    for key, w in weights.items():
        contribution = detail.get(f"{prefix}_{key}")
        if contribution is None or w == 0:
            continue
        out[key] = float(contribution) * total / w
    return out


def _window_baselines(
    ohlcv: pd.DataFrame, start: str, end: str, horizon: int,
) -> tuple[float, float, int]:
    """(sharpe buy&hold, tasa de subida, n) sobre la ventana [start, end]."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    window = ohlcv.loc[(ohlcv.index >= lo) & (ohlcv.index < hi)]
    if len(window) < 2:
        return float("nan"), float("nan"), len(window)

    close = window["close"].to_numpy()
    bh = buy_and_hold_sharpe(close)

    # Tasa de subida del retorno forward al horizonte, sobre la serie completa para no
    # perder las últimas `horizon` barras de la ventana.
    full = ohlcv["close"]
    fwd = np.log(full.shift(-horizon) / full).reindex(window.index).dropna()
    up = float((fwd > 0).mean()) if len(fwd) else float("nan")
    return bh, up, len(window)


def run(log_path: Path, out_path: Path, include_aaa: bool) -> None:
    root = project_root()
    cfg = load_yaml(root / "src" / "config" / "base.yaml")
    promo_cfg = get_nested(cfg, ["lifecycle", "promotion"], default={}) or {}

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    print(f"Entradas en el log: {len(entries)}")

    ohlcv_cache: dict[str, pd.DataFrame | None] = {}
    rows: list[list] = []
    skipped: dict[str, int] = {}
    recon_errors: list[float] = []

    for e in entries:
        strategy = e.get("strategy", "")
        ticker = e.get("ticker", "")
        mode = e.get("comparison_mode") or "none"

        if ticker in _EXCLUDED_TICKERS and not include_aaa:
            skipped["ticker excluido (fixture)"] = skipped.get("ticker excluido (fixture)", 0) + 1
            continue
        if mode not in ("fair_window", "stored"):
            skipped[f"sin comparación ({mode})"] = skipped.get(f"sin comparación ({mode})", 0) + 1
            continue

        detail = e.get("detail") or {}
        weights = get_strategy_promotion_config(promo_cfg, strategy).get("scoring_weights", {})
        cand = _recover_raw_metrics(detail, "candidate", weights)
        champ = _recover_raw_metrics(detail, "champion", weights)
        if not cand or not champ:
            skipped["detail incompleto"] = skipped.get("detail incompleto", 0) + 1
            continue

        # Validación de la inversión: recomputar el score v1 desde las métricas
        # recuperadas debe reproducir el que quedó registrado.
        v1_recomputed, _ = compute_score(cand, weights)
        logged = e.get("candidate_score")
        if isinstance(logged, (int, float)):
            recon_errors.append(abs(v1_recomputed - float(logged)))

        start, end = e.get("eval_window_start"), e.get("eval_window_end")
        n_eval = e.get("eval_n_samples")
        v1_promote = bool(e.get("should_promote"))

        if mode != "fair_window" or not (start and end):
            # `stored` compara métricas de ventanas DISTINTAS: no hay una ventana común
            # contra la cual medir el exceso sobre buy & hold, así que la v2 no es
            # evaluable. Se registra la fila sin decisión, para no inflar los agregados
            # con reconstrucciones inventadas.
            rows.append([
                e.get("timestamp", ""), strategy, ticker, mode, n_eval,
                v1_promote, e.get("improvement_pct"),
                None, None, None, None, None, None,
                "v2 no evaluable: modo `stored`, sin ventana OOS común",
            ])
            continue

        # Las dos métricas que la v2 necesita y el log no tenía.
        if ticker not in ohlcv_cache:
            ohlcv_cache[ticker] = _load_ohlcv(ticker, root)
        ohlcv = ohlcv_cache[ticker]
        if ohlcv is None:
            skipped["sin OHLCV"] = skipped.get("sin OHLCV", 0) + 1
            continue
        horizon = _HORIZON_BY_STRATEGY.get(strategy, 90)
        bh, up_rate, _ = _window_baselines(ohlcv, start, end, horizon)
        for m in (cand, champ):
            m["bt_sharpe_excess"] = m.get("bt_sharpe", float("nan")) - bh
            m["ml_dir_acc_edge"] = m.get("ml_directional_accuracy", float("nan")) - up_rate

        v2 = compare_v2(cand, champ, promo_cfg)
        v2_promote = v2.get("would_promote")

        rows.append([
            e.get("timestamp", ""), strategy, ticker, mode, n_eval,
            v1_promote, e.get("improvement_pct"),
            v2_promote, v2.get("gates_passed"),
            v2.get("candidate_score"), v2.get("champion_score"), v2.get("improvement_abs"),
            (v1_promote == v2_promote) if v2_promote is not None else None,
            v2.get("reason", ""),
        ])

    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_FIELDS)
        w.writerows(rows)

    if recon_errors:
        worst = max(recon_errors)
        status = "OK" if worst < 1e-6 else "REVISAR"
        print(f"\nValidación de la inversión (score v1 recomputado vs registrado): "
              f"error máx={worst:.2e} sobre {len(recon_errors)} entradas → {status}")

    _summary(rows, skipped, out_path)


def _summary(rows: list[list], skipped: dict[str, int], out_path: Path) -> None:
    print("\nOmitidas:")
    for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {reason}")

    if not rows:
        print("\nNada reconstruido.")
        return

    df = pd.DataFrame(rows, columns=_FIELDS)
    print(f"\nReconstruidas: {len(df)}")
    for mode, grp in df.groupby("comparison_mode"):
        evaluable = grp[grp.v2_would_promote.notna()]
        print(f"\n  {mode} (n={len(grp)}, v2 evaluable={len(evaluable)})")
        if evaluable.empty:
            continue
        agree = evaluable.agree.sum()
        print(f"    v1 promueve: {evaluable.v1_should_promote.sum():3d}  |  "
              f"v2 promovería: {evaluable.v2_would_promote.sum():3d}  |  "
              f"coinciden: {agree}/{len(evaluable)} ({agree / len(evaluable):.0%})")
        flip_no = evaluable[(evaluable.v1_should_promote) & (~evaluable.v2_would_promote.astype(bool))]
        flip_si = evaluable[(~evaluable.v1_should_promote) & (evaluable.v2_would_promote.astype(bool))]
        print(f"    v1 promovía y v2 NO: {len(flip_no)}  |  v1 no promovía y v2 SÍ: {len(flip_si)}")
        gates_fail = evaluable[~evaluable.v2_gates_passed.astype(bool)]
        print(f"    rechazadas por compuertas v2: {len(gates_fail)}")

    print(f"\nCSV: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill de la regla v2 sobre promotion_log.jsonl")
    ap.add_argument("--log", type=str, default="models/promotion_log.jsonl")
    ap.add_argument("--out", type=str, default="reports/promotion/score_v2_backfill.csv")
    ap.add_argument("--include-aaa", action="store_true",
                    help="incluir el ticker fixture AAA (por defecto se excluye)")
    args = ap.parse_args()

    root = project_root()
    log = Path(args.log)
    out = Path(args.out)
    if not log.is_absolute():
        log = root / log
    if not out.is_absolute():
        out = root / out

    if not log.exists():
        raise SystemExit(f"No existe el log: {log}")
    run(log, out, args.include_aaa)


if __name__ == "__main__":
    main()

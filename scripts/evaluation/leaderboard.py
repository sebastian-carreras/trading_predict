"""Leaderboard diario: "mejores tickers para invertir" según el champion actual.

La predicción de cada fila es **inferencia en vivo**: se corre el champion
congelado sobre los datos más recientes (``src.lifecycle.signals``), sin
re-entrenar y sin leer el CSV walk-forward del run — ese CSV es una predicción
de la fecha en que se entrenó el champion, no una señal de hoy. La columna
``AS_OF`` dice sobre qué día de datos se calculó cada predicción.

Rankea los champions del registry por el MISMO score compuesto que usa la
promoción (``src.lifecycle.promotion.compute_score`` con los pesos por estrategia),
de modo que el ranking es coherente con las decisiones de lifecycle.

Además compara contra el snapshot del día anterior (``leaderboard_history.jsonl``)
para responder "¿los champions están mejorando o empeorando con el paso de los
días?": muestra el delta de score y el cambio de posición (▲/▼) por ticker.

Lee:
    models/registry.json                     (champions y sus métricas)
    src/config/base.yaml                     (scoring_weights por estrategia)
    reports/dashboard/leaderboard_history.jsonl  (snapshots previos, para el delta)

Escribe:
    reports/dashboard/leaderboard_<YYYYMMDD>.csv   (snapshot del día)
    reports/dashboard/leaderboard_latest.csv       (alias al último)
    reports/dashboard/leaderboard_history.jsonl    (append: 1 fila por champion/día)

Uso:
    python -m scripts.evaluation.leaderboard
    python -m scripts.evaluation.leaderboard --strategies e1,e2 --top 10
    python -m scripts.evaluation.leaderboard --no-write   # solo imprime, no persiste
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.lifecycle.registry import ModelRegistry
from src.lifecycle.promotion import compute_score, get_strategy_promotion_config
from src.lifecycle import signals
from src.utils import load_yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"
CONFIG_PATH = ROOT / "src" / "config" / "base.yaml"
DASHBOARD_DIR = ROOT / "reports" / "dashboard"
HISTORY_PATH = DASHBOARD_DIR / "leaderboard_history.jsonl"

DEFAULT_STRATEGIES = ["e1", "e2"]

# Un champion se marca "viejo" (⚠) si su último día de entrenamiento supera esto.
# Ojo: es distinto de la frescura de los DATOS de la predicción, que la controla
# ``reporting.max_signal_age_days`` y se muestra en DAGE.
STALE_DAYS = 10

# Columnas de métricas que se muestran junto al score (higher-is-better salvo DD).
DISPLAY_METRICS = [
    "bt_sharpe",
    "ml_ic",
    "ml_directional_accuracy",
    "bt_calmar",
    "bt_max_drawdown",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_weights(config: dict[str, Any], strategy: str) -> dict[str, float]:
    """Pesos de scoring por estrategia, idénticos a los que usa la promoción."""
    promo_cfg = config.get("lifecycle", {}).get("promotion", {})
    resolved = get_strategy_promotion_config(promo_cfg, strategy)
    return resolved.get("scoring_weights", {})


def _run_dir_date(run_dir: str | None) -> str | None:
    """Extrae la fecha del timestamp del run_dir (``.../YYYYMMDD_HHMMSS/...``)."""
    import re
    m = re.search(r"(\d{8})_\d{6}", str(run_dir or ""))
    if not m:
        return None
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def champion_data_cutoff(champ: dict[str, Any]) -> str | None:
    """Último día de datos que el champion vio al entrenarse (ISO date).

    ``train_data_end`` (si el registry lo trackea) o, si falta (champions
    viejos), la fecha embebida en ``run_dir`` como proxy — verificado contra
    los champions que trackean ambos campos: coinciden (el training corre
    sobre datos frescos del mismo día). NO se usa la última predicción del
    walk-forward: esa es la fecha de la última *señal*, que cae `horizonte`
    días ANTES del corte real (no es la fecha de corte).
    """
    return champ.get("train_data_end") or _run_dir_date(champ.get("run_dir"))


def _forecast_and_freshness(
    strategy: str,
    ticker: str,
    registry: ModelRegistry,
    champ: dict[str, Any],
    config: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Señal de hoy del champion (inferencia en vivo) + frescura del modelo.

    La predicción sale de ``signals.champion_signal``: el champion congelado
    corrido sobre los datos más recientes. Las columnas de frescura son dos y
    miden cosas distintas — ``days_ago``/``stale`` es qué tan viejo es el
    ENTRENAMIENTO del champion; ``data_age_days``/``stale_data`` es qué tan
    viejos son los DATOS sobre los que predijo.
    """
    signal_row = signals.champion_signal(
        strategy, ticker, registry, config, root=ROOT
    ) or signals.unavailable_signal()

    trained_at = champion_data_cutoff(champ)
    days_ago = None
    if trained_at:
        try:
            d = datetime.fromisoformat(str(trained_at)[:10]).date()
            days_ago = (datetime.now(timezone.utc).date() - d).days
        except ValueError:
            days_ago = None
    stale = bool(days_ago is not None and days_ago > STALE_DAYS)

    recent = champ.get("recent_metrics")
    recent_score = None
    if isinstance(recent, dict) and recent:
        rs, _ = compute_score(recent, weights)
        recent_score = round(float(rs), 6)

    return {
        **signal_row,
        "trained_at": str(trained_at)[:10] if trained_at else None,
        "days_ago": days_ago,
        "stale": stale,
        "recent_score": recent_score,
    }


def _normalize(s: pd.Series) -> pd.Series:
    """Min-max a [0,1]; NaN → 0; constante → 0.5."""
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return x.notna().astype(float) * 0.5
    return ((x - lo) / (hi - lo)).fillna(0.0)


def build_leaderboard(
    registry: ModelRegistry,
    config: dict[str, Any],
    strategies: list[str],
    rank_by: str = "score",
) -> pd.DataFrame:
    """Construye el ranking de champions.

    ``rank_by``: ``score`` (calidad histórica de entrenamiento, default),
    ``pred`` (mayor retorno anualizado predicho para hoy) o ``blend``
    (normalizado score + pred).
    """
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        weights = _resolve_weights(config, strategy)
        champions = registry.list_all(strategy=strategy, stage="champion")
        for champ in champions:
            metrics = champ.get("metrics", {}) or {}
            score, _detail = compute_score(metrics, weights)
            variant = champ.get("variant")
            ticker = champ.get("ticker")
            row: dict[str, Any] = {
                "strategy": strategy,
                "ticker": ticker,
                "variant": variant,
                "score": round(float(score), 6),
                "promoted_at": champ.get("promoted_at"),
                "run_dir": champ.get("run_dir"),
            }
            for m in DISPLAY_METRICS:
                val = metrics.get(m)
                row[m] = round(float(val), 6) if isinstance(val, (int, float)) else None
            row.update(
                _forecast_and_freshness(strategy, ticker, registry, champ, config, weights)
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if rank_by == "pred":
        df["_sort"] = pd.to_numeric(df["pred_annualized"], errors="coerce")
    elif rank_by == "blend":
        df["_sort"] = _normalize(df["score"]) + _normalize(df["pred_annualized"])
    else:
        df["_sort"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.sort_values("_sort", ascending=False, na_position="last", ignore_index=True)
    df = df.drop(columns="_sort")
    df.insert(0, "rank", df.index + 1)
    return df


def _load_previous_snapshot() -> dict[tuple[str, str], dict[str, Any]]:
    """Último snapshot por (strategy, ticker) de fechas ANTERIORES a hoy.

    Devuelve {(strategy, ticker): {"score":..., "rank":..., "date":...}} tomando,
    para cada ticker, el registro más reciente cuya fecha sea distinta a la de hoy.
    """
    if not HISTORY_PATH.exists():
        return {}
    today = datetime.now(timezone.utc).date().isoformat()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("date") == today:
                continue  # ignorar re-corridas del mismo día para el delta
            key = (rec.get("strategy"), rec.get("ticker"))
            prev = latest.get(key)
            if prev is None or str(rec.get("date", "")) >= str(prev.get("date", "")):
                latest[key] = rec
    return latest


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega delta de score y de rank contra el último snapshot previo."""
    if df.empty:
        return df
    prev = _load_previous_snapshot()
    d_scores, d_ranks, trends = [], [], []
    for _, r in df.iterrows():
        key = (r["strategy"], r["ticker"])
        p = prev.get(key)
        if p is None:
            d_scores.append(None)
            d_ranks.append(None)
            trends.append("new")
            continue
        ds = round(float(r["score"]) - float(p.get("score", 0.0)), 6)
        # rank previo: rank menor = mejor, por eso delta_rank = prev_rank - rank_hoy
        dr = (int(p["rank"]) - int(r["rank"])) if p.get("rank") is not None else None
        d_scores.append(ds)
        d_ranks.append(dr)
        if ds > 1e-9:
            trends.append("up")
        elif ds < -1e-9:
            trends.append("down")
        else:
            trends.append("flat")
    df = df.copy()
    df["delta_score"] = d_scores
    df["delta_rank"] = d_ranks
    df["trend"] = trends
    return df


def _append_history(df: pd.DataFrame) -> None:
    """Append 1 línea por champion al history (para el delta de días siguientes)."""
    date = datetime.now(timezone.utc).date().isoformat()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        for _, r in df.iterrows():
            rec = {
                "date": date,
                "timestamp": _now_iso(),
                "strategy": r["strategy"],
                "ticker": r["ticker"],
                "rank": int(r["rank"]),
                "score": float(r["score"]),
                "variant": r.get("variant"),
                "pred_return": (float(r["pred_return"]) if pd.notna(r.get("pred_return")) else None),
                "pred_annualized": (float(r["pred_annualized"]) if pd.notna(r.get("pred_annualized")) else None),
                "signal": r.get("signal"),
                "trained_at": r.get("trained_at"),
                "as_of": r.get("as_of"),
                "data_age_days": (int(r["data_age_days"]) if pd.notna(r.get("data_age_days")) else None),
                "pred_mode": r.get("pred_mode"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(df: pd.DataFrame) -> Path:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    dated = DASHBOARD_DIR / f"leaderboard_{date}.csv"
    latest = DASHBOARD_DIR / "leaderboard_latest.csv"
    df.to_csv(dated, index=False)
    df.to_csv(latest, index=False)
    return dated


_ARROW = {"up": "▲", "down": "▼", "flat": "=", "new": "★"}


def _pct(x: Any) -> str:
    return f"{float(x) * 100:+.1f}%" if pd.notna(x) else "   —"


def print_leaderboard(df: pd.DataFrame, top: int | None, rank_by: str = "score") -> None:
    if df.empty:
        print("Leaderboard vacío: no hay champions en el registry para las estrategias pedidas.")
        return
    view = df.head(top) if top else df
    W = 112
    print("=" * W)
    print(f"  LEADERBOARD — qué invertir hoy  ({datetime.now():%Y-%m-%d %H:%M})   ordenado por: {rank_by}")
    print(f"  DE HOY: PRED% = retorno al horizonte que el champion congelado predice sobre los datos más")
    print(f"  recientes (inferencia en vivo, sin re-entrenar) · ANN% = anualizado · AS_OF = día de datos")
    print(f"  usado · DAGE = su antigüedad (⚠ = pasó el umbral, no se emite señal)")
    print(f"  HISTÓRICO: SCORE/SHARPE/IC = calidad del champion en su ENTRENAMIENTO original, fijos, NO")
    print(f"  de hoy · TRAINED = antigüedad de ese entrenamiento (⚠ >{STALE_DAYS}d) · RECENT = score de la")
    print(f"  última reevaluación fair-window (— = nunca reevaluado contra un candidate)")
    print("=" * W)
    print(f"{'#':>2}  {'STRAT':<5} {'TICKER':<9} {'PRED%':>7} {'ANN%':>7} {'SIG':<4} {'AS_OF':>10} {'DAGE':>6} "
          f"{'SCORE':>7} {'RECENT':>7} {'TRAINED':>9} {'SHARPE':>7} {'IC':>6}")
    print("-" * W)
    for _, r in view.iterrows():
        arrow = _ARROW.get(r.get("trend", ""), " ")
        sig = str(r.get("signal") or "—")
        recent = r.get("recent_score")
        recent_s = f"{float(recent):.4f}" if pd.notna(recent) else "   —"
        days = r.get("days_ago")
        stale_mark = "⚠" if r.get("stale") else " "
        trained = f"{int(days)}d{stale_mark}" if pd.notna(days) else f"—{stale_mark}"
        as_of = str(r.get("as_of") or "—")
        dage = r.get("data_age_days")
        dage_s = f"{int(dage)}d{'⚠' if r.get('stale_data') else ' '}" if pd.notna(dage) else "  — "
        sharpe = r.get("bt_sharpe")
        ic = r.get("ml_ic")
        print(
            f"{int(r['rank']):>2}  {r['strategy']:<5} {str(r['ticker']):<9} "
            f"{_pct(r.get('pred_return')):>7} {_pct(r.get('pred_annualized')):>7} {sig:<4} {as_of:>10} {dage_s:>6} "
            f"{r['score']:>7.4f} {recent_s:>7} {trained:>9} "
            f"{(sharpe if pd.notna(sharpe) else float('nan')):>7.3f} "
            f"{(ic if pd.notna(ic) else float('nan')):>6.3f}{arrow}"
        )
    print("-" * W)
    n_stale = int(df["stale"].sum())
    n_stale_data = int(df["stale_data"].sum())
    n_unavailable = int((df["pred_mode"] == "unavailable").sum())
    n_long = int((df["signal"] == "LONG").sum())
    ups = int((df["trend"] == "up").sum())
    downs = int((df["trend"] == "down").sum())
    print(f"Señales LONG {n_long}/{len(df)}   ·   datos rancios ⚠ {n_stale_data}   ·   "
          f"sin inferencia {n_unavailable}   ·   champions viejos ⚠ {n_stale}   ·   "
          f"Δscore ▲{ups} ▼{downs} (vs snapshot previo)")
    if n_stale_data:
        print(f"  ⚠ {n_stale_data} ticker(s) con datos por encima de "
              f"reporting.max_signal_age_days: se muestra PRED% pero NO se emite señal. "
              f"Corré scripts/data/refresh_champion_data.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Leaderboard diario de champions.")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Estrategias a rankear, separadas por coma (default: e1,e2).",
    )
    parser.add_argument("--top", type=int, default=None, help="Mostrar solo top-N filas.")
    parser.add_argument(
        "--rank-by",
        choices=["score", "pred", "blend"],
        default="score",
        help="Ordenar por score de entrenamiento (default), pred (retorno anualizado "
             "predicho para hoy) o blend (score + pred normalizados).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Solo imprime; no escribe CSV ni history (útil para inspección).",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    config = load_yaml(CONFIG_PATH)
    registry = ModelRegistry(REGISTRY_PATH)

    df = build_leaderboard(registry, config, strategies, rank_by=args.rank_by)
    df = add_deltas(df)
    print_leaderboard(df, args.top, rank_by=args.rank_by)

    if not args.no_write and not df.empty:
        csv_path = _write_csv(df)
        _append_history(df)
        print(f"\nGuardado: {csv_path.relative_to(ROOT)}  (+ leaderboard_latest.csv, history.jsonl)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

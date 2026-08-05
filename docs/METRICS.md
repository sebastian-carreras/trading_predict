# Evaluation metrics

Every strategy is evaluated on two levels: how well the model **predicts** (offline ML metrics)
and how the resulting strategy **performs when traded** (online trading metrics). A model can
look good on the first set and still lose money on the second — which is exactly why both are
reported.

All backtests are **net of transaction costs** (10 bps round-trip for daily strategies, 20 bps
for intraday).

---

## ML metrics (offline)

### MAE / RMSE — prediction error
MAE is the mean absolute error and is robust to outliers. RMSE squares the errors, so it
penalizes large misses much harder. Reporting both shows whether the error is spread evenly or
driven by a few bad days.

### Directional accuracy — did we get the sign right?
The share of predictions where the model got the direction of the move right (up or down),
regardless of magnitude. This matters more than MAE for a trading system: being off by 2% but
right about the direction is tradeable; being precise about magnitude while wrong about the
sign is not.

> ⚠️ **The benchmark is not 50%.** Over a 90-day horizon a stock with positive drift rises
> 66–86% of the time, so a model that always predicts "up" scores that same number without
> any predictive ability. Read `ml_dir_acc_edge` (below), never raw directional accuracy.

### Directional accuracy edge — did we beat "always up"?
`ml_dir_acc_edge = ml_directional_accuracy − ml_naive_up_rate`, where `ml_naive_up_rate` is the
share of test dates whose realized forward return was positive. This is the drift-neutral
version: **0 means the model added nothing over assuming the asset always rises.**

The correction is not academic. Measured on the walk-forward runs of 2026-07-31:

| strategy | median directional accuracy | median naive "always up" | **median edge** | beat naive |
|---|---|---|---|---|
| E1 | 0.593 | 0.705 | **−0.127** | 1/10 |
| E2 | 0.534 | 0.585 | **−0.033** | 3/30 |

And in the champion-vs-candidate fair-window comparisons, 38 of 41 evaluations had an edge of
**exactly 0.0000** — an identity that can only hold if the model emits a positive prediction on
every single date. Individual folds show the same thing: GGAL.BA fold 3 scored a directional
accuracy of **1.000** with an IC of **0.000**.

### Pesaran-Timmermann test — is the edge distinguishable from chance?
`ml_pt_stat` / `ml_pt_pvalue`, from Pesaran & Timmermann (1992). Non-parametric test of
directional forecasting that compares the observed hit rate against the rate implied by the
*marginal* frequencies of predicted and realized signs — the same correction as the edge, but
with a distribution attached.

**Informational only, never a promotion gate.** The test assumes serial independence, and with
overlapping labels (E1 shares 89 of 90 days between consecutive samples) it is oversized: it
rejects the null more often than it should. Treat a small p-value as "worth a look", not proof.

### Information Coefficient (IC) — ranking quality
The Spearman correlation between predicted and realized returns. It measures whether the model
ranks assets correctly, which is what matters when choosing *which* asset to trade.
**IC > 0.05** is generally considered meaningful in finance. **IC < 0** signals overfitting or
no predictive ability.

---

## Trading metrics (online)

### CAGR — compound annual growth rate
The annualized compound return of the strategy. Over windows shorter than a year it stays
mathematically valid but gets noisy, because annualizing amplifies short-run returns. Read it
alongside max drawdown, Sharpe/Sortino, Calmar and hit rate — never alone.

### Sharpe ratio — return per unit of risk
The standard risk-adjusted return measure, comparing excess return against the strategy's
volatility. Design targets for this project: E1 ≥ 0.9, E2 ≥ 0.8, E4 ≥ 1.0.

### Sharpe excess — did we beat buying and holding?
`bt_sharpe_excess = bt_sharpe − sharpe(buy & hold over the same window)`. The drift-neutral
counterpart to raw Sharpe, and the **primary promotion metric** (see below).

Raw Sharpe rewards a model for the asset having gone up. In a bull market a strategy that is
simply long most of the time posts a good Sharpe while adding nothing over a passive position.
Sharpe excess answers the only question that justifies the model's existence: *does it beat
doing nothing?* A permanently-long strategy with no costs scores exactly 0 by construction —
that invariant is enforced in `tests/test_skill_metrics.py`.

### Sortino ratio — return per unit of *downside* risk
Like Sharpe, but only penalizes downside volatility. Closer to how an investor actually
experiences risk, since upside volatility is not a problem anyone wants solved.

### Profit factor / hit rate — trade quality
Profit factor is gross profits divided by gross losses (minimum target 1.2–1.4 for the E3
intraday strategy). Hit rate is the share of winning trades, ideally above 50%. A strategy can
have a low hit rate and still be profitable if the winners are much larger than the losers.

### Max drawdown / Calmar — loss control
Max drawdown is the worst peak-to-trough loss over the period — the maximum financial pain an
investor would have sat through. It matters more than average risk for two reasons that averages
hide: it drives **risk of ruin**, and it drives the chance you **abandon the strategy right
before it recovers**.

Calmar divides CAGR by the absolute max drawdown: return per unit of worst-case loss. Think of
max drawdown as peak psychological stress and CAGR as the average economic benefit — Calmar is
the ratio between them. Low drawdown is essential for the conservative (E1) and moderate (E2)
strategies, whose whole premise is capital preservation.

### Turnover — how often we rebalance
Total value traded divided by capital under management. Critical because high turnover multiplies
transaction costs and can erase a model's entire predicted edge. This is why every backtest here
runs net of costs: a strategy that only works before fees does not work.

---

## Promotion rule — which model becomes champion

### v1 (current, still deciding) — weighted sum
`score = 0.35·Sharpe + 0.25·IC + 0.20·DirectionalAccuracy + 0.20·Calmar`, promoting on a **3%
relative** improvement.

It has a structural flaw, documented in `src/lifecycle/promotion.py`: the metrics are summed on
their **raw scales**, so a metric influences the decision in proportion to how much it *varies*
between candidates, not to its written weight. Measured over the 20 E1 and 62 E2 models in the
registry:

| strategy | metric | nominal weight | **effective weight** |
|---|---|---|---|
| E1 | `bt_calmar` | 0.20 | **0.443** |
| E1 | `bt_sharpe` | 0.35 | 0.405 |
| E1 | `ml_ic` | 0.25 | 0.095 |
| E1 | `ml_directional_accuracy` | 0.20 | **0.057** |
| E2 | `bt_sharpe` | 0.30 | **0.458** |
| E2 | `bt_calmar` | 0.20 | 0.363 |
| E2 | `ml_ic` | 0.20 | 0.097 |
| E2 | `ml_directional_accuracy` | **0.30** | **0.082** |

So `bt_sharpe` + `bt_calmar` decide 85% (E1) / 82% (E2) of every promotion, and both derive from
the *same* equity curve — the score is P&L shape counted twice. Meanwhile IC, the only
drift-neutral skill metric, decides ~10%. And the 3% hysteresis sits against run-to-run noise
the module itself estimates at ±30–65%.

### v2 (shadow mode) — gates plus one primary metric
Simpler, not more complex: it removes the weights entirely, so nominal and effective weight
coincide by construction. Config lives in `lifecycle.promotion.v2` in `base.yaml`.

**Hard gates** — a candidate that fails any of these does not compete:

| gate | what it guarantees |
|---|---|
| `bt_sharpe > 0` | doesn't lose money |
| `ml_ic > 0` | ranks better than chance |
| `ml_dir_acc_edge > 0` | beats the always-long naive |

**Ranking**: `bt_sharpe_excess`, with an **absolute** minimum improvement. Absolute rather than
relative because `(cand − champ) / champ` on an arbitrary scale has no interpretation and blows
up when the denominator is small. A tie never promotes.

**Status**: `enabled: true` (computed and logged to `models/promotion_log.jsonl`),
`active: false` (v1 still executes the decision). Activating it requires the noise floor from
`scripts/evaluation/noise_floor.py` to calibrate `min_improvement_abs`, plus a review of
`scripts/evaluation/backfill_score_v2.py` output.

### Why not the Deflated Sharpe Ratio?
Bailey & López de Prado's Probabilistic and Deflated Sharpe Ratios are the rigorous answer to
"is this Sharpe real, given how many trials I ran?" — a direct fit for a repo that retrains
daily and keeps the best. They were deliberately skipped in favour of a metric explainable in
one sentence. They remain the natural refinement if a formal statistical gate is ever needed;
Ledoit & Wolf (2008) provides the matching test for *differences* in Sharpe between two
strategies, which is literally the champion-versus-candidate question.

---

## References

- Pesaran, M. H. & Timmermann, A. (1992). "A Simple Nonparametric Test of Predictive
  Performance". *Journal of Business & Economic Statistics*, 10(4), 461–465.
- Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality".
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios". *Financial Analysts Journal*, 58(4).
- Ledoit, O. & Wolf, M. (2008). "Robust performance hypothesis testing with the Sharpe ratio".
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 4 (overlapping
  labels, average uniqueness, effective sample size).

---

## Baseline

Every strategy is measured against a fixed **Linear Regression baseline** that is never promoted.
It acts as a sanity floor: if a deep-learning model can't beat a linear model on the same
features, the extra complexity isn't earning its keep. Current champion-vs-baseline numbers are
in the [README](../README.md).

# Data cleaning and validation

Automated OHLCV cleaning that detects and fixes data-quality problems before training.

---

## Why this exists

Data downloaded from yfinance or IOL routinely contains **null values** (missing dates, failed
API calls), **duplicate timestamps** (API errors), **zero volume** (non-trading days or bad
records), and **temporal gaps** (long stretches of missing data).

Left uncleaned, those propagate: NaNs spread through feature computation, models train on corrupt
inputs, and the Information Coefficient goes negative for reasons that have nothing to do with
the model. Cleaning the data moved IC from **−0.124 to +0.232** on the affected tickers — a
larger improvement than any architecture change made in this project.

---

## Data sources

### Daily data (E1 / E2)

| Source | Coverage | Role |
|---|---|---|
| **yfinance** | All tickers | Primary source for US tickers |
| **IOL API** | Argentine assets (`.BA`, `AL*`, `GD*`, `AE*`) | Primary source for Argentine tickers |

### Intraday data (E3)

| Source | Coverage | Frequency |
|---|---|---|
| **Alpaca Markets** | US equities (AAPL, NVDA, SPY, …) | 5-minute bars |
| **yfinance** | Fallback when Alpaca is unavailable | 5-minute bars |

---

## Diagnostics

The cleaner detects four problem classes and reports each per ticker:

**Null values**
```
Ticker: AAPL
  ⚠  Nulls detected in: volume, adj_close
     - volume: 12 (0.5%)
     - adj_close: 3 (0.1%)
```

**Duplicate timestamps** — `⚠  5 duplicate timestamps`

**Zero volume** — `⚠  18 days with volume=0`

**Temporal gaps**
```
  ⚠  2 large gaps in the time series (>4 days)
     - 2020-03-15: 7-day gap
     - 2023-11-23: 5-day gap
```

---

## Cleaning strategies

### 1. Forward fill (default) — `forward_fill`

Propagates the last valid value forward; if nulls appear at the start with no prior value, it
falls back to a backward fill. Conservative: it never invents data, it reuses the last known value.

Use it always, unless you have a specific reason not to. It matches how prices actually behave —
a price holds until the next tick — and for volume it assumes the previous day's level.

```
close:  [100, 105, NaN, NaN, 110]
        ↓
clean:  [100, 105, 105, 105, 110]
```

### 2. Linear interpolation — `interpolate`

Interpolates linearly between known values. Smoother than forward fill.

Reasonable for small gaps (1–2 days) and continuous non-price features. **Be careful with
prices:** interpolation invents values that never traded, which is a subtle form of lying to your
backtest.

```
close:  [100, NaN, NaN, 110]
        ↓
clean:  [100, 103.3, 106.6, 110]
```

### 3. Drop rows — `drop`

Removes any row containing at least one null.

**Never use this for time series.** It breaks temporal continuity, which silently corrupts every
window-based feature and every walk-forward split downstream. Only defensible if you have a large
surplus of history and can afford to lose days.

```
df:     [100, 105, NaN, 110, 115]
        ↓
clean:  [100, 105, 110, 115]   # the NaN day is gone, and so is the time continuity
```

---

## Usage

### Downloading

```bash
python -m src.data.download_daily            # daily (yfinance + IOL fallback)
python -m src.data.download_daily --force    # force a full re-download
```

### Cleaning

```bash
python scripts/data/run_data_cleaning.py                # diagnose + forward fill (default)
python scripts/data/run_data_cleaning.py --interpolate  # use interpolation instead
python scripts/data/run_data_cleaning.py --drop         # drop null rows (not recommended)
```

Output:

```
================================================================================
DATA CLEANING — strategy: FORWARD_FILL
================================================================================

 AAPL
  ✓ No nulls detected
  ✓ Saved: AAPL_daily.csv (2515 days)

 YPFD.BA
  ⚠  Nulls detected in: volume
     - volume: 5 (0.2%)
  🔧 volume: 5 nulls (0.2%) → ✓ forward fill
  ✓ Saved: YPFD.BA_daily.csv (1980 days)

================================================================================
CLEANING SUMMARY
================================================================================
Total tickers processed: 101
  ✓ Cleaned:  99
    Rejected: 2

📄 Quality report saved to: data/clean/data_quality_report.json
```

---

## File layout

```
data/
├── raw/
│   ├── daily/                   # ~105 daily OHLCV CSVs (yfinance + IOL)
│   │   ├── AAPL_daily.csv
│   │   ├── YPFD.BA_daily.csv
│   │   └── …
│   └── intraday/                # 14 CSVs of 5-min bars (Alpaca)
│       ├── AAPL_5min.csv
│       ├── NVDA_5min.csv
│       └── …
├── clean/                       # ~101 cleaned CSVs, ready for training
│   ├── AAPL_daily.csv
│   ├── YPFD.BA_daily.csv
│   ├── …
│   └── data_quality_report.json # per-ticker diagnostic report
├── cache/                       # temporary cache
├── features/                    # feature-engineering outputs
└── snapshots/                   # EDA outputs
    ├── e1_conservative_eda_v1/
    ├── e2_moderate_eda_v1/
    └── e3_intraday_eda_v1/
```

---

## Supported tickers (~101 in `data/clean/`)

**Argentine assets (~65)**
- *Blue-chip equities:* YPFD.BA, GGAL.BA, PAMP.BA, BYMA.BA, CEPU.BA, BBAR.BA, BMA.BA, EDN.BA,
  LOMA.BA, ALUA.BA, AGRO.BA, METR.BA, TGSU2.BA, TGNO4.BA
- *Sovereign bonds:* AL29, AL30, AL35, AL41, GD29, GD30, GD35, GD38, GD41, GD46, AE38
- *Others:* A3.BA, CECO2.BA, CELU.BA, ETHA.BA, YPF

**International assets (~36)**
- *Technology:* AAPL, NVDA, GOOGL, AMZN, META, NFLX, AMD, TSLA
- *Financials & industrials:* JPM, BAC, CAT, DE, XOM, CVX
- *Consumer & healthcare:* JNJ, PG, V, KO, PEP
- *Index ETFs:* SPY, QQQ

The per-strategy ticker universe is defined in `src/config/base.yaml`.

---

## Quality report (JSON)

`data/clean/data_quality_report.json` holds the full per-ticker diagnosis:

```json
{
  "AAPL": {
    "ticker": "AAPL",
    "total_rows": 2515,
    "null_counts": {},
    "null_percentages": {},
    "features_with_nulls": [],
    "zero_volume_days": 0,
    "duplicate_timestamps": 0,
    "data_gaps_days": [],
    "status": "cleaned",
    "rows_after_cleaning": 2515
  },
  "CEPU.BA": {
    "ticker": "CEPU.BA",
    "total_rows": 180,
    "null_counts": { "open": 50, "high": 50, "low": 50 },
    "null_percentages": { "open": 27.8, "high": 27.8, "low": 27.8 },
    "features_with_nulls": ["open", "high", "low"],
    "zero_volume_days": 10,
    "duplicate_timestamps": 0,
    "data_gaps_days": [{ "date": "2023-11-23", "gap_days": 5 }],
    "status": "rejected",
    "rows_after_cleaning": 130
  }
}
```

---

## Configuration

Ticker universe and date range come from `src/config/base.yaml`:

```yaml
data:
  start_date: "2016-03-03"
  end_date: "2026-03-03"
  timezone: "America/New_York"
```

Minimum history threshold:

```python
min_days=252   # one trading year — tickers with less are rejected
```

---

## Deciding what to do with nulls

When a feature has a lot of nulls, first look at the cause:

```bash
cat data/clean/data_quality_report.json | jq '.TICKER'
```

Then apply the threshold:

| Null share | Action |
|---|---|
| < 5% | Forward fill is safe |
| 5–10% | Consider dropping the feature or the ticker |
| > 10% | Drop the ticker — the data is too poor to trust |

Rejection happens in the training pipeline, before the model is built.

---

## Post-cleaning validation

The E1 and E2 pipelines automatically prefer cleaned data:

```python
# src/e1/train_pipeline.py / src/e2/train_pipeline.py
clean_csv_path = data_dir / "clean" / f"{ticker}_daily.csv"

if clean_csv_path.exists():
    csv_path = clean_csv_path                                    # use cleaned data
else:
    csv_path = data_dir / "raw" / "daily" / f"{ticker}_daily.csv"  # fall back to raw
```

---

## Troubleshooting

**Ticker rejected for insufficient history**

```
 TICKER discarded: only 180 days after cleaning (< 252 required)
```

The ticker had too many nulls. Either remove it from `src/config/base.yaml`, download more
history (`period="15y"`), or lower `min_days` — the last one only with good reason, since short
series make walk-forward folds meaningless.

**Features still contain NaN after cleaning**

```python
features = compute_features(ohlcv)
print(features.isna().sum())
```

Usually expected: window-based features need warm-up (an SMA(200) is NaN for the first 200 days).
`make_sequences()` already drops NaN rows when building sequences, so this is normal.

**Missing intraday data (E3)**

```bash
ls data/raw/intraday/                # check what exists
python -m src.e3.intraday_data       # re-download from Alpaca (needs credentials in .env)
```

---

## Reference

| Component | File |
|---|---|
| Daily download | `src/data/download_daily.py` |
| Daily cleaning | `src/data/clean_daily.py` |
| Manual script | `scripts/data/run_data_cleaning.py` |
| Intraday data (E3) | `src/e3/intraday_data.py` |
| IOL API | `src/data/iol_api.py` |
| E1 pipeline | `src/e1/train_pipeline.py` |
| E2 pipeline | `src/e2/train_pipeline.py` |
| Configuration | `src/config/base.yaml` |

# Build Market Labels

## Purpose

Turn candle-anchored feature tables into supervised labels for baseline models and backtest experiments.

## Output Table

The label builder currently emits one table family:

- `market_labels`: one row per feature row, with forward returns and classification labels.

## Output Layout

```text
data/features/market_labels/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Current Label Set

For each configured horizon, the builder emits:

- forward return,
- forward return in basis points,
- forward log return,
- direction label with a dead zone,
- fee-aware tradeability label,
- future trend slope, slope t-stat, linear fit quality, and trend efficiency,
- future trend-state, persistence, and break labels,
- triple-barrier direction, hit time, and realized barrier return.

It also emits:

- future realized volatility over a configurable window,
- mean future absolute return over that same window.

## Example Commands

Build labels for all feature tables with default horizons:

```bash
python3 scripts/data/build_market_labels.py
```

Build only 1-bar and 5-bar labels for Binance BTCUSDT:

```bash
python3 scripts/data/build_market_labels.py \
  --provider binance \
  --symbol BTCUSDT \
  --horizon 1 \
  --horizon 5
```

Use a wider dead zone and higher cost threshold:

```bash
python3 scripts/data/build_market_labels.py \
  --direction-dead-zone-bps 8 \
  --trade-cost-bps 12
```

Tune trend and triple-barrier labels:

```bash
python3 scripts/data/build_market_labels.py \
  --trend-slope-dead-zone-bps-per-bar 0.75 \
  --trend-slope-tstat-threshold 1.25 \
  --trend-efficiency-threshold 0.4 \
  --triple-barrier-profit-bps 25 \
  --triple-barrier-stop-bps 20
```

## Notes

- Labels are only emitted when the current row and future target rows are marked as closed candles.
- Future trend and triple-barrier columns are labels, not model features. The training dataset loader excludes the `future_trend_` and `triple_barrier_` prefixes from automatic feature selection.
- Longer-horizon labels remain blank until enough future candles exist.
- Provider and symbol series are processed across multiple dates, so day-end rows can use future closes from the next day's file when that data exists.

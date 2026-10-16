# Build Market Features

## Purpose

Turn staged market tables into candle-anchored feature tables for model training, validation, and backtests.

## Output Table

The feature builder currently emits one table family:

- `market_features`: one row per candle window, with aligned trade and top-of-book aggregates.

## Output Layout

```text
data/features/market_features/<provider>/<symbol>/YYYY/MM/DD.csv
```

Each output file is rebuilt deterministically from staged inputs for that provider, symbol, and date.

## Current Feature Set

Each row includes:

- candle OHLCV context,
- candle range and body percentages,
- 1-bar and 5-bar close returns,
- 5-bar rolling return standard deviation,
- multi-window trend returns, EMAs, EMA spreads, and realized volatility,
- trend-shape diagnostics from rolling log-price regressions: slope, slope t-stat, linear fit quality, path efficiency, chop score, drawdown, and runup,
- short-vs-medium and medium-vs-long trend slope acceleration,
- rolling volume expansion ratio and z-score,
- support/resistance context from prior high/low ranges, breakout flags, and range position,
- online regime/changepoint features from the HMM-style regime filter and BOCPD-style change detector,
- trade-window counts, volumes, VWAP, and aggressor imbalance,
- top-of-book mean and last mid price,
- mean spread, spread in basis points, and size imbalance.

## Example Commands

Build features for all staged datasets:

```bash
python3 scripts/data/build_market_features.py
```

Build only Coinbase BTC-USD features:

```bash
python3 scripts/data/build_market_features.py \
  --provider coinbase \
  --symbol BTC-USD
```

Smoke-test one dataset:

```bash
python3 scripts/data/build_market_features.py \
  --limit-datasets 1
```

## Notes

- Candles are the anchor timeline. A feature row is only emitted when a staged candle exists.
- Trade and top-of-book features are aligned to each candle window, not to fixed one-minute buckets.
- Multiple candle updates for the same `candle_start` are deduplicated before features are built, with closed candles preferred when available.

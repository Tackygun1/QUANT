# Backfill Binance Historical Data

## Purpose

Pull free historical Binance spot market data into the repo’s raw JSONL layout so the existing staging, feature, label, training, and backtest pipeline can run on a larger real dataset.

## Inputs

- Binance Spot REST `klines`
- Binance Spot REST `aggTrades`

## Outputs

```text
data/raw/binance/<channel>/<symbol>/YYYY/MM/DD/HH.jsonl
```

## Example

```bash
python3 scripts/data/backfill_binance_historical.py \
  --config configs/venues/binance-historical.example.json
```

Or with flags:

```bash
python3 scripts/data/backfill_binance_historical.py \
  --symbol BTCUSDT \
  --channel candles \
  --start-time 2026-04-12T00:00:00Z \
  --end-time 2026-04-12T12:00:00Z \
  --output-root data/historical_demo/raw
```

## Notes

- Historical `candles` and `trades` are supported.
- Historical top-of-book snapshots are not available from the free Binance Spot REST market-data endpoints, so `book_ticker` is not backfilled here.
- The backfill writes the same normalized raw envelope as the live collector, which means the downstream stager and feature pipeline can be reused unchanged.
- For high-volume symbols, `aggTrades` backfills can be much larger and slower than candle-only backfills.

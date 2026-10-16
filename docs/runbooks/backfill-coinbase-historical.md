# Backfill Coinbase Historical Data

## Purpose

Pull free historical Coinbase Exchange candles and trades into the repo's raw JSONL layout so the existing staging, feature, label, training, stacking, and backtest pipeline can run on a larger real dataset.

## Inputs

- Coinbase Exchange REST `GET /products/{product_id}/candles`
- Coinbase Exchange REST `GET /products/{product_id}/trades`

## Outputs

```text
data/raw/coinbase/candles/<product>/YYYY/MM/DD/HH.jsonl
```

## Example

```bash
python3 scripts/data/backfill_coinbase_historical.py \
  --config configs/venues/coinbase-historical.example.json
```

Or with flags:

```bash
python3 scripts/data/backfill_coinbase_historical.py \
  --product BTC-USD \
  --channel candles \
  --start-time 2026-04-06T00:00:00Z \
  --end-time 2026-04-13T00:00:00Z \
  --candle-interval 5m \
  --output-root data/historical_demo/raw
```

## Notes

- This backfill supports `candles` and `trades`.
- Coinbase Exchange documents a `300` candle maximum per request, so the backfill pages the requested interval into safe slices automatically.
- Coinbase trades use cursor pagination from the latest page backward, so trade backfills are most practical for recent windows.
- The backfill writes the same normalized raw envelope as the live collector, which means the downstream stager and feature pipeline can be reused unchanged.
- This is the preferred free historical fallback when Binance REST is geoblocked.

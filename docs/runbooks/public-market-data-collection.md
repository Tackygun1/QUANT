# Public Market Data Collection

## Purpose

Collect continuous free public crypto market data into partitioned JSONL files under `data/raw/`.

## Supported Providers

- Coinbase public Advanced Trade WebSocket feed.
- Binance public spot market data feed via `data-stream.binance.vision`.

## Output Layout

Collected data is partitioned by provider, logical channel, symbol, and hour:

```text
data/raw/<provider>/<channel>/<symbol>/YYYY/MM/DD/HH.jsonl
```

## Example Commands

Install the local package first:

```bash
python3 -m pip install -e .
```

Run the Coinbase collector from the checked-in example config:

```bash
python3 scripts/data/collect_market_data.py \
  --config configs/venues/coinbase-public.json
```

Run Binance with an explicit symbol list and stop after 25 records for a smoke test:

```bash
python3 scripts/data/collect_market_data.py \
  --provider binance \
  --product BTCUSDT \
  --channel trades \
  --channel book_ticker \
  --channel candles \
  --max-records 25
```

Run Coinbase with a narrower feed:

```bash
python3 scripts/data/collect_market_data.py \
  --provider coinbase \
  --product BTC-USD \
  --channel trades \
  --channel level2
```

## Notes

- Coinbase heartbeats are subscribed automatically to keep sparse channels open, but they are not persisted.
- Binance connections are expected to reconnect periodically because the public WebSocket connection is time-limited by the venue.
- The collector stores provider-native payloads plus a normalized envelope so later pipelines can reprocess raw data without reconnecting to the exchange.

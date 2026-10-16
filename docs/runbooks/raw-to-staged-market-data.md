# Raw To Staged Market Data

## Purpose

Convert provider-native raw JSONL market data into cleaner staged CSV tables that are easier to use for features, model training, and backtests.

## Output Tables

The staging pass currently emits:

- `trades`: one row per normalized trade.
- `candles`: one row per normalized candle event.
- `order_book_top`: best bid and ask snapshots from ticker-style channels.
- `order_book_events`: message-level summaries for level-2 order-book payloads.
- `order_book_levels`: flattened price-level updates or snapshots from level-2 payloads.

## Output Layout

```text
data/staging/<table>/<provider>/<symbol>/YYYY/MM/DD/HH.csv
```

Each staged file is deterministically regenerated from one raw partition file, so rerunning the parser overwrites that staged partition rather than appending duplicates.

## Example Commands

Stage everything currently under `data/raw/`:

```bash
python3 scripts/data/stage_market_data.py
```

Stage only Coinbase BTC-USD raw files:

```bash
python3 scripts/data/stage_market_data.py \
  --provider coinbase \
  --symbol BTC-USD
```

Stage only trade partitions for a quick smoke test:

```bash
python3 scripts/data/stage_market_data.py \
  --channel trades \
  --limit-files 2
```

## Notes

- The parser relies on the normalized raw envelope written by `collect_market_data.py`.
- Binance partial-depth snapshots are flattened into `order_book_levels` rows.
- Coinbase level-2 deltas are also flattened into `order_book_levels`, with one row per update.
- `order_book_top` rows are derived from Coinbase `ticker` and Binance `book_ticker` channels.

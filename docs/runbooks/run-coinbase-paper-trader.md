# Run Coinbase Paper Trader

## Purpose

Backfill a recent Coinbase window, rebuild live features, score the latest closed candle with a saved model artifact, and write paper signals, orders, ledger rows, and state under `data/paper`.

## Inputs

- Coinbase Exchange REST `candles`
- Coinbase Exchange REST `trades`
- A saved model artifact with metadata. Direct tabular and stacked meta-model artifacts are supported.

## Outputs

```text
data/paper/signals/<provider>/<symbol>/YYYY/MM/DD/HH.jsonl
data/paper/orders/<provider>/<symbol>/YYYY/MM/DD/HH.jsonl
data/paper/ledger/<provider>/<symbol>/YYYY/MM/DD/HH.jsonl
data/paper/state/<provider>/<symbol>/<model_name>.json
data/paper/runtime/<provider>/<symbol>/<run_slug>/
```

## Example

Train a live-scoring artifact first:

```bash
python3 scripts/models/train_tabular_baselines.py \
  --input-root data/trade_demo/features \
  --output-root data/trade_demo/models \
  --provider coinbase \
  --symbol BTC-USD \
  --model histgb_return \
  --validation-fraction 0.2 \
  --minimum-rows 96
```

Then run one paper-trading snapshot:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --config configs/strategies/coinbase_paper_trading.example.json
```

The current promoted hybrid-direction runtime uses the tuned untouched-holdout execution settings from April 17, 2026 UTC:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --config configs/strategies/runtime/coinbase_hybrid_direction_tuned_20260417T013104Z.json
```

For a stacked meta-model, point the trader at the ensemble artifact. It will auto-discover the latest supporting base artifacts under the same `models` root:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --artifact-dir data/trade_demo/models/ensembles/lightgbm_return_meta/coinbase/BTC-USD \
  --product BTC-USD \
  --lookback-hours 3 \
  --iterations 1
```

Or run a deterministic replay against a fixed close time:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --artifact-dir data/trade_demo/models/tabular/histgb_return/coinbase/BTC-USD \
  --product BTC-USD \
  --lookback-hours 3 \
  --iterations 1 \
  --end-time 2026-04-13T03:05:00Z
```

## Notes

- Direct artifacts score the latest `market_features` row directly.
- Stacked meta-model artifacts compute live base-model predictions first, including the latest available Transformer sequence forecast, then score the ensemble row.
- The sizing config now supports common discretionary guardrails: 1% max trade risk, stop-distance-based sizing, capped position changes, daily loss stops, and portfolio drawdown halts.
- Each paper signal now carries a trade plan with entry reference, stop-loss, and take-profit levels. The default take-profit distance is `2.0x` the stop distance and can be changed with `--take-profit-multiple`.
- Use `--artifact-search-root` if the supporting base artifacts are not stored under the same ancestor `models` directory as the target artifact.
- Each iteration writes a fresh runtime dataset under `data/paper/runtime` and appends the resulting signal, order, and ledger events.

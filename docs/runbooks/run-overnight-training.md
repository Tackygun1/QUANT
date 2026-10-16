# Run Overnight Training

## Purpose

Build a larger Coinbase `BTC-USD` research dataset and run an overnight evolutionary search over the main return models, including the hybrid Transformer plus LightGBM meta-model stack, with checkpointed progress and retryable dataset assembly.

## What It Does

- Backfills a large candle window and a shorter high-resolution trade window from free Coinbase Exchange REST endpoints.
- Rebuilds `raw`, `staging`, `market_features`, `market_labels`, and `market_supervised` inside a single versioned workspace.
- Searches `histgb_return`, `ridge_return`, `lightgbm_hybrid_return_meta`, and `lightgbm_hybrid_direction_meta` candidates over an `8` hour budget by mutating the best prior candidates.
- For hybrid candidates, trains a final hybrid Transformer, runs purged walk-forward OOF generation, builds `market_hybrid_stacked`, trains the meta-model, and backtests the meta artifact.
- Saves every candidate artifact, validation prediction file, fee-aware backtest, leaderboard, and checkpointed state.

## Inputs

- Coinbase Exchange REST `candles`
- Coinbase Exchange REST `trades`
- A JSON config like [overnight_coinbase_btcusd.example.json](/home/gofish/QUANT/configs/research/overnight_coinbase_btcusd.example.json)

## Outputs

Inside the configured `workspace_root`:

```text
raw/
staging/
features/
models/evolution/
reports/backtests/
search/candidates.jsonl
search/best_results.json
state/overnight_state.json
state/dataset_summary.json
state/overnight_summary.json
logs/
```

Hybrid candidates also write candidate-local pipeline assets under each candidate directory:

```text
models/evolution/lightgbm_hybrid_return_meta/<candidate_id>/
  hybrid_transformer/
  pipeline/features/
  pipeline/models/walk_forward/
  model.pkl
  metadata.json
  validation_predictions.csv
```

## Example

```bash
python3 scripts/research/run_overnight_training.py \
  --config configs/research/overnight_coinbase_btcusd.example.json
```

To launch it detached for the night:

```bash
nohup bash -lc '
  while true; do
    PYTHONPATH=src ./.venv/bin/python scripts/research/run_overnight_training.py \
      --config configs/research/runtime/overnight-current.json
    status=$?
    if [ "$status" -eq 0 ]; then
      break
    fi
    sleep 30
  done
' > experiments/overnight/coinbase/BTC-USD/current/logs/overnight.log 2>&1 &
```

## Notes

- The dataset phase is rebuilt from scratch inside the workspace if it has not completed successfully yet. This avoids duplicate raw-hour partitions on retry.
- The legacy single-output transformer search still exists, but the default overnight loop now prioritizes the hybrid stack.
- Hybrid candidates now use a materially larger recent row window by default, plus purged walk-forward windows, so the stacked meta-model trains on a less starved sample.
- Hybrid candidates also use a looser model-specific execution search space and an inactivity penalty, so zero-trade candidates do not dominate the hybrid leaderboard by default.
- Execution assumptions are part of the search objective through the sizing config. That includes churn controls like `min_rebalance_abs_delta`, `min_hold_bars`, signal thresholds, and hysteresis thresholds for exits and flips.
- Resume works by reusing `state/overnight_state.json` and appending new candidates into `search/candidates.jsonl`.

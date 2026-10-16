# Build Stacked Dataset

## Purpose

Join out-of-fold walk-forward predictions back onto the supervised dataset so a meta-model can train on base-model forecasts without leakage.

For the hybrid stack, this usually means engineered features from `market_supervised` plus leak-safe multi-horizon Transformer quantiles and quantile spreads from the hybrid walk-forward run.

## Inputs

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
data/models/walk_forward/<family>/<model_name>/<provider>/<symbol>/<run-id>/validation_predictions.csv
```

## Outputs

```text
data/features/training_datasets/market_stacked/<provider>/<symbol>/YYYY/MM/DD.csv
```

Each dataset write also creates a sibling manifest file:

- `<day>.manifest.json`

## Example

```bash
python3 scripts/models/build_stacked_dataset.py \
  --config configs/models/stacked_dataset.example.json
```

Trend-state probability stack:

```bash
python3 scripts/models/build_stacked_dataset.py \
  --config configs/models/trend_state_stacked_dataset.example.json
```

Executable trend stack:

```bash
python3 scripts/models/build_stacked_dataset.py \
  --config configs/models/executable_trend_stacked_dataset.example.json
```

## Notes

- The stacked dataset only keeps rows that have complete out-of-fold forecasts for every requested prediction source.
- The builder uses the latest walk-forward run per model, provider, and symbol.
- `histgb_trend_state` can contribute hard trend-state predictions, trend-state margins, class probabilities, and confidence through the `oof_trend_state_*` columns.
- `histgb_trend_state_fast`, `histgb_trend_state`, and `histgb_triple_barrier_fast` can be combined into an execution-focused stack via `oof_fast_trend_state_*`, `oof_slow_trend_state_*`, and `oof_fast_triple_barrier_*`.
- Manifest files record which walk-forward runs were used for the join.

# Train Meta Models

## Purpose

Train LightGBM meta-models on the stacked dataset that combines engineered features with out-of-fold base-model forecasts.

The preferred hybrid setup is a LightGBM meta-learner trained on `market_hybrid_stacked`, where the extra columns come from the multi-horizon quantile Transformer.

For trend-recognition experiments, train on `market_trend_state_stacked` and include the `trend_state_oof` feature group so the meta-model sees leak-safe trend-state probabilities instead of raw future labels.

For execution-focused trend retrains, train on `market_executable_trend_stacked` and use `fast_trend_state_oof`, `slow_trend_state_oof`, and `fast_triple_barrier_oof` together with `baseline_core`.

## Inputs

```text
data/features/training_datasets/market_stacked/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Outputs

```text
data/models/ensembles/<model_name>/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `model.pkl`
- `metadata.json`
- `validation_predictions.csv`
- `training_config.json`

## Example

```bash
python3 -m pip install -e '.[stack]'
python3 scripts/models/train_meta_models.py \
  --config configs/models/meta_models.example.json
```

Trend-aware retrain example:

```bash
python3 scripts/models/train_meta_models.py \
  --input-root experiments/layered_retrain_recent/coinbase/BTC-USD/20260414T131253Z/features \
  --output-root experiments/layered_retrain_recent/coinbase/BTC-USD/20260414T131253Z/models \
  --provider coinbase \
  --symbol BTC-USD \
  --model lightgbm_hybrid_return_meta \
  --feature-group baseline_core \
  --feature-group hybrid_transformer_oof \
  --feature-group trend_regime_layer \
  --sample-weight-scheme trend_regime_target_balance
```

Trend-state probability meta example:

```bash
python3 scripts/models/train_meta_models.py \
  --config configs/models/trend_state_meta.example.json
```

Executable trend direction meta example:

```bash
python3 scripts/models/train_meta_models.py \
  --config configs/models/executable_trend_direction_meta.example.json
```

Executable trend triple-barrier meta example:

```bash
python3 scripts/models/train_meta_models.py \
  --config configs/models/executable_trend_triple_barrier_meta.example.json
```

## Notes

- The meta-model trainer uses chronological train/validation splits on the stacked dataset.
- The current implementation supports legacy stacked models and explicit hybrid meta-model ids such as `lightgbm_hybrid_return_meta`.
- `trend_regime_layer` is a pruned trend-context feature group intended to improve regime recognition without adding the full layered feature set.
- `trend_state_oof` contains out-of-fold trend-state predictions, margins, probabilities, and confidence from `histgb_trend_state`.
- `fast_trend_state_oof` contains the 6-bar execution-facing trend-state OOF columns from `histgb_trend_state_fast`.
- `slow_trend_state_oof` contains the 24-bar regime-prior OOF columns from `histgb_trend_state`.
- `fast_triple_barrier_oof` contains the 6-bar after-cost barrier OOF columns from `histgb_triple_barrier_fast`.
- `--sample-weight-scheme trend_regime_balance` or `trend_regime_target_balance` can rebalance bullish, bearish, and range regimes during meta training.
- If the stacked dataset is too short or collapses to one class on the training split, the run is skipped.

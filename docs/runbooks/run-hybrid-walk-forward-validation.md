# Run Hybrid Walk-Forward Validation

## Purpose

Generate leak-resistant out-of-fold hybrid Transformer forecasts for stacking.

## Inputs

- A supervised dataset with multi-horizon labels.
- A hybrid Transformer config with horizons, quantiles, and lookback.

## Outputs

```text
data/models/walk_forward/hybrid_transformers/quantile_forecaster/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `fold_metrics.csv`
- `validation_predictions.csv`
- `summary.json`
- `run_config.json`

## Example

```bash
python3 scripts/models/run_hybrid_walk_forward_validation.py \
  --config configs/models/walk_forward_hybrid_transformer.example.json
```

## Notes

- This path supports both `embargo_rows` and `purge_rows`.
- `purge_rows` should be at least as large as the longest overlapping label horizon you want to protect.
- The resulting `validation_predictions.csv` is designed to feed `build_stacked_dataset.py` without leaking in-sample Transformer forecasts into the tree meta-learner.

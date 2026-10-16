# Run Walk-Forward Validation

## Purpose

Evaluate baseline tabular models or the Transformer forecaster with expanding-window, chronologically ordered validation folds.

## Inputs

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Outputs

```text
data/models/walk_forward/<family>/<model_name>/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `summary.json`
- `run_config.json`
- `fold_metrics.csv`
- `validation_predictions.csv`

## Tabular Example

```bash
python3 -m pip install -e '.[ml]'
python3 scripts/models/run_walk_forward_validation.py \
  --config configs/models/walk_forward_tabular.example.json
```

Dedicated trend-state OOF example:

```bash
python3 scripts/models/run_walk_forward_validation.py \
  --family tabular \
  --model histgb_trend_state \
  --target-column future_trend_state_label_24_bar \
  --embargo-rows 24
```

## Transformer Example

```bash
python3 -m pip install -e '.[ml]'
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 scripts/models/run_walk_forward_validation.py \
  --config configs/models/walk_forward_transformer.example.json
```

## Notes

- The evaluator uses expanding training windows and fixed-size validation windows.
- `summary.json` contains aggregated out-of-sample metrics over all validation folds.
- `fold_metrics.csv` keeps per-fold train and validation metrics for stability checks.
- Classification models with supported ternary targets emit probability/margin columns in `validation_predictions.csv`; `histgb_trend_state` uses these as leak-safe inputs for stacked meta-models.
- Very short series are marked as skipped instead of forcing unrealistic folds.

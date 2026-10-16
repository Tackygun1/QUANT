# Train Hybrid Transformer

## Purpose

Train the multi-horizon quantile Transformer used in the hybrid tree-plus-sequence stack.

## Inputs

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
```

The dataset should include multi-horizon labels such as:

- `forward_return_1_bar`
- `forward_return_6_bar`
- `forward_return_24_bar`
- `forward_return_96_bar`

## Outputs

```text
data/models/hybrid_transformers/quantile_forecaster/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `model.pt`
- `metadata.json`
- `validation_predictions.csv`
- `training_config.json`

## Example

```bash
python3 -m pip install -e '.[ml]'
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 scripts/models/train_hybrid_transformer.py \
  --config configs/models/hybrid_transformer_forecaster.example.json
```

## Notes

- The saved predictions include multiple horizons and quantiles, plus per-horizon quantile spreads for uncertainty.
- The primary backtest-compatible forecast is stored in metadata as `primary_prediction_column`.
- This artifact is the live sequence model that hybrid meta-model paper trading can call at inference time.

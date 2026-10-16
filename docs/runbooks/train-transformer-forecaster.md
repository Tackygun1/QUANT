# Train Transformer Forecaster

## Purpose

Train a sequence Transformer that forecasts forward returns from rolling windows of the joined supervised dataset.

## Inputs

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Outputs

```text
data/models/transformers/return_forecaster/<provider>/<symbol>/<run-id>/
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
python3 scripts/models/train_transformer_forecaster.py \
  --config configs/models/transformer_forecaster.example.json
```

## Notes

- The Transformer path requires `torch`.
- On Linux, installing `torch` from the default PyPI index may pull a CUDA build. Use the CPU index above unless you explicitly want GPU dependencies.
- Sequence windows are built from chronologically sorted closed candles.
- Early rows are dropped until enough history exists to fill the configured lookback.

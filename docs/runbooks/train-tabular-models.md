# Train Tabular Models

## Purpose

Train the first baseline models from the joined supervised dataset:

- ridge regression for forward returns,
- logistic regression for direction labels,
- histogram gradient boosting for returns and direction.

## Inputs

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Outputs

```text
data/models/tabular/<model_name>/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `model.pkl`
- `metadata.json`
- `validation_predictions.csv`
- `training_config.json`

## Example

```bash
python3 -m pip install -e '.[ml]'
python3 scripts/models/train_tabular_baselines.py \
  --config configs/models/tabular_baselines.example.json
```

## Notes

- The command trains per provider and symbol series, not per day.
- Rows are split chronologically into train and validation segments.
- The current implementation requires `scikit-learn`.

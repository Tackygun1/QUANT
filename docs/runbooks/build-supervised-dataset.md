# Build Supervised Dataset

## Purpose

Join `market_features` and `market_labels` into one supervised table that can be consumed directly by baseline model training code.

## Output Layout

```text
data/features/training_datasets/market_supervised/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Notes

- Joins are keyed by provider, symbol, feature date, candle start, and candle end.
- The output keeps all feature columns, appends label columns, and adds:
  - `has_any_forward_label`
  - `has_1_bar_label`

## Example

```bash
python3 scripts/data/build_supervised_dataset.py
```

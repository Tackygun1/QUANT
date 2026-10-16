# Run Model Backtest

## Purpose

Run a fee-aware backtest against a saved model artifact by joining its validation predictions to the matching supervised or stacked dataset.

## Inputs

```text
<artifact-dir>/metadata.json
<artifact-dir>/validation_predictions.csv
data/features/training_datasets/<dataset-name>/<provider>/<symbol>/YYYY/MM/DD.csv
```

## Outputs

```text
reports/backtests/<model_name>/<provider>/<symbol>/<run-id>/
```

Each run writes:

- `summary.json`
- `periods.csv`
- `trades.csv`
- `backtest_config.json`

## Example

```bash
python3 scripts/models/run_model_backtest.py \
  --artifact-dir data/models/ensembles/lightgbm_return_meta/binance/BTCUSDT/<run-id> \
  --input-root data/features \
  --output-root reports/backtests
```

Or with a config file:

```bash
python3 scripts/models/run_model_backtest.py \
  --config configs/strategies/model_backtest.example.json
```

## Notes

- The backtester uses out-of-sample validation predictions, not in-sample training predictions.
- Regression models use prediction magnitude for signal strength; classification models use predicted direction with volatility scaling.
- Transaction costs are charged on turnover using `fee_bps + slippage_bps`.
- The sizing layer can enforce common guardrails such as the 1% risk rule, capped per-bar position changes, daily loss stops, and portfolio drawdown halts.
- Churn controls are available in the sizing layer too: `hold_signal_threshold_bps`, `flip_signal_threshold_bps`, `min_hold_bars`, and `min_rebalance_abs_delta`.
- Trend-aware execution overlays can be combined with churn controls via `trend_filter_*` sizing fields.
- The summary includes compounded net return, turnover, hit rate, drawdown, and a simple Sharpe-like statistic.
- The Sharpe-like statistic is suppressed until at least 30 periods are available.

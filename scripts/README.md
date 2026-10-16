# Scripts

Keep small operational entry points here. Prefer thin wrappers that call package code in `src/quant` rather than embedding business logic directly in scripts.

Current entry points:

- `data/collect_market_data.py`: run the public market-data collector for Coinbase or Binance.
- `data/backfill_binance_historical.py`: backfill free historical Binance spot candles or aggregate trades.
- `data/backfill_coinbase_historical.py`: backfill free historical Coinbase Exchange candles and trades.
- `data/stage_market_data.py`: convert raw JSONL partitions into staged CSV tables.
- `data/build_market_features.py`: build candle-anchored feature tables from staged data.
- `data/build_market_labels.py`: build supervised labels from feature tables.
- `data/build_supervised_dataset.py`: join features and labels into a training-ready table.
- `data/run_market_viewer.py`: serve a local browser UI for candles, volume, paper entries, stops, take-profits, and exit hits.
- `models/train_tabular_baselines.py`: train ridge, logistic, and gradient-boosting baselines.
- `models/train_transformer_forecaster.py`: train the sequence Transformer forecaster.
- `models/train_hybrid_transformer.py`: train the multi-horizon quantile Transformer for the hybrid stack.
- `models/run_walk_forward_validation.py`: run expanding-window validation for tabular or Transformer models.
- `models/run_hybrid_walk_forward_validation.py`: run purged walk-forward validation for the hybrid quantile Transformer.
- `models/build_stacked_dataset.py`: join walk-forward out-of-fold predictions into a stacked training table.
- `models/train_meta_models.py`: train LightGBM meta-models on the stacked dataset.
- `models/run_model_backtest.py`: run a fee-aware backtest on saved validation predictions.
- `trading/run_coinbase_paper_trader.py`: run a Coinbase paper-trading snapshot or loop from a saved artifact with live trade-plan fields.
- `trading/run_historical_replay.py`: replay saved supervised rows through a trained artifact at accelerated speed for the viewer.

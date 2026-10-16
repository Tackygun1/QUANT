# Configs

This directory holds checked-in configuration templates and defaults.

Use separate subdirectories so venue settings, feature definitions, model configs, strategy rules, and risk limits can evolve independently without becoming one large opaque file.

Current examples:

- `venues/coinbase-public.json`
- `venues/coinbase-historical.example.json`
- `venues/binance-public.json`
- `venues/binance-historical.example.json`
- `models/tabular_baselines.example.json`
- `models/transformer_forecaster.example.json`
- `models/walk_forward_tabular.example.json`
- `models/walk_forward_transformer.example.json`
- `models/stacked_dataset.example.json`
- `models/trend_state_stacked_dataset.example.json`
- `models/meta_models.example.json`
- `models/trend_state_meta.example.json`
- `models/executable_trend_stacked_dataset.example.json`
- `models/executable_trend_direction_meta.example.json`
- `models/executable_trend_triple_barrier_meta.example.json`
- `strategies/coinbase_paper_trading.example.json`
- `strategies/model_backtest.example.json`
- `strategies/model_backtest.hybrid_direction_low_turnover.json`

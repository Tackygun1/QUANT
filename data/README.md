# Data

This directory is reserved for local datasets and generated artifacts that support research, backtesting, and paper trading.

- `raw/`: venue-native captures such as trades, candles, and order-book updates.
- `staging/`: cleaned or normalized intermediate datasets.
- `features/`: feature tables and sequence tensors derived from staged data.
- `models/`: serialized model artifacts kept local during development.
- `backtests/`: simulation outputs and performance exports.
- `paper/`: paper-trading logs and event streams.

Large files and generated outputs should stay out of git unless there is a deliberate reason to version a small reference fixture.

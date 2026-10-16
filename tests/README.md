# Tests

Test layout is intentionally split by purpose:

- `unit/`: fast, deterministic package-level tests.
- `integration/`: exchange adapters, storage, and pipeline wiring.
- `backtests/`: strategy and simulator regression checks.
- `smoke/`: end-to-end sanity checks for core workflows.
- `fixtures/`: reusable test inputs and small reference datasets.

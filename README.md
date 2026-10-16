# QUANT

This Repo consists of the code used in my Extended Essay.
Downloading this repo and following the directions should enable anyone to use the models I talk about in my essay.
Note: AI was used to help develope this code. This insludes: Bug fixing, graph creation, structuring the repo, writing reports, etc 

## Status

- Repository initialized with a `main` branch.
- Baseline docs, automation, and GitHub templates are in place.
- The repo layout now reflects a Python-based trading research and execution workflow.
- Packaging uses `pyproject.toml`; deployment target still needs to be chosen.

## Quick Start

```bash
make bootstrap
python3 -m pip install -e '.[dev,ml]'
make check
```

`make bootstrap` ensures the expected directories and script permissions exist locally. `make check` validates the repository shell, linting, and unit tests.
Install `.[dev,stack]` plus CPU PyTorch when you want local tests to exercise LightGBM and Transformer paths.

## Public Market Data

The repo now includes a live public market-data collector for free crypto feeds:

- Coinbase Advanced Trade market data WebSocket
- Binance public spot market data WebSocket
- Binance historical spot REST backfill for candles and aggregate trades
- Coinbase Exchange historical REST backfill for candles and trades

Install the package and run a collector:

```bash
python3 -m pip install -e .
python3 scripts/data/collect_market_data.py \
  --config configs/venues/coinbase-public.json
```

See `docs/runbooks/public-market-data-collection.md` for explicit channel choices and example commands.

Backfill historical Binance spot data into the same raw layout:

```bash
python3 scripts/data/backfill_binance_historical.py \
  --config configs/venues/binance-historical.example.json
```

Or use Coinbase historical candles when Binance REST is unavailable:

```bash
python3 scripts/data/backfill_coinbase_historical.py \
  --config configs/venues/coinbase-historical.example.json
```

Convert raw JSONL into staged CSV tables:

```bash
python3 scripts/data/stage_market_data.py
```

See `docs/runbooks/raw-to-staged-market-data.md` for the emitted tables and filters.

Build model-ready market features from staged data:

```bash
python3 scripts/data/build_market_features.py
```

See `docs/runbooks/build-market-features.md` for the feature set and output layout.

Build supervised labels from the feature tables:

```bash
python3 scripts/data/build_market_labels.py
```

For the hybrid stack, use multi-horizon labels such as `1/6/24/96` bars:

```bash
python3 scripts/data/build_market_labels.py \
  --horizon 1 --horizon 6 --horizon 24 --horizon 96
```

See `docs/runbooks/build-market-labels.md` for horizons, dead-zone settings, and output files.

Join features and labels into one supervised dataset:

```bash
python3 scripts/data/build_supervised_dataset.py
```

Run the local viewer for candles and volume:

```bash
python3 scripts/data/run_market_viewer.py --data-root data/features
```

Train baseline tabular models:

```bash
python3 -m pip install -e '.[ml]'
python3 scripts/models/train_tabular_baselines.py \
  --config configs/models/tabular_baselines.example.json
```

Train the Transformer forecaster:

```bash
python3 -m pip install -e '.[ml]'
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 scripts/models/train_transformer_forecaster.py \
  --config configs/models/transformer_forecaster.example.json
```

Train the hybrid multi-horizon quantile Transformer:

```bash
python3 -m pip install -e '.[ml]'
python3 -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python3 scripts/models/train_hybrid_transformer.py \
  --config configs/models/hybrid_transformer_forecaster.example.json
```

Run walk-forward validation on the joined supervised dataset:

```bash
python3 scripts/models/run_walk_forward_validation.py \
  --config configs/models/walk_forward_tabular.example.json
```

Run purged walk-forward validation for the hybrid Transformer:

```bash
python3 scripts/models/run_hybrid_walk_forward_validation.py \
  --config configs/models/walk_forward_hybrid_transformer.example.json
```

Build a stacked dataset from out-of-fold forecasts:

```bash
python3 scripts/models/build_stacked_dataset.py \
  --config configs/models/stacked_dataset.example.json
```

Train the LightGBM meta-models:

```bash
python3 -m pip install -e '.[stack]'
python3 scripts/models/train_meta_models.py \
  --config configs/models/meta_models.example.json
```

The current recommended hybrid path is:

1. `build_market_labels.py` with `1/6/24/96` horizons.
2. `train_hybrid_transformer.py` for the live sequence artifact.
3. `run_hybrid_walk_forward_validation.py` for leak-safe OOF Transformer features.
4. `build_stacked_dataset.py` into `market_hybrid_stacked`.
5. `train_meta_models.py` with `lightgbm_hybrid_*` models.

Run a fee-aware backtest on a saved model artifact:

```bash
python3 scripts/models/run_model_backtest.py \
  --config configs/strategies/model_backtest.example.json
```

Run a Coinbase paper-trading snapshot from a saved tabular artifact:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --config configs/strategies/coinbase_paper_trading.example.json
```

Run the current promoted hybrid-direction paper-trading snapshot:

```bash
python3 scripts/trading/run_coinbase_paper_trader.py \
  --config configs/strategies/runtime/coinbase_hybrid_direction_tuned_20260417T013104Z.json
```

## Repository Layout

```text
.
|-- .github/              # PR templates, issue templates, workflow definitions
|-- configs/              # Venue, feature, model, strategy, and risk config
|-- data/                 # Local data and generated artifacts (kept out of git)
|-- docs/                 # Architecture notes, research, ADRs, and runbooks
|-- experiments/          # Experiment specs and run metadata
|-- infra/                # Runtime and deployment scaffolding
|-- notebooks/            # Exploratory and validation notebooks
|-- reports/              # Backtest output, figures, and summaries
|-- scripts/              # Operational and developer scripts
|-- src/quant/            # Python application package
|-- tests/                # Unit, integration, smoke, and backtest tests
|-- CHANGELOG.md
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- Makefile
`-- SECURITY.md
```

## Working Agreements

- Keep new features documented at a high level in `docs/architecture.md`.
- Record meaningful architectural decisions in `docs/adr/`.
- Keep research code, production code, and generated data clearly separated.
- Use small pull requests with explicit tests or a clear statement of why tests are not yet applicable.
- Treat secrets, credentials, and production data as out of scope for the repository.

## Next Decisions

1. Pick the first venue and instrument set to target for ingestion and paper trading.
2. Choose deployment/runtime targets for scheduled training and paper trading.
3. Add heavier integration checks and backtest regressions once representative fixtures are stable.

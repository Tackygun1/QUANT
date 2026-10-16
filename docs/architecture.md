# Architecture

## Current State

The repository is now organized as a Python-first ML trading project. The immediate target is a low-cost, student-scale workflow that can ingest exchange-native market data, build a research dataset, train hybrid models, validate them with time-series-safe methods, and run paper or tightly controlled live execution.

## Initial Principles

- Prefer exchange-native data collection over expensive third-party feeds where practical.
- Separate research workflows from execution workflows so experimental code does not leak directly into live trading.
- Design evaluation around walk-forward testing and leakage controls rather than random splits.
- Keep risk controls explicit: pre-trade checks, position limits, throttles, and kill switches.
- Capture decisions while context is fresh.

## Initial Domain Shape

- `src/quant/connectors`: exchange clients and streaming integrations.
- `src/quant/ingestion`: collection jobs for trades, candles, and order-book updates.
- `src/quant/storage`: local persistence and dataset assembly.
- `src/quant/features` and `src/quant/labels`: model-ready tabular and sequence inputs.
- `src/quant/models`: Transformer forecasters, tree ensembles, and hybrid/meta models.
- `src/quant/validation`: walk-forward, purged CV, and metrics for backtest integrity.
- `src/quant/backtesting`: simulation logic, fills, fee models, and evaluation.
- `src/quant/execution`, `src/quant/portfolio`, and `src/quant/risk`: paper/live routing and guardrails.
- `src/quant/monitoring`: alerts, health checks, and runtime observability.

## Pending Decisions

- First exchange and asset universe.
- Python dependency and environment management approach.
- Local storage format and metadata catalog.
- Paper-trading path versus live rollout sequence.
- CI, release, and deployment design.

## When To Update This Document

Update this file when the project gains a runtime, service boundaries, external dependencies, or infrastructure commitments that affect how contributors reason about the system.

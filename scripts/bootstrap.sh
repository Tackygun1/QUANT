#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  .github/ISSUE_TEMPLATE \
  .github/workflows \
  configs/environments \
  configs/features \
  configs/models \
  configs/risk \
  configs/strategies \
  configs/venues \
  data/backtests \
  data/features \
  data/models \
  data/paper \
  data/raw \
  data/staging \
  docs/adr \
  docs/design \
  docs/research \
  docs/runbooks \
  experiments/specs \
  infra/docker \
  infra/systemd \
  notebooks/exploration \
  notebooks/modeling \
  notebooks/validation \
  reports/backtests \
  reports/figures \
  scripts \
  scripts/data \
  scripts/models \
  scripts/trading \
  src \
  src/quant/backtesting \
  src/quant/connectors \
  src/quant/execution \
  src/quant/features \
  src/quant/ingestion \
  src/quant/labels \
  src/quant/models/ensembles \
  src/quant/models/transformers \
  src/quant/models/trees \
  src/quant/monitoring \
  src/quant/portfolio \
  src/quant/risk \
  src/quant/storage \
  src/quant/strategy \
  src/quant/training \
  src/quant/utils \
  src/quant/validation \
  tests/backtests \
  tests/fixtures \
  tests/integration \
  tests/smoke \
  tests/unit

chmod +x scripts/*.sh

printf 'Repository shell is ready.\n'
printf 'Next command: make check\n'

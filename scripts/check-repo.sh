#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_files=(
  ".editorconfig"
  ".gitattributes"
  ".gitignore"
  "CHANGELOG.md"
  "CODE_OF_CONDUCT.md"
  "CONTRIBUTING.md"
  "Makefile"
  "pyproject.toml"
  "README.md"
  "SECURITY.md"
  ".github/CODEOWNERS"
  ".github/pull_request_template.md"
  ".github/ISSUE_TEMPLATE/bug_report.yml"
  ".github/ISSUE_TEMPLATE/config.yml"
  ".github/ISSUE_TEMPLATE/feature_request.yml"
  ".github/workflows/repo-hygiene.yml"
  "configs/README.md"
  "configs/models/meta_models.example.json"
  "configs/models/stacked_dataset.example.json"
  "configs/models/walk_forward_tabular.example.json"
  "configs/models/walk_forward_transformer.example.json"
  "configs/strategies/coinbase_paper_trading.example.json"
  "configs/strategies/model_backtest.example.json"
  "configs/venues/coinbase-historical.example.json"
  "configs/venues/binance-historical.example.json"
  "data/.gitignore"
  "data/README.md"
  "docs/architecture.md"
  "docs/adr/README.md"
  "docs/design/README.md"
  "docs/research/README.md"
  "docs/runbooks/README.md"
  "docs/runbooks/backfill-coinbase-historical.md"
  "docs/runbooks/backfill-binance-historical.md"
  "docs/runbooks/build-market-features.md"
  "docs/runbooks/build-market-labels.md"
  "docs/runbooks/build-stacked-dataset.md"
  "docs/runbooks/build-supervised-dataset.md"
  "docs/runbooks/public-market-data-collection.md"
  "docs/runbooks/raw-to-staged-market-data.md"
  "docs/runbooks/run-coinbase-paper-trader.md"
  "docs/runbooks/run-model-backtest.md"
  "docs/runbooks/run-market-viewer.md"
  "docs/runbooks/run-walk-forward-validation.md"
  "docs/runbooks/train-meta-models.md"
  "docs/runbooks/train-tabular-models.md"
  "docs/runbooks/train-transformer-forecaster.md"
  "experiments/.gitignore"
  "experiments/README.md"
  "infra/README.md"
  "notebooks/README.md"
  "reports/.gitignore"
  "reports/README.md"
  "scripts/README.md"
  "scripts/bootstrap.sh"
  "scripts/check-repo.sh"
  "scripts/data/backfill_coinbase_historical.py"
  "scripts/data/backfill_binance_historical.py"
  "scripts/data/build_market_features.py"
  "scripts/data/build_market_labels.py"
  "scripts/data/build_supervised_dataset.py"
  "scripts/data/run_market_viewer.py"
  "scripts/data/stage_market_data.py"
  "scripts/models/build_stacked_dataset.py"
  "scripts/models/run_model_backtest.py"
  "scripts/models/train_tabular_baselines.py"
  "scripts/models/train_meta_models.py"
  "scripts/models/train_transformer_forecaster.py"
  "scripts/models/run_walk_forward_validation.py"
  "scripts/trading/run_coinbase_paper_trader.py"
  "src/quant/__init__.py"
  "src/quant/backtesting/__init__.py"
  "src/quant/backtesting/model_backtest.py"
  "src/quant/backtesting/model_backtest_cli.py"
  "src/quant/execution/__init__.py"
  "src/quant/execution/paper_trading.py"
  "src/quant/execution/paper_trading_cli.py"
  "src/quant/models/ensembles/__init__.py"
  "src/quant/models/ensembles/stacked_meta.py"
  "src/quant/models/registry.py"
  "src/quant/models/trees/lightgbm.py"
  "src/quant/risk/__init__.py"
  "src/quant/risk/sizing.py"
  "src/quant/ingestion/coinbase_historical.py"
  "src/quant/ingestion/coinbase_historical_cli.py"
  "src/quant/ingestion/binance_historical.py"
  "src/quant/ingestion/binance_historical_cli.py"
  "src/quant/training/stacked_dataset.py"
  "src/quant/training/stacked_dataset_cli.py"
  "src/quant/training/train_meta.py"
  "src/quant/training/train_meta_cli.py"
  "src/quant/utils/__init__.py"
  "src/quant/utils/config.py"
  "src/quant/validation/leakage.py"
  "src/quant/validation/metrics.py"
  "src/quant/validation/walk_forward.py"
  "src/quant/validation/walk_forward_cli.py"
  "tests/README.md"
  "tests/unit/test_artifact_provenance.py"
  "tests/unit/test_config_utils.py"
  "tests/unit/test_coinbase_historical.py"
  "tests/unit/test_binance_historical.py"
  "tests/unit/test_model_backtest.py"
  "tests/unit/test_paper_trading.py"
  "tests/unit/test_validation_leakage.py"
)

required_dirs=(
  ".github"
  ".github/ISSUE_TEMPLATE"
  ".github/workflows"
  "configs"
  "configs/environments"
  "configs/features"
  "configs/models"
  "configs/risk"
  "configs/strategies"
  "configs/venues"
  "data"
  "data/backtests"
  "data/features"
  "data/models"
  "data/paper"
  "data/raw"
  "data/staging"
  "docs"
  "docs/adr"
  "docs/design"
  "docs/research"
  "docs/runbooks"
  "experiments"
  "experiments/specs"
  "infra"
  "infra/docker"
  "infra/systemd"
  "notebooks"
  "notebooks/exploration"
  "notebooks/modeling"
  "notebooks/validation"
  "reports"
  "reports/backtests"
  "reports/figures"
  "scripts"
  "scripts/data"
  "scripts/models"
  "scripts/trading"
  "src"
  "src/quant"
  "src/quant/backtesting"
  "src/quant/connectors"
  "src/quant/execution"
  "src/quant/features"
  "src/quant/ingestion"
  "src/quant/labels"
  "src/quant/models"
  "src/quant/models/ensembles"
  "src/quant/models/transformers"
  "src/quant/models/trees"
  "src/quant/monitoring"
  "src/quant/portfolio"
  "src/quant/risk"
  "src/quant/storage"
  "src/quant/strategy"
  "src/quant/training"
  "src/quant/utils"
  "src/quant/validation"
  "tests"
  "tests/backtests"
  "tests/fixtures"
  "tests/integration"
  "tests/smoke"
  "tests/unit"
)

missing=0

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required file: %s\n' "$path" >&2
    missing=1
  fi
done

for path in "${required_dirs[@]}"; do
  if [[ ! -d "$path" ]]; then
    printf 'Missing required directory: %s\n' "$path" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

if ! grep -q '^# QUANT$' README.md; then
  printf 'README.md should begin with the project title.\n' >&2
  exit 1
fi

for script in scripts/*.sh; do
  if [[ ! -x "$script" ]]; then
    printf 'Script is not executable: %s\n' "$script" >&2
    exit 1
  fi
done

printf 'Repository shell check passed.\n'

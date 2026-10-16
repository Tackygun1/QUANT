from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.oracle_profit_bound import OracleProfitBoundConfig, run_oracle_profit_bound
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  config = OracleProfitBoundConfig(
    backtest_summary_path=Path(config_value(args.backtest_summary_path, payload, "backtest_summary_path")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/oracle_profit_bounds")),
  )
  summary = run_oracle_profit_bound(config)
  logging.getLogger(__name__).info(
    "Oracle profit bound complete. model_return=%s oracle_return=%s output_dir=%s",
    summary.model.total_net_return,
    summary.cost_aware_oracle.total_net_return,
    summary.output_dir,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Compute the cost-aware oracle profit bound for an existing backtest.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--backtest-summary-path")
  parser.add_argument("--output-root")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


if __name__ == "__main__":
  raise SystemExit(main())

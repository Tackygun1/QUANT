from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

from quant.research.strategy_benchmark import (
  DEFAULT_BENCHMARK_SIZING,
  StrategyBenchmarkConfig,
  run_strategy_benchmark,
)
from quant.risk.sizing import PositionSizingConfig
from quant.strategy.basic_strategies import available_basic_strategies
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  config = StrategyBenchmarkConfig(
    artifact_dir=Path(config_value(args.artifact_dir, payload, "artifact_dir")),
    input_root=Path(config_value(args.input_root, payload, "input_root")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/strategy_benchmarks")),
    strategy_names=tuple(config_value(args.strategy_names, payload, "strategy_names", available_basic_strategies())),
    strategy_params=payload.get("strategy_params"),
    dataset_name=config_value(args.dataset_name, payload, "dataset_name"),
    realized_return_column=config_value(args.realized_return_column, payload, "realized_return_column", "forward_return_1_bar"),
    fee_bps=float(args.fee_bps if args.fee_bps is not None else payload.get("fee_bps", 1.0)),
    slippage_bps=float(args.slippage_bps if args.slippage_bps is not None else payload.get("slippage_bps", 0.5)),
    sizing=PositionSizingConfig(**payload.get("sizing", asdict(DEFAULT_BENCHMARK_SIZING))),
    config_path=args.config,
  )

  summary = run_strategy_benchmark(config)
  logging.getLogger(__name__).info(
    "Strategy benchmark complete. benchmark_dir=%s best_entry=%s best_entry_type=%s",
    summary.benchmark_dir,
    summary.best_entry_name,
    summary.best_entry_type,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Benchmark a trained model against basic rule-based strategies.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--artifact-dir")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--dataset-name")
  parser.add_argument("--realized-return-column")
  parser.add_argument("--fee-bps", type=float)
  parser.add_argument("--slippage-bps", type=float)
  parser.add_argument("--strategy", dest="strategy_names", action="append")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.features.market_features import FeatureBuildConfig, build_market_features


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)

  config = FeatureBuildConfig(
    input_root=args.input_root,
    output_root=args.output_root,
    providers=tuple(args.providers or ()),
    symbols=tuple(args.symbols or ()),
    limit_datasets=args.limit_datasets,
  )

  summary = build_market_features(config)
  logging.getLogger(__name__).info(
    "Feature build complete. datasets_processed=%s output_files_written=%s feature_rows_written=%s",
    summary.datasets_processed,
    summary.output_files_written,
    summary.feature_rows_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Build candle-anchored market feature tables from staged market data.",
  )
  parser.add_argument(
    "--input-root",
    type=Path,
    default=Path("data/staging"),
    help="Root directory containing staged CSV tables.",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data/features"),
    help="Root directory for feature tables.",
  )
  parser.add_argument(
    "--provider",
    dest="providers",
    action="append",
    help="Optional provider filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--symbol",
    dest="symbols",
    action="append",
    help="Optional symbol filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--limit-datasets",
    type=int,
    help="Optional cap on the number of provider/symbol/day datasets to process.",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    help="Logging level. Defaults to INFO.",
  )
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

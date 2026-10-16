from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.training.supervised_dataset import SupervisedDatasetConfig, build_supervised_datasets


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)

  config = SupervisedDatasetConfig(
    input_root=args.input_root,
    output_root=args.output_root,
    providers=tuple(args.providers or ()),
    symbols=tuple(args.symbols or ()),
    limit_datasets=args.limit_datasets,
  )
  summary = build_supervised_datasets(config)
  logging.getLogger(__name__).info(
    "Supervised dataset build complete. datasets_processed=%s output_files_written=%s joined_rows_written=%s",
    summary.datasets_processed,
    summary.output_files_written,
    summary.joined_rows_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Join market feature and label tables into a supervised training dataset.",
  )
  parser.add_argument("--input-root", type=Path, default=Path("data/features"))
  parser.add_argument("--output-root", type=Path, default=Path("data/features"))
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--limit-datasets", type=int)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

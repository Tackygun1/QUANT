from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.ingestion.staging import StageConfig, run_stager


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)

  config = StageConfig(
    input_root=args.input_root,
    output_root=args.output_root,
    providers=tuple(args.providers or ()),
    channels=tuple(args.channels or ()),
    symbols=tuple(args.symbols or ()),
    limit_files=args.limit_files,
  )

  summary = run_stager(config)
  logging.getLogger(__name__).info(
    "Staging complete. files_processed=%s output_files_written=%s rows_by_table=%s",
    summary.files_processed,
    summary.output_files_written,
    dict(summary.rows_by_table),
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Convert raw market-data JSONL files into staged CSV tables.",
  )
  parser.add_argument(
    "--input-root",
    type=Path,
    default=Path("data/raw"),
    help="Root directory containing raw JSONL partitions.",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data/staging"),
    help="Root directory for staged CSV partitions.",
  )
  parser.add_argument(
    "--provider",
    dest="providers",
    action="append",
    help="Optional provider filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--channel",
    dest="channels",
    action="append",
    help="Optional raw channel filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--symbol",
    dest="symbols",
    action="append",
    help="Optional symbol filter. Repeat for multiple values.",
  )
  parser.add_argument(
    "--limit-files",
    type=int,
    help="Optional cap on the number of raw partition files to process.",
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

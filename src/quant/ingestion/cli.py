from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from quant.ingestion.collector import CollectorConfig, run_collector


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)

  payload = _load_payload(args.config)
  provider = args.provider or payload.get("provider")
  products = tuple(args.products or payload.get("products", []))
  channels = tuple(args.channels or payload.get("channels", []))
  output_root = Path(args.output_root or payload.get("output_root", "data/raw"))
  candle_interval = args.candle_interval or payload.get("candle_interval", "1m")
  max_backoff_seconds = float(
    args.max_backoff_seconds or payload.get("max_backoff_seconds", 30.0)
  )
  max_records = args.max_records if args.max_records is not None else payload.get("max_records")

  if provider is None:
    raise SystemExit("A provider is required. Use --provider or pass one in --config.")
  if not products:
    raise SystemExit("At least one --product is required, or include products in --config.")
  if not channels:
    raise SystemExit("At least one --channel is required, or include channels in --config.")

  config = CollectorConfig(
    provider=provider,
    products=products,
    channels=channels,
    output_root=output_root,
    candle_interval=candle_interval,
    max_backoff_seconds=max_backoff_seconds,
    max_records=max_records,
  )

  try:
    asyncio.run(run_collector(config))
  except KeyboardInterrupt:
    logging.getLogger(__name__).info("Collector interrupted by user.")
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Collect continuous free public crypto market data into partitioned JSONL files.",
  )
  parser.add_argument(
    "--config",
    type=Path,
    help="Path to a JSON collector config file.",
  )
  parser.add_argument(
    "--provider",
    choices=("binance", "coinbase"),
    help="Public market data provider.",
  )
  parser.add_argument(
    "--product",
    dest="products",
    action="append",
    help="Product or symbol to collect. Repeat the flag for multiple values.",
  )
  parser.add_argument(
    "--channel",
    dest="channels",
    action="append",
    help="Channel to collect. Repeat the flag for multiple values.",
  )
  parser.add_argument(
    "--output-root",
    help="Base directory for raw market data.",
  )
  parser.add_argument(
    "--candle-interval",
    help="Candlestick interval for Binance. Coinbase candles use fixed 5-minute buckets.",
  )
  parser.add_argument(
    "--max-backoff-seconds",
    type=float,
    help="Maximum reconnect backoff.",
  )
  parser.add_argument(
    "--max-records",
    type=int,
    help="Optional cap for test runs so the collector exits after N records.",
  )
  parser.add_argument(
    "--log-level",
    default="INFO",
    help="Logging level. Defaults to INFO.",
  )
  return parser


def _load_payload(path: Path | None) -> dict[str, Any]:
  if path is None:
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

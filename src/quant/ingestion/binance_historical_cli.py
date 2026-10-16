from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from quant.ingestion.binance_historical import (
  BinanceHistoricalBackfillConfig,
  run_binance_historical_backfill,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  config = BinanceHistoricalBackfillConfig(
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    channels=tuple(config_value(args.channels, payload, "channels", ("candles",))),
    start_time=_parse_dt(config_value(args.start_time, payload, "start_time")),
    end_time=_parse_dt(config_value(args.end_time, payload, "end_time")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/raw")),
    candle_interval=config_value(args.candle_interval, payload, "candle_interval", "1m"),
    request_timeout_seconds=float(config_value(args.request_timeout_seconds, payload, "request_timeout_seconds", 20.0)),
    request_pause_seconds=float(config_value(args.request_pause_seconds, payload, "request_pause_seconds", 0.05)),
  )

  summary = run_binance_historical_backfill(config)
  logging.getLogger(__name__).info(
    "Historical backfill complete. requests_made=%s records_written=%s rows_by_channel=%s",
    summary.requests_made,
    summary.records_written,
    summary.rows_by_channel,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Backfill free historical Binance spot data into raw JSONL partitions.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--channel", dest="channels", action="append")
  parser.add_argument("--start-time")
  parser.add_argument("--end-time")
  parser.add_argument("--output-root")
  parser.add_argument("--candle-interval")
  parser.add_argument("--request-timeout-seconds", type=float)
  parser.add_argument("--request-pause-seconds", type=float)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)

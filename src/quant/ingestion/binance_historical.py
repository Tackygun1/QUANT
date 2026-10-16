from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from quant.connectors.public_market_data import build_record, iso_from_unix_millis, utc_now_iso
from quant.storage.jsonl import PartitionedJsonlWriter

LOGGER = logging.getLogger(__name__)

REST_BASE_URL = "https://api.binance.com"
SUPPORTED_CHANNELS = frozenset({"candles", "trades"})
KLINES_LIMIT = 1000
AGGTRADES_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class BinanceHistoricalBackfillConfig:
  symbols: tuple[str, ...]
  channels: tuple[str, ...]
  start_time: datetime
  end_time: datetime
  output_root: Path = Path("data/raw")
  candle_interval: str = "1m"
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.05

  def __post_init__(self) -> None:
    if not self.symbols:
      raise ValueError("At least one symbol is required for historical backfill.")
    if not self.channels:
      raise ValueError("At least one channel is required for historical backfill.")
    unsupported = sorted(set(self.channels) - SUPPORTED_CHANNELS)
    if unsupported:
      raise ValueError(
        f"Unsupported historical channels: {', '.join(unsupported)}. "
        f"Supported channels: {', '.join(sorted(SUPPORTED_CHANNELS))}."
      )
    if self.end_time <= self.start_time:
      raise ValueError("end_time must be after start_time.")


@dataclass(slots=True)
class BinanceHistoricalBackfillSummary:
  requests_made: int = 0
  records_written: int = 0
  rows_by_channel: dict[str, int] = field(default_factory=dict)


def run_binance_historical_backfill(config: BinanceHistoricalBackfillConfig) -> BinanceHistoricalBackfillSummary:
  writer = PartitionedJsonlWriter(config.output_root)
  summary = BinanceHistoricalBackfillSummary()
  rows_by_channel: dict[str, int] = {}

  try:
    for symbol in config.symbols:
      for channel in config.channels:
        LOGGER.info(
          "Backfilling provider=binance symbol=%s channel=%s start=%s end=%s",
          symbol,
          channel,
          _format_iso(config.start_time),
          _format_iso(config.end_time),
        )
        if channel == "candles":
          requests, records = _backfill_klines(symbol, config, writer)
        elif channel == "trades":
          requests, records = _backfill_agg_trades(symbol, config, writer)
        else:  # pragma: no cover
          raise ValueError(f"Unsupported channel: {channel}")

        summary.requests_made += requests
        summary.records_written += records
        rows_by_channel[channel] = rows_by_channel.get(channel, 0) + records
    summary.rows_by_channel = rows_by_channel
    return summary
  finally:
    writer.close()


def _backfill_klines(
  symbol: str,
  config: BinanceHistoricalBackfillConfig,
  writer: PartitionedJsonlWriter,
) -> tuple[int, int]:
  interval_ms = _interval_to_milliseconds(config.candle_interval)
  start_ms = _to_unix_millis(config.start_time)
  end_ms = _to_unix_millis(config.end_time)
  cursor = start_ms
  requests_made = 0
  records_written = 0

  while cursor < end_ms:
    payload = _http_get_json(
      "/api/v3/klines",
      {
        "symbol": symbol,
        "interval": config.candle_interval,
        "startTime": cursor,
        "endTime": end_ms - 1,
        "limit": KLINES_LIMIT,
      },
      timeout_seconds=config.request_timeout_seconds,
    )
    requests_made += 1
    rows = payload if isinstance(payload, list) else []
    if not rows:
      break

    for row in rows:
      record = kline_to_record(
        symbol=symbol,
        interval=config.candle_interval,
        row=row,
      )
      writer.write(record)
      records_written += 1

    last_open_time = int(rows[-1][0])
    next_cursor = last_open_time + interval_ms
    if next_cursor <= cursor:
      break
    cursor = next_cursor
    time.sleep(config.request_pause_seconds)

  return requests_made, records_written


def _backfill_agg_trades(
  symbol: str,
  config: BinanceHistoricalBackfillConfig,
  writer: PartitionedJsonlWriter,
) -> tuple[int, int]:
  start_ms = _to_unix_millis(config.start_time)
  end_ms = _to_unix_millis(config.end_time)
  requests_made = 0
  records_written = 0

  initial_rows = _http_get_json(
    "/api/v3/aggTrades",
    {
      "symbol": symbol,
      "startTime": start_ms,
      "endTime": end_ms - 1,
      "limit": AGGTRADES_LIMIT,
    },
    timeout_seconds=config.request_timeout_seconds,
  )
  requests_made += 1
  rows = initial_rows if isinstance(initial_rows, list) else []
  if not rows:
    return requests_made, records_written

  next_from_id: int | None = None
  while rows:
    for row in rows:
      trade_time = int(row["T"])
      if trade_time >= end_ms:
        return requests_made, records_written
      record = agg_trade_to_record(symbol=symbol, row=row)
      writer.write(record)
      records_written += 1

    next_from_id = int(rows[-1]["a"]) + 1
    rows = _http_get_json(
      "/api/v3/aggTrades",
      {
        "symbol": symbol,
        "fromId": next_from_id,
        "limit": AGGTRADES_LIMIT,
      },
      timeout_seconds=config.request_timeout_seconds,
    )
    requests_made += 1
    rows = rows if isinstance(rows, list) else []
    time.sleep(config.request_pause_seconds)

  return requests_made, records_written


def kline_to_record(
  *,
  symbol: str,
  interval: str,
  row: list[Any],
) -> dict[str, Any]:
  received_at = utc_now_iso()
  open_time = int(row[0])
  close_time = int(row[6])
  payload = {
    "open_time": open_time,
    "open": row[1],
    "high": row[2],
    "low": row[3],
    "close": row[4],
    "volume": row[5],
    "close_time": close_time,
    "quote_volume": row[7],
    "trade_count": row[8],
    "taker_buy_base_volume": row[9],
    "taker_buy_quote_volume": row[10],
  }
  return build_record(
    provider="binance",
    channel="candles",
    source_channel=f"rest_klines_{interval}",
    symbol=symbol,
    record_type="candle",
    received_at=received_at,
    event_time=iso_from_unix_millis(open_time),
    payload=payload,
    extra={
      "interval": interval,
      "is_closed": True,
      "candle_start": iso_from_unix_millis(open_time),
      "candle_end": iso_from_unix_millis(close_time),
      "open": row[1],
      "high": row[2],
      "low": row[3],
      "close": row[4],
      "volume": row[5],
      "quote_volume": row[7],
      "trade_count": row[8],
      "sequence_num": open_time,
      "update_type": "historical_backfill",
    },
  )


def agg_trade_to_record(
  *,
  symbol: str,
  row: dict[str, Any],
) -> dict[str, Any]:
  received_at = utc_now_iso()
  taker_side = "SELL" if row.get("m") else "BUY"
  return build_record(
    provider="binance",
    channel="trades",
    source_channel="rest_aggTrades",
    symbol=symbol,
    record_type="trade",
    received_at=received_at,
    event_time=iso_from_unix_millis(row["T"]),
    payload=row,
    extra={
      "trade_id": row.get("a"),
      "price": row.get("p"),
      "size": row.get("q"),
      "taker_side": taker_side,
      "is_buyer_maker": row.get("m"),
      "sequence_num": row.get("a"),
      "update_type": "historical_backfill",
    },
  )


def _http_get_json(path: str, params: dict[str, Any], *, timeout_seconds: float) -> Any:
  query = urlencode({key: value for key, value in params.items() if value is not None})
  url = f"{REST_BASE_URL}{path}?{query}"
  with urlopen(url, timeout=timeout_seconds) as response:
    return json.loads(response.read().decode("utf-8"))


def _interval_to_milliseconds(interval: str) -> int:
  suffix = interval[-1]
  value = int(interval[:-1])
  if suffix == "s":
    return value * 1000
  if suffix == "m":
    return value * 60_000
  if suffix == "h":
    return value * 3_600_000
  if suffix == "d":
    return value * 86_400_000
  if suffix == "w":
    return value * 7 * 86_400_000
  if suffix == "M":
    return value * 30 * 86_400_000
  raise ValueError(f"Unsupported interval: {interval}")


def _to_unix_millis(value: datetime) -> int:
  return int(value.astimezone(UTC).timestamp() * 1000)


def _format_iso(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

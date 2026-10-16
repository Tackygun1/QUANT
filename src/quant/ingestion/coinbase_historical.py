from __future__ import annotations

import json
import logging
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from quant.connectors.public_market_data import build_record, iso_from_unix_seconds, utc_now_iso
from quant.storage.jsonl import PartitionedJsonlWriter

LOGGER = logging.getLogger(__name__)

REST_BASE_URL = "https://api.exchange.coinbase.com"
SUPPORTED_CHANNELS = frozenset({"candles", "trades"})
MAX_CANDLES_PER_REQUEST = 300
MAX_TRADES_PER_REQUEST = 1000
HTTP_HEADERS = {
  "Accept": "application/json",
  "User-Agent": "quant/0.1",
}

INTERVAL_SECONDS = {
  "1m": 60,
  "5m": 300,
  "15m": 900,
  "1h": 3600,
  "6h": 21600,
  "1d": 86400,
}


@dataclass(frozen=True, slots=True)
class CoinbaseHistoricalBackfillConfig:
  products: tuple[str, ...]
  channels: tuple[str, ...]
  start_time: datetime
  end_time: datetime
  output_root: Path = Path("data/raw")
  candle_interval: str = "5m"
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.1
  request_retry_attempts: int = 5
  request_retry_backoff_seconds: float = 1.0

  def __post_init__(self) -> None:
    if not self.products:
      raise ValueError("At least one Coinbase product is required for historical backfill.")
    if not self.channels:
      raise ValueError("At least one channel is required for historical backfill.")
    unsupported = sorted(set(self.channels) - SUPPORTED_CHANNELS)
    if unsupported:
      raise ValueError(
        f"Unsupported Coinbase historical channels: {', '.join(unsupported)}. "
        f"Supported channels: {', '.join(sorted(SUPPORTED_CHANNELS))}."
      )
    if self.candle_interval not in INTERVAL_SECONDS:
      supported = ", ".join(sorted(INTERVAL_SECONDS))
      raise ValueError(f"Unsupported Coinbase candle interval: {self.candle_interval}. Supported: {supported}.")
    if self.end_time <= self.start_time:
      raise ValueError("end_time must be after start_time.")
    if self.request_retry_attempts < 1:
      raise ValueError("request_retry_attempts must be at least 1.")
    if self.request_retry_backoff_seconds < 0:
      raise ValueError("request_retry_backoff_seconds must be non-negative.")


@dataclass(slots=True)
class CoinbaseHistoricalBackfillSummary:
  requests_made: int = 0
  records_written: int = 0
  rows_by_channel: dict[str, int] = field(default_factory=dict)


def run_coinbase_historical_backfill(config: CoinbaseHistoricalBackfillConfig) -> CoinbaseHistoricalBackfillSummary:
  writer = PartitionedJsonlWriter(config.output_root)
  summary = CoinbaseHistoricalBackfillSummary()
  rows_by_channel: dict[str, int] = {}

  try:
    for product in config.products:
      for channel in config.channels:
        LOGGER.info(
          "Backfilling provider=coinbase product=%s channel=%s start=%s end=%s interval=%s",
          product,
          channel,
          _format_iso(config.start_time),
          _format_iso(config.end_time),
          config.candle_interval,
        )
        if channel == "candles":
          requests, records = _backfill_candles(product, config, writer)
        elif channel == "trades":
          requests, records = _backfill_trades(product, config, writer)
        else:  # pragma: no cover
          raise ValueError(f"Unsupported channel: {channel}")

        summary.requests_made += requests
        summary.records_written += records
        rows_by_channel[channel] = rows_by_channel.get(channel, 0) + records

    summary.rows_by_channel = rows_by_channel
    return summary
  finally:
    writer.close()


def _backfill_candles(
  product: str,
  config: CoinbaseHistoricalBackfillConfig,
  writer: PartitionedJsonlWriter,
) -> tuple[int, int]:
  granularity_seconds = interval_to_seconds(config.candle_interval)
  # Coinbase rejects requests over 300 points. Using 299 intervals keeps the request
  # comfortably within the documented cap even if the endpoint treats both edges as inclusive.
  max_span = timedelta(seconds=granularity_seconds * (MAX_CANDLES_PER_REQUEST - 1))
  cursor = config.start_time.astimezone(UTC)
  end_time = config.end_time.astimezone(UTC)
  requests_made = 0
  records_written = 0
  seen_starts: set[int] = set()

  while cursor < end_time:
    chunk_end = min(end_time, cursor + max_span)
    rows, _headers = _http_get_json(
      path=f"/products/{quote(product, safe='')}/candles",
      params={
        "granularity": granularity_seconds,
        "start": _format_iso(cursor),
        "end": _format_iso(chunk_end),
      },
      timeout_seconds=config.request_timeout_seconds,
      retry_attempts=config.request_retry_attempts,
      retry_backoff_seconds=config.request_retry_backoff_seconds,
    )
    requests_made += 1

    for row in sorted(_coerce_candle_rows(rows), key=lambda item: int(item[0])):
      start_seconds = int(row[0])
      candle_start = datetime.fromtimestamp(start_seconds, tz=UTC)
      if candle_start < cursor or candle_start >= chunk_end:
        continue
      if start_seconds in seen_starts:
        continue

      writer.write(
        coinbase_candle_to_record(
          product=product,
          interval=config.candle_interval,
          granularity_seconds=granularity_seconds,
          row=row,
        )
      )
      records_written += 1
      seen_starts.add(start_seconds)

    cursor = chunk_end
    if cursor < end_time:
      time.sleep(config.request_pause_seconds)

  return requests_made, records_written


def _backfill_trades(
  product: str,
  config: CoinbaseHistoricalBackfillConfig,
  writer: PartitionedJsonlWriter,
) -> tuple[int, int]:
  start_time = config.start_time.astimezone(UTC)
  end_time = config.end_time.astimezone(UTC)
  after_cursor: str | None = None
  requests_made = 0
  records_written = 0

  while True:
    rows, headers = _http_get_json(
      path=f"/products/{quote(product, safe='')}/trades",
      params={
        "limit": MAX_TRADES_PER_REQUEST,
        "after": after_cursor,
      },
      timeout_seconds=config.request_timeout_seconds,
      retry_attempts=config.request_retry_attempts,
      retry_backoff_seconds=config.request_retry_backoff_seconds,
    )
    requests_made += 1

    trade_rows = _coerce_trade_rows(rows)
    if not trade_rows:
      break

    for row in trade_rows:
      trade_dt = _parse_trade_time(row["time"])
      if trade_dt > end_time:
        continue
      if trade_dt < start_time:
        break

      writer.write(coinbase_trade_to_record(product=product, row=row))
      records_written += 1

    if requests_made % 25 == 0:
      LOGGER.info(
        "Coinbase trade backfill progress product=%s requests=%s records=%s oldest_trade=%s",
        product,
        requests_made,
        records_written,
        trade_rows[-1].get("time", ""),
      )

    oldest_trade_dt = _parse_trade_time(trade_rows[-1]["time"])
    if oldest_trade_dt < start_time:
      break

    next_after = headers.get("CB-AFTER")
    if not next_after or next_after == after_cursor:
      break
    after_cursor = next_after
    time.sleep(config.request_pause_seconds)

  return requests_made, records_written


def coinbase_candle_to_record(
  *,
  product: str,
  interval: str,
  granularity_seconds: int,
  row: list[Any],
) -> dict[str, Any]:
  received_at = utc_now_iso()
  start_seconds = int(row[0])
  candle_start = iso_from_unix_seconds(start_seconds)
  candle_end = iso_from_unix_seconds(start_seconds + granularity_seconds)
  payload = {
    "time": start_seconds,
    "low": row[1],
    "high": row[2],
    "open": row[3],
    "close": row[4],
    "volume": row[5],
    "granularity": granularity_seconds,
  }
  return build_record(
    provider="coinbase",
    channel="candles",
    source_channel=f"rest_candles_{interval}",
    symbol=product,
    record_type="candle",
    received_at=received_at,
    event_time=candle_start,
    payload=payload,
    extra={
      "interval": interval,
      "is_closed": True,
      "candle_start": candle_start,
      "candle_end": candle_end,
      "open": row[3],
      "high": row[2],
      "low": row[1],
      "close": row[4],
      "volume": row[5],
      "sequence_num": start_seconds,
      "update_type": "historical_backfill",
    },
  )


def coinbase_trade_to_record(
  *,
  product: str,
  row: dict[str, Any],
) -> dict[str, Any]:
  received_at = utc_now_iso()
  maker_side = str(row.get("side", "")).upper()
  taker_side = _invert_side(maker_side)
  trade_id = row.get("trade_id")

  return build_record(
    provider="coinbase",
    channel="trades",
    source_channel="rest_trades",
    symbol=product,
    record_type="trade",
    received_at=received_at,
    event_time=row.get("time"),
    payload=row,
    extra={
      "trade_id": trade_id,
      "price": row.get("price"),
      "size": row.get("size"),
      "maker_side": maker_side,
      "taker_side": taker_side,
      "sequence_num": trade_id,
      "update_type": "historical_backfill",
    },
  )


def interval_to_seconds(interval: str) -> int:
  try:
    return INTERVAL_SECONDS[interval]
  except KeyError as exc:
    raise ValueError(f"Unsupported Coinbase candle interval: {interval}") from exc


def _coerce_candle_rows(payload: Any) -> list[list[Any]]:
  if not isinstance(payload, list):
    return []
  return [row for row in payload if isinstance(row, list) and len(row) >= 6]


def _coerce_trade_rows(payload: Any) -> list[dict[str, Any]]:
  if not isinstance(payload, list):
    return []
  return [
    row
    for row in payload
    if isinstance(row, dict) and all(key in row for key in ("trade_id", "time", "price", "size"))
  ]


def _http_get_json(
  path: str,
  params: dict[str, Any],
  *,
  timeout_seconds: float,
  retry_attempts: int = 5,
  retry_backoff_seconds: float = 1.0,
) -> tuple[Any, dict[str, str]]:
  query = urlencode({key: value for key, value in params.items() if value is not None})
  request = Request(
    f"{REST_BASE_URL}{path}?{query}",
    headers=HTTP_HEADERS,
  )
  last_error: BaseException | None = None

  for attempt in range(1, retry_attempts + 1):
    try:
      with urlopen(request, timeout=timeout_seconds) as response:
        headers = {key.upper(): value for key, value in response.headers.items()}
        return json.loads(response.read().decode("utf-8")), headers
    except HTTPError as exc:
      if exc.code not in {408, 429} and exc.code < 500:
        raise
      last_error = exc
    except (TimeoutError, socket.timeout, URLError) as exc:
      last_error = exc

    if attempt >= retry_attempts:
      assert last_error is not None
      raise last_error

    pause_seconds = retry_backoff_seconds * attempt
    LOGGER.warning(
      "Coinbase historical request failed path=%s attempt=%s/%s error=%s; retrying in %.1fs",
      path,
      attempt,
      retry_attempts,
      last_error,
      pause_seconds,
    )
    if pause_seconds > 0:
      time.sleep(pause_seconds)

  raise RuntimeError("Coinbase historical request retry loop exited unexpectedly.")


def _parse_trade_time(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _invert_side(maker_side: str) -> str:
  if maker_side == "BUY":
    return "SELL"
  if maker_side == "SELL":
    return "BUY"
  return ""


def _format_iso(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

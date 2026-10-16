from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)

TRADES_FIELDS = (
  "provider",
  "symbol",
  "event_time",
  "received_at",
  "source_channel",
  "record_type",
  "trade_id",
  "price",
  "size",
  "notional_quote",
  "aggressor_side",
  "is_buyer_maker",
  "sequence_num",
  "update_type",
)

CANDLES_FIELDS = (
  "provider",
  "symbol",
  "event_time",
  "received_at",
  "source_channel",
  "record_type",
  "interval",
  "candle_start",
  "candle_end",
  "is_closed",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "quote_volume",
  "trade_count",
  "sequence_num",
  "update_type",
)

ORDER_BOOK_TOP_FIELDS = (
  "provider",
  "symbol",
  "event_time",
  "received_at",
  "source_channel",
  "record_type",
  "best_bid",
  "best_bid_size",
  "best_ask",
  "best_ask_size",
  "mid_price",
  "spread",
  "sequence_num",
  "update_type",
)

ORDER_BOOK_EVENTS_FIELDS = (
  "provider",
  "symbol",
  "event_time",
  "received_at",
  "source_channel",
  "record_type",
  "update_kind",
  "sequence_num",
  "update_type",
  "last_update_id",
  "update_count",
  "bid_levels",
  "ask_levels",
)

ORDER_BOOK_LEVELS_FIELDS = (
  "provider",
  "symbol",
  "event_time",
  "received_at",
  "source_channel",
  "record_type",
  "update_kind",
  "level_side",
  "level_index",
  "price",
  "quantity",
  "update_action",
  "sequence_num",
  "update_type",
  "last_update_id",
)

TABLE_SCHEMAS = {
  "trades": TRADES_FIELDS,
  "candles": CANDLES_FIELDS,
  "order_book_top": ORDER_BOOK_TOP_FIELDS,
  "order_book_events": ORDER_BOOK_EVENTS_FIELDS,
  "order_book_levels": ORDER_BOOK_LEVELS_FIELDS,
}


@dataclass(frozen=True, slots=True)
class StageConfig:
  input_root: Path = Path("data/raw")
  output_root: Path = Path("data/staging")
  providers: tuple[str, ...] = ()
  channels: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  limit_files: int | None = None


@dataclass(frozen=True, slots=True)
class RawPartition:
  provider: str
  channel: str
  symbol: str
  year: str
  month: str
  day: str
  hour: str
  path: Path


@dataclass(slots=True)
class StageSummary:
  files_processed: int = 0
  output_files_written: int = 0
  rows_by_table: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def run_stager(config: StageConfig) -> StageSummary:
  summary = StageSummary()

  for index, raw_file in enumerate(discover_raw_files(config), start=1):
    if config.limit_files is not None and index > config.limit_files:
      break

    row_groups = _stage_raw_file(raw_file, config.output_root)
    summary.files_processed += 1

    for table_name, rows in row_groups.items():
      if not rows:
        continue
      summary.output_files_written += 1
      summary.rows_by_table[table_name] = summary.rows_by_table.get(table_name, 0) + len(rows)

  return summary


def discover_raw_files(config: StageConfig) -> list[RawPartition]:
  raw_files: list[RawPartition] = []
  if not config.input_root.exists():
    LOGGER.warning("Input root does not exist: %s", config.input_root)
    return raw_files

  provider_filter = set(config.providers)
  channel_filter = set(config.channels)
  symbol_filter = set(config.symbols)

  for path in sorted(config.input_root.rglob("*.jsonl")):
    relative_parts = path.relative_to(config.input_root).parts
    if len(relative_parts) != 7:
      LOGGER.debug("Skipping unexpected raw path: %s", path)
      continue

    provider, channel, symbol, year, month, day, filename = relative_parts
    hour = Path(filename).stem

    if provider_filter and provider not in provider_filter:
      continue
    if channel_filter and channel not in channel_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue

    raw_files.append(
      RawPartition(
        provider=provider,
        channel=channel,
        symbol=symbol,
        year=year,
        month=month,
        day=day,
        hour=hour,
        path=path,
      )
    )

  return raw_files


def _stage_raw_file(
  raw_file: RawPartition,
  output_root: Path,
) -> dict[str, list[dict[str, Any]]]:
  row_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

  with raw_file.path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      line = line.strip()
      if not line:
        continue

      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        LOGGER.warning("Skipping invalid JSON in %s:%s: %s", raw_file.path, line_number, exc)
        continue

      try:
        for table_name, row in iter_staged_rows(record):
          row_groups[table_name].append(row)
      except Exception as exc:
        LOGGER.warning(
          "Skipping malformed record in %s:%s due to parser error: %s",
          raw_file.path,
          line_number,
          exc,
        )

  for table_name, rows in row_groups.items():
    if not rows:
      continue
    output_path = staged_output_path(output_root, table_name, raw_file)
    write_csv_rows(output_path, TABLE_SCHEMAS[table_name], rows)
    LOGGER.info("Wrote %s rows to %s", len(rows), output_path)

  return row_groups


def staged_output_path(output_root: Path, table_name: str, raw_file: RawPartition) -> Path:
  return (
    output_root
    / table_name
    / raw_file.provider
    / raw_file.symbol
    / raw_file.year
    / raw_file.month
    / raw_file.day
    / f"{raw_file.hour}.csv"
  )


def write_csv_rows(
  path: Path,
  fieldnames: Iterable[str],
  rows: list[dict[str, Any]],
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow({key: _csv_safe(row.get(key)) for key in writer.fieldnames})


def iter_staged_rows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
  channel = record.get("channel")
  if channel == "trades":
    return [("trades", _trade_row(record))]
  if channel == "candles":
    return [("candles", _candle_row(record))]
  if channel == "ticker" or channel == "book_ticker":
    return [("order_book_top", _order_book_top_row(record))]
  if channel == "level2":
    return _level2_rows(record)
  return []


def _trade_row(record: dict[str, Any]) -> dict[str, Any]:
  price = record.get("price")
  size = record.get("size")
  aggressor_side = record.get("side") or record.get("taker_side")

  return {
    "provider": record.get("provider"),
    "symbol": record.get("symbol"),
    "event_time": record.get("event_time"),
    "received_at": record.get("received_at"),
    "source_channel": record.get("source_channel"),
    "record_type": record.get("record_type"),
    "trade_id": record.get("trade_id"),
    "price": price,
    "size": size,
    "notional_quote": _decimal_product(price, size),
    "aggressor_side": aggressor_side,
    "is_buyer_maker": record.get("is_buyer_maker"),
    "sequence_num": record.get("sequence_num"),
    "update_type": record.get("update_type"),
  }


def _candle_row(record: dict[str, Any]) -> dict[str, Any]:
  payload = record.get("payload", {})
  if isinstance(payload, dict) and "k" in payload:
    kline = payload.get("k", {})
    quote_volume = kline.get("q")
    trade_count = kline.get("n")
  else:
    quote_volume = payload.get("quote_volume") if isinstance(payload, dict) else None
    trade_count = payload.get("trade_count") if isinstance(payload, dict) else None

  return {
    "provider": record.get("provider"),
    "symbol": record.get("symbol"),
    "event_time": record.get("event_time"),
    "received_at": record.get("received_at"),
    "source_channel": record.get("source_channel"),
    "record_type": record.get("record_type"),
    "interval": record.get("interval") or "",
    "candle_start": record.get("candle_start") or record.get("event_time"),
    "candle_end": record.get("candle_end") or "",
    "is_closed": record.get("is_closed"),
    "open": record.get("open"),
    "high": record.get("high"),
    "low": record.get("low"),
    "close": record.get("close"),
    "volume": record.get("volume"),
    "quote_volume": quote_volume,
    "trade_count": trade_count,
    "sequence_num": record.get("sequence_num"),
    "update_type": record.get("update_type"),
  }


def _order_book_top_row(record: dict[str, Any]) -> dict[str, Any]:
  payload = record.get("payload", {})
  best_bid = record.get("best_bid") or _payload_value(payload, "best_bid") or _payload_value(payload, "b")
  best_ask = record.get("best_ask") or _payload_value(payload, "best_ask") or _payload_value(payload, "a")
  best_bid_size = (
    record.get("best_bid_size")
    or _payload_value(payload, "best_bid_size")
    or _payload_value(payload, "best_bid_quantity")
    or _payload_value(payload, "B")
  )
  best_ask_size = (
    record.get("best_ask_size")
    or _payload_value(payload, "best_ask_size")
    or _payload_value(payload, "best_ask_quantity")
    or _payload_value(payload, "A")
  )

  return {
    "provider": record.get("provider"),
    "symbol": record.get("symbol"),
    "event_time": record.get("event_time"),
    "received_at": record.get("received_at"),
    "source_channel": record.get("source_channel"),
    "record_type": record.get("record_type"),
    "best_bid": best_bid,
    "best_bid_size": best_bid_size,
    "best_ask": best_ask,
    "best_ask_size": best_ask_size,
    "mid_price": _decimal_midpoint(best_bid, best_ask),
    "spread": _decimal_difference(best_ask, best_bid),
    "sequence_num": record.get("sequence_num"),
    "update_type": record.get("update_type"),
  }


def _level2_rows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
  provider = record.get("provider")
  if provider == "coinbase":
    return _coinbase_level2_rows(record)
  if provider == "binance":
    return _binance_level2_rows(record)
  return []


def _coinbase_level2_rows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
  payload = record.get("payload", {})
  updates = payload.get("updates", []) if isinstance(payload, dict) else []
  base_row = {
    "provider": record.get("provider"),
    "symbol": record.get("symbol"),
    "event_time": record.get("event_time"),
    "received_at": record.get("received_at"),
    "source_channel": record.get("source_channel"),
    "record_type": record.get("record_type"),
    "update_kind": "delta",
    "sequence_num": record.get("sequence_num"),
    "update_type": record.get("update_type"),
    "last_update_id": "",
    "update_count": len(updates),
    "bid_levels": "",
    "ask_levels": "",
  }

  rows: list[tuple[str, dict[str, Any]]] = [("order_book_events", base_row)]
  for index, update in enumerate(updates):
    quantity = update.get("new_quantity")
    rows.append(
      (
        "order_book_levels",
        {
          "provider": record.get("provider"),
          "symbol": record.get("symbol"),
          "event_time": update.get("event_time") or record.get("event_time"),
          "received_at": record.get("received_at"),
          "source_channel": record.get("source_channel"),
          "record_type": record.get("record_type"),
          "update_kind": "delta",
          "level_side": update.get("side"),
          "level_index": index,
          "price": update.get("price_level"),
          "quantity": quantity,
          "update_action": "remove" if _is_zero(quantity) else "set",
          "sequence_num": record.get("sequence_num"),
          "update_type": record.get("update_type"),
          "last_update_id": "",
        },
      )
    )
  return rows


def _binance_level2_rows(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
  payload = record.get("payload", {})
  bids = payload.get("bids", []) if isinstance(payload, dict) else []
  asks = payload.get("asks", []) if isinstance(payload, dict) else []
  last_update_id = record.get("last_update_id") or payload.get("lastUpdateId")
  rows: list[tuple[str, dict[str, Any]]] = [
    (
      "order_book_events",
      {
        "provider": record.get("provider"),
        "symbol": record.get("symbol"),
        "event_time": record.get("event_time"),
        "received_at": record.get("received_at"),
        "source_channel": record.get("source_channel"),
        "record_type": record.get("record_type"),
        "update_kind": "snapshot",
        "sequence_num": record.get("sequence_num"),
        "update_type": record.get("update_type"),
        "last_update_id": last_update_id,
        "update_count": len(bids) + len(asks),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
      },
    )
  ]

  for side, levels in (("BUY", bids), ("SELL", asks)):
    for index, level in enumerate(levels):
      price, quantity = level
      rows.append(
        (
          "order_book_levels",
          {
            "provider": record.get("provider"),
            "symbol": record.get("symbol"),
            "event_time": record.get("event_time"),
            "received_at": record.get("received_at"),
            "source_channel": record.get("source_channel"),
            "record_type": record.get("record_type"),
            "update_kind": "snapshot",
            "level_side": side,
            "level_index": index,
            "price": price,
            "quantity": quantity,
            "update_action": "remove" if _is_zero(quantity) else "set",
            "sequence_num": record.get("sequence_num"),
            "update_type": record.get("update_type"),
            "last_update_id": last_update_id,
          },
        )
      )

  return rows


def _decimal_product(left: Any, right: Any) -> str:
  left_decimal = _to_decimal(left)
  right_decimal = _to_decimal(right)
  if left_decimal is None or right_decimal is None:
    return ""
  return format(left_decimal * right_decimal, "f")


def _decimal_difference(left: Any, right: Any) -> str:
  left_decimal = _to_decimal(left)
  right_decimal = _to_decimal(right)
  if left_decimal is None or right_decimal is None:
    return ""
  return format(left_decimal - right_decimal, "f")


def _decimal_midpoint(left: Any, right: Any) -> str:
  left_decimal = _to_decimal(left)
  right_decimal = _to_decimal(right)
  if left_decimal is None or right_decimal is None:
    return ""
  return format((left_decimal + right_decimal) / Decimal("2"), "f")


def _to_decimal(value: Any) -> Decimal | None:
  if value in (None, ""):
    return None
  try:
    return Decimal(str(value))
  except (InvalidOperation, ValueError):
    return None


def _payload_value(payload: Any, key: str) -> Any:
  if isinstance(payload, dict):
    return payload.get(key)
  return None


def _is_zero(value: Any) -> bool:
  decimal_value = _to_decimal(value)
  return decimal_value == 0 if decimal_value is not None else False


def _csv_safe(value: Any) -> Any:
  if value is None:
    return ""
  if isinstance(value, bool):
    return "true" if value else "false"
  return value

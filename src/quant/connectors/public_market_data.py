from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

Record = dict[str, Any]


@dataclass(frozen=True, slots=True)
class StreamRequest:
  products: tuple[str, ...]
  channels: tuple[str, ...]
  candle_interval: str = "1m"

  def __post_init__(self) -> None:
    if not self.products:
      raise ValueError("At least one product or symbol is required.")
    if not self.channels:
      raise ValueError("At least one channel is required.")


class PublicMarketDataFeed(ABC):
  name: str
  supported_channels: frozenset[str]

  def validate_request(self, request: StreamRequest) -> None:
    unsupported = sorted(set(request.channels) - self.supported_channels)
    if unsupported:
      supported = ", ".join(sorted(self.supported_channels))
      raise ValueError(
        f"{self.name} does not support channels: {', '.join(unsupported)}. "
        f"Supported channels: {supported}."
      )

  @abstractmethod
  async def iter_records(self, request: StreamRequest) -> AsyncIterator[Record]:
    raise NotImplementedError


def utc_now_iso() -> str:
  return _to_utc_iso(datetime.now(timezone.utc))


def iso_from_unix_millis(value: int | str | float) -> str:
  timestamp = float(value) / 1000.0
  return _to_utc_iso(datetime.fromtimestamp(timestamp, tz=timezone.utc))


def iso_from_unix_seconds(value: int | str | float) -> str:
  timestamp = float(value)
  return _to_utc_iso(datetime.fromtimestamp(timestamp, tz=timezone.utc))


def build_record(
  *,
  provider: str,
  channel: str,
  source_channel: str,
  symbol: str,
  record_type: str,
  received_at: str,
  event_time: str | None,
  payload: dict[str, Any],
  extra: dict[str, Any] | None = None,
) -> Record:
  record: Record = {
    "provider": provider,
    "channel": channel,
    "source_channel": source_channel,
    "symbol": symbol,
    "record_type": record_type,
    "received_at": received_at,
    "event_time": event_time or received_at,
    "payload": payload,
  }
  if extra:
    record.update(extra)
  return record


def _to_utc_iso(value: datetime) -> str:
  return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
    "+00:00",
    "Z",
  )

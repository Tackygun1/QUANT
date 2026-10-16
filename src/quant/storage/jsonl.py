from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


@dataclass
class _OpenFile:
  path: Path
  handle: TextIO


class PartitionedJsonlWriter:
  def __init__(self, root: Path) -> None:
    self.root = root
    self.root.mkdir(parents=True, exist_ok=True)
    self._open_files: dict[tuple[str, str, str], _OpenFile] = {}

  def write(self, record: dict[str, object]) -> Path:
    provider = str(record["provider"])
    channel = str(record["channel"])
    symbol = _path_safe(str(record["symbol"]))
    record_path = self._record_path(
      provider=provider,
      channel=channel,
      symbol=symbol,
      event_time=str(record["event_time"]),
    )

    stream_key = (provider, channel, symbol)
    state = self._open_files.get(stream_key)
    if state is None or state.path != record_path:
      if state is not None:
        state.handle.close()
      record_path.parent.mkdir(parents=True, exist_ok=True)
      handle = record_path.open("a", encoding="utf-8", buffering=1)
      self._open_files[stream_key] = _OpenFile(path=record_path, handle=handle)

    json.dump(record, self._open_files[stream_key].handle, separators=(",", ":"))
    self._open_files[stream_key].handle.write("\n")
    return record_path

  def close(self) -> None:
    for state in self._open_files.values():
      state.handle.close()
    self._open_files.clear()

  def _record_path(
    self,
    *,
    provider: str,
    channel: str,
    symbol: str,
    event_time: str,
  ) -> Path:
    event_dt = _parse_iso8601(event_time)
    return (
      self.root
      / provider
      / channel
      / symbol
      / event_dt.strftime("%Y")
      / event_dt.strftime("%m")
      / event_dt.strftime("%d")
      / f"{event_dt.strftime('%H')}.jsonl"
    )


def _parse_iso8601(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _path_safe(value: str) -> str:
  return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)

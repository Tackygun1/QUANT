from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_json_config(path: Path | None) -> dict[str, Any]:
  if path is None:
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def config_value(
  cli_value: Any,
  payload: Mapping[str, Any],
  key: str,
  default: Any = None,
) -> Any:
  if cli_value is not None:
    return cli_value
  return payload.get(key, default)


def optional_path(value: Any) -> Path | None:
  if value in (None, ""):
    return None
  return Path(value)


def tuple_value(value: Any) -> tuple[Any, ...]:
  if value is None:
    return ()
  return tuple(value)

from __future__ import annotations

import re

_TARGET_HORIZON_PATTERN = re.compile(r"_(\d+)_bar(?:_|$)")


def infer_target_horizon_rows(target_columns: str | list[str] | tuple[str, ...]) -> int:
  columns = (target_columns,) if isinstance(target_columns, str) else tuple(target_columns)
  horizons: list[int] = []
  for column in columns:
    match = _TARGET_HORIZON_PATTERN.search(column)
    if match is not None:
      horizons.append(int(match.group(1)))
  return max(horizons, default=0)


def effective_purge_rows(configured_purge_rows: int | None, target_columns: str | list[str] | tuple[str, ...]) -> int:
  inferred = infer_target_horizon_rows(target_columns)
  if configured_purge_rows is None:
    return inferred
  if configured_purge_rows < 0:
    raise ValueError("purge_rows cannot be negative.")
  return max(configured_purge_rows, inferred)

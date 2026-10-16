from __future__ import annotations

import csv
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LabelBuildConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/features")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  horizons: tuple[int, ...] = (1, 6, 24, 96)
  direction_dead_zone_bps: Decimal = Decimal("5")
  trade_cost_bps: Decimal = Decimal("10")
  realized_volatility_window: int = 5
  trend_slope_dead_zone_bps_per_bar: Decimal = Decimal("0.5")
  trend_slope_tstat_threshold: float = 1.0
  trend_efficiency_threshold: float = 0.35
  triple_barrier_profit_bps: Decimal = Decimal("20")
  triple_barrier_stop_bps: Decimal = Decimal("20")
  limit_series: int | None = None


@dataclass(frozen=True, slots=True)
class FeatureSeriesKey:
  provider: str
  symbol: str


@dataclass(slots=True)
class LabelBuildSummary:
  series_processed: int = 0
  output_files_written: int = 0
  label_rows_written: int = 0


@dataclass(frozen=True, slots=True)
class FeaturePoint:
  provider: str
  symbol: str
  feature_date: str
  candle_start: str
  candle_end: str
  event_time: str
  high: Decimal | None
  low: Decimal | None
  close: Decimal | None
  is_closed: bool | None

  @property
  def candle_dt(self) -> datetime:
    return _parse_dt(self.candle_start)


def build_market_labels(config: LabelBuildConfig) -> LabelBuildSummary:
  summary = LabelBuildSummary()
  series_keys = discover_feature_series(config)

  for index, series_key in enumerate(series_keys, start=1):
    if config.limit_series is not None and index > config.limit_series:
      break

    rows_by_date = build_label_rows_for_series(config.input_root, series_key, config)
    if not rows_by_date:
      continue

    summary.series_processed += 1
    for feature_date, rows in rows_by_date.items():
      output_path = label_output_path(config.output_root, series_key, feature_date)
      write_label_rows(output_path, rows, config)
      LOGGER.info("Wrote %s label rows to %s", len(rows), output_path)
      summary.output_files_written += 1
      summary.label_rows_written += len(rows)

  return summary


def discover_feature_series(config: LabelBuildConfig) -> list[FeatureSeriesKey]:
  features_root = config.input_root / "market_features"
  if not features_root.exists():
    LOGGER.warning("No feature root found at %s", features_root)
    return []

  provider_filter = set(config.providers)
  symbol_filter = set(config.symbols)
  seen: dict[tuple[str, str], FeatureSeriesKey] = {}

  for path in sorted(features_root.rglob("*.csv")):
    parts = path.relative_to(features_root).parts
    if len(parts) != 5:
      LOGGER.debug("Skipping unexpected feature path: %s", path)
      continue

    provider, symbol, _year, _month, _filename = parts
    if provider_filter and provider not in provider_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue

    seen[(provider, symbol)] = FeatureSeriesKey(provider=provider, symbol=symbol)

  return list(seen.values())


def build_label_rows_for_series(
  input_root: Path,
  series_key: FeatureSeriesKey,
  config: LabelBuildConfig,
) -> dict[str, list[dict[str, str]]]:
  points = load_feature_points(input_root, series_key)
  if not points:
    return {}

  rows_by_date: dict[str, list[dict[str, str]]] = {}
  one_bar_log_returns = _compute_one_bar_log_returns(points)

  for index, point in enumerate(points):
    row = {
      "provider": point.provider,
      "symbol": point.symbol,
      "feature_date": point.feature_date,
      "candle_start": point.candle_start,
      "candle_end": point.candle_end,
      "event_time": point.event_time,
      "close": _format_decimal(point.close),
      "is_closed": _format_bool(point.is_closed),
    }
    row.update(_build_horizon_labels(points, index, config))
    row.update(_build_realized_volatility_labels(points, index, one_bar_log_returns, config))
    row.update(_build_future_trend_labels(points, index, config))
    row.update(_build_triple_barrier_labels(points, index, config))

    rows_by_date.setdefault(point.feature_date, []).append(row)

  return rows_by_date


def load_feature_points(
  input_root: Path,
  series_key: FeatureSeriesKey,
) -> list[FeaturePoint]:
  series_root = input_root / "market_features" / series_key.provider / series_key.symbol
  if not series_root.exists():
    return []

  points: list[FeaturePoint] = []
  for path in sorted(series_root.rglob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        points.append(
          FeaturePoint(
            provider=row["provider"],
            symbol=row["symbol"],
            feature_date=row["feature_date"],
            candle_start=row["candle_start"],
            candle_end=row["candle_end"],
            event_time=row["event_time"],
            high=_decimal_or_none(row.get("high")),
            low=_decimal_or_none(row.get("low")),
            close=_decimal_or_none(row.get("close")),
            is_closed=_parse_bool_optional(row.get("is_closed")),
          )
        )

  points.sort(key=lambda point: point.candle_dt)
  return points


def label_output_path(output_root: Path, series_key: FeatureSeriesKey, feature_date: str) -> Path:
  year, month, day = feature_date.split("-")
  return (
    output_root
    / "market_labels"
    / series_key.provider
    / series_key.symbol
    / year
    / month
    / f"{day}.csv"
  )


def write_label_rows(path: Path, rows: list[dict[str, str]], config: LabelBuildConfig) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = build_label_fieldnames(config)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def build_label_fieldnames(config: LabelBuildConfig) -> list[str]:
  fieldnames = [
    "provider",
    "symbol",
    "feature_date",
    "candle_start",
    "candle_end",
    "event_time",
    "close",
    "is_closed",
  ]
  for horizon in config.horizons:
    fieldnames.extend(
      [
        f"forward_return_{horizon}_bar",
        f"forward_return_{horizon}_bar_bps",
        f"forward_log_return_{horizon}_bar",
        f"direction_label_{horizon}_bar",
        f"tradeable_label_{horizon}_bar",
        f"future_trend_slope_{horizon}_bar",
        f"future_trend_slope_bps_per_bar_{horizon}_bar",
        f"future_trend_slope_tstat_{horizon}_bar",
        f"future_trend_linear_r2_{horizon}_bar",
        f"future_trend_efficiency_{horizon}_bar",
        f"future_trend_state_label_{horizon}_bar",
        f"future_trend_persistence_label_{horizon}_bar",
        f"future_trend_break_label_{horizon}_bar",
        f"triple_barrier_label_{horizon}_bar",
        f"triple_barrier_time_{horizon}_bar",
        f"triple_barrier_return_{horizon}_bar",
      ]
    )
  fieldnames.extend(
    [
      f"realized_volatility_{config.realized_volatility_window}_bar",
      f"future_abs_return_mean_{config.realized_volatility_window}_bar",
    ]
  )
  return fieldnames


def _build_horizon_labels(
  points: list[FeaturePoint],
  index: int,
  config: LabelBuildConfig,
) -> dict[str, str]:
  row: dict[str, str] = {}
  current = points[index]

  for horizon in config.horizons:
    target_index = index + horizon
    if target_index >= len(points):
      _set_blank_horizon_labels(row, horizon)
      continue

    target = points[target_index]
    if current.close in (None, Decimal("0")) or target.close is None:
      _set_blank_horizon_labels(row, horizon)
      continue

    if current.is_closed is not True or target.is_closed is not True:
      _set_blank_horizon_labels(row, horizon)
      continue

    forward_return = (target.close / current.close) - Decimal("1")
    forward_return_bps = forward_return * Decimal("10000")
    forward_log_return = math.log(float(target.close / current.close))
    direction = _direction_label(forward_return_bps, config.direction_dead_zone_bps)
    tradeable = "1" if abs(forward_return_bps) > config.trade_cost_bps else "0"

    row[f"forward_return_{horizon}_bar"] = _format_decimal(forward_return)
    row[f"forward_return_{horizon}_bar_bps"] = _format_decimal(forward_return_bps)
    row[f"forward_log_return_{horizon}_bar"] = _format_float(forward_log_return)
    row[f"direction_label_{horizon}_bar"] = direction
    row[f"tradeable_label_{horizon}_bar"] = tradeable

  return row


def _build_realized_volatility_labels(
  points: list[FeaturePoint],
  index: int,
  one_bar_log_returns: list[float | None],
  config: LabelBuildConfig,
) -> dict[str, str]:
  window = config.realized_volatility_window
  returns_slice = one_bar_log_returns[index : index + window]

  if len(returns_slice) < window or any(value is None for value in returns_slice):
    return {
      f"realized_volatility_{window}_bar": "",
      f"future_abs_return_mean_{window}_bar": "",
    }

  future_points = points[index + 1 : index + 1 + window]
  current = points[index]
  if current.close in (None, Decimal("0")) or current.is_closed is not True:
    return {
      f"realized_volatility_{window}_bar": "",
      f"future_abs_return_mean_{window}_bar": "",
    }

  future_abs_returns: list[Decimal] = []
  for point in future_points:
    if point.close is None or point.is_closed is not True:
      return {
        f"realized_volatility_{window}_bar": "",
        f"future_abs_return_mean_{window}_bar": "",
      }
    future_abs_returns.append(abs((point.close / current.close) - Decimal("1")))

  return {
    f"realized_volatility_{window}_bar": _format_float(statistics.pstdev(returns_slice)),
    f"future_abs_return_mean_{window}_bar": _format_decimal(
      sum(future_abs_returns, Decimal("0")) / Decimal(len(future_abs_returns))
    ),
  }


def _build_future_trend_labels(
  points: list[FeaturePoint],
  index: int,
  config: LabelBuildConfig,
) -> dict[str, str]:
  row: dict[str, str] = {}
  for horizon in config.horizons:
    if horizon < 3 or index + horizon >= len(points):
      _set_blank_future_trend_labels(row, horizon)
      continue

    trend_points = points[index : index + horizon + 1]
    stats = _linear_log_trend_stats([point.close for point in trend_points])
    if stats is None or any(point.is_closed is not True for point in trend_points):
      _set_blank_future_trend_labels(row, horizon)
      continue

    slope_bps_per_bar = Decimal(str(stats["slope"])) * Decimal("10000")
    state = _future_trend_state(
      slope_bps_per_bar=slope_bps_per_bar,
      slope_tstat=stats["slope_tstat"],
      config=config,
    )
    half_horizon = max(3, horizon // 2)
    half_stats = _linear_log_trend_stats([point.close for point in points[index : index + half_horizon + 1]])
    half_state = (
      _future_trend_state(
        slope_bps_per_bar=Decimal(str(half_stats["slope"])) * Decimal("10000"),
        slope_tstat=half_stats["slope_tstat"],
        config=config,
      )
      if half_stats is not None
      else "0"
    )
    persistence = (
      "1"
      if state != "0"
      and half_state == state
      and stats["efficiency"] >= config.trend_efficiency_threshold
      else "0"
    )
    trend_break = "1" if half_state != "0" and state != half_state else "0"

    row[f"future_trend_slope_{horizon}_bar"] = _format_float(stats["slope"])
    row[f"future_trend_slope_bps_per_bar_{horizon}_bar"] = _format_decimal(slope_bps_per_bar)
    row[f"future_trend_slope_tstat_{horizon}_bar"] = _format_float(stats["slope_tstat"])
    row[f"future_trend_linear_r2_{horizon}_bar"] = _format_float(stats["r2"])
    row[f"future_trend_efficiency_{horizon}_bar"] = _format_float(stats["efficiency"])
    row[f"future_trend_state_label_{horizon}_bar"] = state
    row[f"future_trend_persistence_label_{horizon}_bar"] = persistence
    row[f"future_trend_break_label_{horizon}_bar"] = trend_break
  return row


def _build_triple_barrier_labels(
  points: list[FeaturePoint],
  index: int,
  config: LabelBuildConfig,
) -> dict[str, str]:
  row: dict[str, str] = {}
  current = points[index]
  for horizon in config.horizons:
    if index + horizon >= len(points) or current.close in (None, Decimal("0")) or current.is_closed is not True:
      _set_blank_triple_barrier_labels(row, horizon)
      continue

    profit_rate = Decimal(str(config.triple_barrier_profit_bps)) / Decimal("10000")
    stop_rate = Decimal(str(config.triple_barrier_stop_bps)) / Decimal("10000")
    upper_barrier = current.close * (Decimal("1") + profit_rate)
    lower_barrier = current.close * (Decimal("1") - stop_rate)
    label = "0"
    hit_time = horizon
    exit_close = points[index + horizon].close

    if exit_close is None:
      _set_blank_triple_barrier_labels(row, horizon)
      continue

    for offset, point in enumerate(points[index + 1 : index + horizon + 1], start=1):
      if point.is_closed is not True:
        _set_blank_triple_barrier_labels(row, horizon)
        break
      high = point.high if point.high is not None else point.close
      low = point.low if point.low is not None else point.close
      if high is None or low is None or point.close is None:
        _set_blank_triple_barrier_labels(row, horizon)
        break
      hit_upper = high >= upper_barrier
      hit_lower = low <= lower_barrier
      if hit_upper and hit_lower:
        label = "0"
        hit_time = offset
        exit_close = point.close
        break
      if hit_upper:
        label = "1"
        hit_time = offset
        exit_close = upper_barrier
        break
      if hit_lower:
        label = "-1"
        hit_time = offset
        exit_close = lower_barrier
        break
    else:
      row[f"triple_barrier_label_{horizon}_bar"] = label
      row[f"triple_barrier_time_{horizon}_bar"] = str(hit_time)
      row[f"triple_barrier_return_{horizon}_bar"] = _format_decimal((exit_close / current.close) - Decimal("1"))
      continue

    if row.get(f"triple_barrier_label_{horizon}_bar", None) == "":
      continue
    row[f"triple_barrier_label_{horizon}_bar"] = label
    row[f"triple_barrier_time_{horizon}_bar"] = str(hit_time)
    row[f"triple_barrier_return_{horizon}_bar"] = _format_decimal((exit_close / current.close) - Decimal("1"))
  return row


def _compute_one_bar_log_returns(points: list[FeaturePoint]) -> list[float | None]:
  returns: list[float | None] = []
  for current, nxt in zip(points, points[1:]):
    if (
      current.close in (None, Decimal("0"))
      or nxt.close in (None, Decimal("0"))
      or current.is_closed is not True
      or nxt.is_closed is not True
    ):
      returns.append(None)
      continue
    returns.append(math.log(float(nxt.close / current.close)))
  return returns


def _set_blank_horizon_labels(row: dict[str, str], horizon: int) -> None:
  row[f"forward_return_{horizon}_bar"] = ""
  row[f"forward_return_{horizon}_bar_bps"] = ""
  row[f"forward_log_return_{horizon}_bar"] = ""
  row[f"direction_label_{horizon}_bar"] = ""
  row[f"tradeable_label_{horizon}_bar"] = ""


def _set_blank_future_trend_labels(row: dict[str, str], horizon: int) -> None:
  row[f"future_trend_slope_{horizon}_bar"] = ""
  row[f"future_trend_slope_bps_per_bar_{horizon}_bar"] = ""
  row[f"future_trend_slope_tstat_{horizon}_bar"] = ""
  row[f"future_trend_linear_r2_{horizon}_bar"] = ""
  row[f"future_trend_efficiency_{horizon}_bar"] = ""
  row[f"future_trend_state_label_{horizon}_bar"] = ""
  row[f"future_trend_persistence_label_{horizon}_bar"] = ""
  row[f"future_trend_break_label_{horizon}_bar"] = ""


def _set_blank_triple_barrier_labels(row: dict[str, str], horizon: int) -> None:
  row[f"triple_barrier_label_{horizon}_bar"] = ""
  row[f"triple_barrier_time_{horizon}_bar"] = ""
  row[f"triple_barrier_return_{horizon}_bar"] = ""


def _future_trend_state(
  *,
  slope_bps_per_bar: Decimal,
  slope_tstat: float,
  config: LabelBuildConfig,
) -> str:
  dead_zone = Decimal(str(config.trend_slope_dead_zone_bps_per_bar))
  if abs(slope_tstat) < config.trend_slope_tstat_threshold:
    return "0"
  if slope_bps_per_bar > dead_zone:
    return "1"
  if slope_bps_per_bar < -dead_zone:
    return "-1"
  return "0"


def _linear_log_trend_stats(values: list[Decimal | None]) -> dict[str, float] | None:
  if len(values) < 3 or any(value in (None, Decimal("0")) for value in values):
    return None
  cleaned = [value for value in values if value is not None]
  if any(value <= 0 for value in cleaned):
    return None
  log_values = [math.log(float(value)) for value in cleaned]
  x_values = list(range(len(log_values)))
  x_mean = statistics.fmean(x_values)
  y_mean = statistics.fmean(log_values)
  centered_x = [value - x_mean for value in x_values]
  centered_y = [value - y_mean for value in log_values]
  x_variance_sum = sum(value * value for value in centered_x)
  if x_variance_sum <= 0:
    return None
  slope = sum(x * y for x, y in zip(centered_x, centered_y)) / x_variance_sum
  intercept = y_mean - slope * x_mean
  fitted = [intercept + slope * x for x in x_values]
  residuals = [actual - expected for actual, expected in zip(log_values, fitted)]
  ss_residual = sum(value * value for value in residuals)
  ss_total = sum(value * value for value in centered_y)
  r2 = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
  residual_std = math.sqrt(ss_residual / max(1, len(log_values) - 2))
  slope_standard_error = residual_std / math.sqrt(x_variance_sum) if residual_std > 0 else 0.0
  slope_tstat = slope / slope_standard_error if slope_standard_error > 0 else 0.0
  path_length = sum(abs(current - previous) for previous, current in zip(log_values, log_values[1:]))
  net_change = log_values[-1] - log_values[0]
  efficiency = min(1.0, abs(net_change) / path_length) if path_length > 0 else 0.0
  return {
    "slope": slope,
    "slope_tstat": slope_tstat,
    "r2": max(0.0, min(1.0, r2)),
    "efficiency": efficiency,
  }


def _direction_label(forward_return_bps: Decimal, dead_zone_bps: Decimal) -> str:
  if forward_return_bps > dead_zone_bps:
    return "1"
  if forward_return_bps < -dead_zone_bps:
    return "-1"
  return "0"


def _decimal_or_none(value: Any) -> Decimal | None:
  if value in (None, ""):
    return None
  try:
    return Decimal(str(value))
  except (InvalidOperation, ValueError):
    return None


def _parse_bool_optional(value: str | None) -> bool | None:
  if value in (None, ""):
    return None
  lowered = value.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  return None


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_decimal(value: Decimal | None) -> str:
  if value is None:
    return ""
  normalized = format(value.normalize(), "f")
  if "." in normalized:
    normalized = normalized.rstrip("0").rstrip(".")
  return normalized or "0"


def _format_float(value: float | None) -> str:
  if value is None:
    return ""
  return format(value, ".12g")


def _format_bool(value: bool | None) -> str:
  if value is None:
    return ""
  return "true" if value else "false"

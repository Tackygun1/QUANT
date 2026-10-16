from __future__ import annotations

import csv
import logging
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant.features.regime import (
  OnlineRegimeFeatureModel,
  RegimeFeatureSnapshot,
  build_regime_observation,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CANDLE_DELTA = timedelta(minutes=1)
STRUCTURE_LOOKBACK_WINDOWS = (6, 24, 96)
TREND_LOOKBACK_WINDOWS = (6, 24, 96)
TREND_SHAPE_WINDOWS = (6, 24, 96)
TREND_VOLATILITY_WINDOWS = (20, 48)
TREND_VOLUME_WINDOWS = (20,)
ZONE_LOOKBACK_WINDOWS = tuple(window for window in STRUCTURE_LOOKBACK_WINDOWS if window >= 24)
ZONE_ALIGNMENT_WINDOWS = tuple(
  (ZONE_LOOKBACK_WINDOWS[index], ZONE_LOOKBACK_WINDOWS[index + 1])
  for index in range(len(ZONE_LOOKBACK_WINDOWS) - 1)
)
ZONE_PIVOT_HALFWINDOW = 2
ZONE_TOLERANCE_RANGE_FACTOR = Decimal("0.5")
ZONE_TOLERANCE_FLOOR_BPS = Decimal("15")
ZONE_SIDE_NAMES = ("supply", "demand")
ZONE_FIELD_NAMES = (
  "low",
  "high",
  "center",
  "width_pct",
  "distance",
  "age",
  "freshness",
  "touch_count",
  "rejection_count",
  "volume_share",
)
ZONE_FEATURE_FIELDS = tuple(
  f"structure_{side}_zone_{field}_{window}_bar"
  for window in ZONE_LOOKBACK_WINDOWS
  for side in ZONE_SIDE_NAMES
  for field in ZONE_FIELD_NAMES
) + tuple(
  f"structure_{side}_zone_alignment_{short_window}_{long_window}_bar"
  for short_window, long_window in ZONE_ALIGNMENT_WINDOWS
  for side in ZONE_SIDE_NAMES
)
REGIME_FIELDS = (
  "regime_hmm_p_bear",
  "regime_hmm_p_chop",
  "regime_hmm_p_bull",
  "regime_hmm_confidence",
  "regime_hmm_entropy",
  "regime_hmm_trend_score",
  "regime_hmm_state_index",
  "regime_hmm_transition_risk",
  "regime_hmm_expected_duration",
  "regime_bocpd_change_prob",
  "regime_bocpd_expected_run_length",
  "regime_bocpd_stability_score",
  "regime_bocpd_change_flag",
)

FEATURE_FIELDS = (
  "provider",
  "symbol",
  "feature_date",
  "interval",
  "candle_start",
  "candle_end",
  "window_seconds",
  "event_time",
  "source_channel",
  "is_closed",
  "open",
  "high",
  "low",
  "close",
  "volume",
  "quote_volume",
  "candle_trade_count",
  "candle_range_pct",
  "candle_body_pct",
  "close_return_1_bar",
  "log_return_1_bar",
  "close_return_5_bar",
  "rolling_return_std_5_bar",
  "trend_return_6_bar",
  "trend_return_24_bar",
  "trend_return_96_bar",
  "trend_ema_6",
  "trend_ema_24",
  "trend_ema_96",
  "trend_close_vs_ema_6_pct",
  "trend_close_vs_ema_24_pct",
  "trend_close_vs_ema_96_pct",
  "trend_ema_spread_6_24_pct",
  "trend_ema_spread_24_96_pct",
  "trend_log_slope_6_bar",
  "trend_log_slope_tstat_6_bar",
  "trend_linear_r2_6_bar",
  "trend_efficiency_6_bar",
  "trend_signed_efficiency_6_bar",
  "trend_chop_score_6_bar",
  "trend_max_drawdown_6_bar",
  "trend_max_runup_6_bar",
  "trend_log_slope_24_bar",
  "trend_log_slope_tstat_24_bar",
  "trend_linear_r2_24_bar",
  "trend_efficiency_24_bar",
  "trend_signed_efficiency_24_bar",
  "trend_chop_score_24_bar",
  "trend_max_drawdown_24_bar",
  "trend_max_runup_24_bar",
  "trend_log_slope_96_bar",
  "trend_log_slope_tstat_96_bar",
  "trend_linear_r2_96_bar",
  "trend_efficiency_96_bar",
  "trend_signed_efficiency_96_bar",
  "trend_chop_score_96_bar",
  "trend_max_drawdown_96_bar",
  "trend_max_runup_96_bar",
  "trend_slope_acceleration_6_24_bar",
  "trend_slope_acceleration_24_96_bar",
  "trend_volume_ratio_20_bar",
  "trend_volume_zscore_20_bar",
  "trend_realized_volatility_20_bar",
  "trend_realized_volatility_48_bar",
  "structure_session_vwap",
  "structure_close_vs_session_vwap_pct",
  "structure_prev_high_6_bar",
  "structure_prev_low_6_bar",
  "structure_range_position_6_bar",
  "structure_distance_to_high_6_bar",
  "structure_distance_to_low_6_bar",
  "structure_breakout_up_6_bar",
  "structure_breakout_down_6_bar",
  "structure_prev_high_24_bar",
  "structure_prev_low_24_bar",
  "structure_range_position_24_bar",
  "structure_distance_to_high_24_bar",
  "structure_distance_to_low_24_bar",
  "structure_breakout_up_24_bar",
  "structure_breakout_down_24_bar",
  "structure_prev_high_96_bar",
  "structure_prev_low_96_bar",
  "structure_range_position_96_bar",
  "structure_distance_to_high_96_bar",
  "structure_distance_to_low_96_bar",
  "structure_breakout_up_96_bar",
  "structure_breakout_down_96_bar",
  *ZONE_FEATURE_FIELDS,
  *REGIME_FIELDS,
  "trades_window_count",
  "trades_window_base_volume",
  "trades_window_quote_volume",
  "trades_window_vwap",
  "trades_window_avg_size",
  "trades_window_buy_count",
  "trades_window_sell_count",
  "trades_window_buy_base_volume",
  "trades_window_sell_base_volume",
  "trades_window_imbalance",
  "top_window_observation_count",
  "top_window_mid_mean",
  "top_window_mid_last",
  "top_window_spread_mean",
  "top_window_spread_mean_bps",
  "top_window_bid_size_mean",
  "top_window_ask_size_mean",
  "top_window_size_imbalance_mean",
)


@dataclass(frozen=True, slots=True)
class FeatureBuildConfig:
  input_root: Path = Path("data/staging")
  output_root: Path = Path("data/features")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  limit_datasets: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetKey:
  provider: str
  symbol: str
  year: str
  month: str
  day: str

  @property
  def feature_date(self) -> str:
    return f"{self.year}-{self.month}-{self.day}"


@dataclass(slots=True)
class FeatureBuildSummary:
  datasets_processed: int = 0
  output_files_written: int = 0
  feature_rows_written: int = 0


@dataclass(frozen=True, slots=True)
class CandleWindow:
  provider: str
  symbol: str
  feature_date: str
  interval: str
  source_channel: str
  event_time: str
  candle_start: str
  candle_end: str
  start_dt: datetime
  end_dt_exclusive: datetime
  is_closed: bool | None
  open: Decimal | None
  high: Decimal | None
  low: Decimal | None
  close: Decimal | None
  volume: Decimal | None
  quote_volume: Decimal | None
  candle_trade_count: int | None


@dataclass(frozen=True, slots=True)
class TradeEvent:
  event_dt: datetime
  price: Decimal | None
  size: Decimal | None
  quote_notional: Decimal | None
  aggressor_side: str | None


@dataclass(frozen=True, slots=True)
class TopEvent:
  event_dt: datetime
  bid: Decimal | None
  ask: Decimal | None
  bid_size: Decimal | None
  ask_size: Decimal | None
  mid: Decimal | None
  spread: Decimal | None


@dataclass(frozen=True, slots=True)
class StructureZone:
  lower: Decimal
  upper: Decimal
  center: Decimal
  width_abs: Decimal
  width_pct: Decimal | None
  touch_count: int
  rejection_count: int
  age_bars: int
  freshness: Decimal
  volume_share: Decimal | None


def build_market_features(config: FeatureBuildConfig) -> FeatureBuildSummary:
  summary = FeatureBuildSummary()

  for index, dataset in enumerate(discover_feature_datasets(config), start=1):
    if config.limit_datasets is not None and index > config.limit_datasets:
      break

    feature_rows = build_feature_rows_for_dataset(config.input_root, dataset)
    if not feature_rows:
      LOGGER.info("No feature rows produced for %s %s", dataset.provider, dataset.symbol)
      continue

    output_path = feature_output_path(config.output_root, dataset)
    write_feature_rows(output_path, feature_rows)
    LOGGER.info("Wrote %s feature rows to %s", len(feature_rows), output_path)

    summary.datasets_processed += 1
    summary.output_files_written += 1
    summary.feature_rows_written += len(feature_rows)

  return summary


def discover_feature_datasets(config: FeatureBuildConfig) -> list[DatasetKey]:
  candles_root = config.input_root / "candles"
  if not candles_root.exists():
    LOGGER.warning("No staged candles root found at %s", candles_root)
    return []

  provider_filter = set(config.providers)
  symbol_filter = set(config.symbols)
  datasets: dict[tuple[str, str, str, str, str], DatasetKey] = {}

  for path in sorted(candles_root.rglob("*.csv")):
    parts = path.relative_to(candles_root).parts
    if len(parts) != 6:
      LOGGER.debug("Skipping unexpected candles path: %s", path)
      continue

    provider, symbol, year, month, day, _filename = parts
    if provider_filter and provider not in provider_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue

    key = (provider, symbol, year, month, day)
    datasets[key] = DatasetKey(
      provider=provider,
      symbol=symbol,
      year=year,
      month=month,
      day=day,
    )

  return list(datasets.values())


def build_feature_rows_for_dataset(
  input_root: Path,
  dataset: DatasetKey,
) -> list[dict[str, str]]:
  candle_rows = load_staged_rows(input_root, "candles", dataset)
  trade_rows = load_staged_rows(input_root, "trades", dataset)
  top_rows = load_staged_rows(input_root, "order_book_top", dataset)

  candles = _build_candle_windows(candle_rows, dataset)
  trades = _build_trade_events(trade_rows)
  top_events = _build_top_events(top_rows)

  if not candles:
    return []

  feature_rows: list[dict[str, str]] = []
  trade_index = 0
  top_index = 0
  close_history: list[Decimal] = []
  high_history: list[Decimal] = []
  low_history: list[Decimal] = []
  volume_history: list[Decimal] = []
  bar_history: list[CandleWindow] = []
  return_history: list[float] = []
  session_base_volume = Decimal("0")
  session_quote_volume = Decimal("0")
  regime_model = OnlineRegimeFeatureModel()

  for candle in candles:
    trade_index = _advance_index(trades, trade_index, candle.start_dt)
    top_index = _advance_index(top_events, top_index, candle.start_dt)

    trade_end = _window_end_index(trades, trade_index, candle.end_dt_exclusive)
    top_end = _window_end_index(top_events, top_index, candle.end_dt_exclusive)

    trade_window = trades[trade_index:trade_end]
    top_window = top_events[top_index:top_end]

    row = _base_feature_row(candle)
    row.update(_compute_bar_features(candle, close_history, return_history))
    row.update(
      _compute_layered_context_features(
        candle,
        close_history=close_history,
        high_history=high_history,
        low_history=low_history,
        volume_history=volume_history,
        bar_history=bar_history,
        return_history=return_history,
        session_base_volume=session_base_volume,
        session_quote_volume=session_quote_volume,
      )
    )
    row.update(_compute_regime_features(row, regime_model))
    row.update(_aggregate_trade_window(trade_window))
    row.update(_aggregate_top_window(top_window))

    feature_rows.append(row)

    if candle.close is not None:
      close_history.append(candle.close)
      if len(close_history) >= 2:
        previous_close = close_history[-2]
        if previous_close != 0:
          return_history.append(float((candle.close / previous_close) - Decimal("1")))
    if candle.high is not None:
      high_history.append(candle.high)
    if candle.low is not None:
      low_history.append(candle.low)
    if candle.volume is not None:
      session_base_volume += candle.volume
      volume_history.append(candle.volume)
    bar_history.append(candle)
    quote_volume = candle.quote_volume
    if quote_volume is None and candle.close is not None and candle.volume is not None:
      quote_volume = candle.close * candle.volume
    if quote_volume is not None:
      session_quote_volume += quote_volume

    trade_index = trade_end
    top_index = top_end

  return feature_rows


def load_staged_rows(
  input_root: Path,
  table_name: str,
  dataset: DatasetKey,
) -> list[dict[str, str]]:
  table_root = input_root / table_name / dataset.provider / dataset.symbol / dataset.year / dataset.month / dataset.day
  if not table_root.exists():
    return []

  rows: list[dict[str, str]] = []
  for path in sorted(table_root.glob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      rows.extend(csv.DictReader(handle))
  return rows


def feature_output_path(output_root: Path, dataset: DatasetKey) -> Path:
  return (
    output_root
    / "market_features"
    / dataset.provider
    / dataset.symbol
    / dataset.year
    / dataset.month
    / f"{dataset.day}.csv"
  )


def write_feature_rows(path: Path, rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _build_candle_windows(rows: list[dict[str, str]], dataset: DatasetKey) -> list[CandleWindow]:
  chosen_rows: dict[str, dict[str, str]] = {}

  for row in rows:
    candle_start = row.get("candle_start") or row.get("event_time")
    if not candle_start:
      continue

    existing = chosen_rows.get(candle_start)
    if existing is None or _prefer_candle_row(existing, row):
      chosen_rows[candle_start] = row

  if not chosen_rows:
    return []

  sorted_rows = sorted(chosen_rows.values(), key=lambda row: _parse_dt(row.get("candle_start") or row["event_time"]))
  start_times = [_parse_dt(row.get("candle_start") or row["event_time"]) for row in sorted_rows]
  inferred_delta = _infer_candle_delta(sorted_rows, start_times)
  windows: list[CandleWindow] = []

  for index, row in enumerate(sorted_rows):
    start_dt = start_times[index]
    next_start = start_times[index + 1] if index + 1 < len(start_times) else None
    explicit_end = _parse_dt_optional(row.get("candle_end"))
    end_dt_exclusive = _resolve_window_end(start_dt, explicit_end, next_start, inferred_delta)

    windows.append(
      CandleWindow(
        provider=row.get("provider", dataset.provider),
        symbol=row.get("symbol", dataset.symbol),
        feature_date=dataset.feature_date,
        interval=row.get("interval", ""),
        source_channel=row.get("source_channel", ""),
        event_time=row.get("event_time", ""),
        candle_start=_format_dt(start_dt),
        candle_end=_format_dt(end_dt_exclusive),
        start_dt=start_dt,
        end_dt_exclusive=end_dt_exclusive,
        is_closed=_parse_bool_optional(row.get("is_closed")),
        open=_decimal_or_none(row.get("open")),
        high=_decimal_or_none(row.get("high")),
        low=_decimal_or_none(row.get("low")),
        close=_decimal_or_none(row.get("close")),
        volume=_decimal_or_none(row.get("volume")),
        quote_volume=_decimal_or_none(row.get("quote_volume")),
        candle_trade_count=_int_or_none(row.get("trade_count")),
      )
    )

  return windows


def _build_trade_events(rows: list[dict[str, str]]) -> list[TradeEvent]:
  events: list[TradeEvent] = []
  for row in rows:
    event_time = row.get("event_time")
    if not event_time:
      continue
    events.append(
      TradeEvent(
        event_dt=_parse_dt(event_time),
        price=_decimal_or_none(row.get("price")),
        size=_decimal_or_none(row.get("size")),
        quote_notional=_decimal_or_none(row.get("notional_quote")),
        aggressor_side=(row.get("aggressor_side") or "").upper() or None,
      )
    )
  events.sort(key=lambda event: event.event_dt)
  return events


def _build_top_events(rows: list[dict[str, str]]) -> list[TopEvent]:
  events: list[TopEvent] = []
  for row in rows:
    event_time = row.get("event_time")
    if not event_time:
      continue
    bid = _decimal_or_none(row.get("best_bid"))
    ask = _decimal_or_none(row.get("best_ask"))
    mid = _decimal_or_none(row.get("mid_price"))
    if mid is None and bid is not None and ask is not None:
      mid = (bid + ask) / Decimal("2")
    spread = _decimal_or_none(row.get("spread"))
    if spread is None and bid is not None and ask is not None:
      spread = ask - bid

    events.append(
      TopEvent(
        event_dt=_parse_dt(event_time),
        bid=bid,
        ask=ask,
        bid_size=_decimal_or_none(row.get("best_bid_size")),
        ask_size=_decimal_or_none(row.get("best_ask_size")),
        mid=mid,
        spread=spread,
      )
    )
  events.sort(key=lambda event: event.event_dt)
  return events


def _base_feature_row(candle: CandleWindow) -> dict[str, str]:
  return {
    "provider": candle.provider,
    "symbol": candle.symbol,
    "feature_date": candle.feature_date,
    "interval": candle.interval,
    "candle_start": candle.candle_start,
    "candle_end": candle.candle_end,
    "window_seconds": _format_int(int((candle.end_dt_exclusive - candle.start_dt).total_seconds())),
    "event_time": candle.event_time,
    "source_channel": candle.source_channel,
    "is_closed": _format_bool(candle.is_closed),
    "open": _format_decimal(candle.open),
    "high": _format_decimal(candle.high),
    "low": _format_decimal(candle.low),
    "close": _format_decimal(candle.close),
    "volume": _format_decimal(candle.volume),
    "quote_volume": _format_decimal(candle.quote_volume),
    "candle_trade_count": _format_int(candle.candle_trade_count),
  }


def _compute_bar_features(
  candle: CandleWindow,
  close_history: list[Decimal],
  return_history: list[float],
) -> dict[str, str]:
  open_price = candle.open
  high_price = candle.high
  low_price = candle.low
  close_price = candle.close

  row = {
    "candle_range_pct": _format_decimal(_ratio(_difference(high_price, low_price), open_price)),
    "candle_body_pct": _format_decimal(_ratio(_difference(close_price, open_price), open_price)),
    "close_return_1_bar": "",
    "log_return_1_bar": "",
    "close_return_5_bar": "",
    "rolling_return_std_5_bar": "",
  }

  if close_price is None or not close_history:
    return row

  previous_close = close_history[-1]
  one_bar_return = _ratio(_difference(close_price, previous_close), previous_close)
  if one_bar_return is not None:
    row["close_return_1_bar"] = _format_decimal(one_bar_return)
    if close_price > 0 and previous_close > 0:
      row["log_return_1_bar"] = _format_float(math.log(float(close_price / previous_close)))

  if len(close_history) >= 5:
    five_bar_close = close_history[-5]
    five_bar_return = _ratio(_difference(close_price, five_bar_close), five_bar_close)
    row["close_return_5_bar"] = _format_decimal(five_bar_return)

  if len(return_history) >= 4 and one_bar_return is not None:
    rolling_values = [*return_history[-4:], float(one_bar_return)]
    row["rolling_return_std_5_bar"] = _format_float(statistics.pstdev(rolling_values))

  return row


def _compute_layered_context_features(
  candle: CandleWindow,
  *,
  close_history: list[Decimal],
  high_history: list[Decimal],
  low_history: list[Decimal],
  volume_history: list[Decimal],
  bar_history: list[CandleWindow],
  return_history: list[float],
  session_base_volume: Decimal,
  session_quote_volume: Decimal,
) -> dict[str, str]:
  row = _blank_layered_context_row()
  close_price = candle.close
  if close_price is None:
    return row

  current_quote_volume = candle.quote_volume
  if current_quote_volume is None and candle.volume is not None:
    current_quote_volume = close_price * candle.volume
  session_vwap = _session_vwap(
    base_volume=session_base_volume,
    quote_volume=session_quote_volume,
    current_base_volume=candle.volume,
    current_quote_volume=current_quote_volume,
  )
  row["structure_session_vwap"] = _format_decimal(session_vwap)
  row["structure_close_vs_session_vwap_pct"] = _format_decimal(
    _ratio(_difference(close_price, session_vwap), session_vwap)
  )

  close_series = [*close_history, close_price]
  previous_close = close_history[-1] if close_history else None
  current_one_bar_return = _decimal_to_float(_ratio(_difference(close_price, previous_close), previous_close))

  ema_values: dict[int, Decimal | None] = {}
  for window in TREND_LOOKBACK_WINDOWS:
    if len(close_history) >= window:
      past_close = close_history[-window]
      row[f"trend_return_{window}_bar"] = _format_decimal(
        _ratio(_difference(close_price, past_close), past_close)
      )

    if len(close_series) >= window:
      ema_value = _ema_decimal(close_series, span=window)
      ema_values[window] = ema_value
      row[f"trend_ema_{window}"] = _format_decimal(ema_value)
      row[f"trend_close_vs_ema_{window}_pct"] = _format_decimal(
        _ratio(_difference(close_price, ema_value), ema_value)
      )

  row["trend_ema_spread_6_24_pct"] = _format_decimal(
    _ratio(_difference(ema_values.get(6), ema_values.get(24)), ema_values.get(24))
  )
  row["trend_ema_spread_24_96_pct"] = _format_decimal(
    _ratio(_difference(ema_values.get(24), ema_values.get(96)), ema_values.get(96))
  )

  slope_values: dict[int, float] = {}
  for window in TREND_SHAPE_WINDOWS:
    if len(close_series) < window:
      continue
    trend_stats = _linear_log_trend_stats(close_series[-window:])
    if trend_stats is None:
      continue
    slope_values[window] = trend_stats["slope"]
    row[f"trend_log_slope_{window}_bar"] = _format_float(trend_stats["slope"])
    row[f"trend_log_slope_tstat_{window}_bar"] = _format_float(trend_stats["slope_tstat"])
    row[f"trend_linear_r2_{window}_bar"] = _format_float(trend_stats["r2"])
    row[f"trend_efficiency_{window}_bar"] = _format_float(trend_stats["efficiency"])
    row[f"trend_signed_efficiency_{window}_bar"] = _format_float(trend_stats["signed_efficiency"])
    row[f"trend_chop_score_{window}_bar"] = _format_float(1.0 - trend_stats["efficiency"])
    row[f"trend_max_drawdown_{window}_bar"] = _format_float(trend_stats["max_drawdown"])
    row[f"trend_max_runup_{window}_bar"] = _format_float(trend_stats["max_runup"])

  if 6 in slope_values and 24 in slope_values:
    row["trend_slope_acceleration_6_24_bar"] = _format_float(slope_values[6] - slope_values[24])
  if 24 in slope_values and 96 in slope_values:
    row["trend_slope_acceleration_24_96_bar"] = _format_float(slope_values[24] - slope_values[96])

  for window in TREND_VOLUME_WINDOWS:
    if candle.volume is None or len(volume_history) < window - 1:
      continue
    volume_values = [float(value) for value in [*volume_history[-(window - 1):], candle.volume]]
    volume_mean = statistics.fmean(volume_values)
    if volume_mean > 0:
      row[f"trend_volume_ratio_{window}_bar"] = _format_float(float(candle.volume) / volume_mean)
    volume_std = statistics.pstdev(volume_values)
    if volume_std > 0:
      row[f"trend_volume_zscore_{window}_bar"] = _format_float((float(candle.volume) - volume_mean) / volume_std)

  for window in TREND_VOLATILITY_WINDOWS:
    if current_one_bar_return is None or len(return_history) < window - 1:
      continue
    rolling_values = [*return_history[-(window - 1):], current_one_bar_return]
    row[f"trend_realized_volatility_{window}_bar"] = _format_float(statistics.pstdev(rolling_values))

  for window in STRUCTURE_LOOKBACK_WINDOWS:
    if len(high_history) < window or len(low_history) < window:
      continue
    previous_high = max(high_history[-window:])
    previous_low = min(low_history[-window:])
    row[f"structure_prev_high_{window}_bar"] = _format_decimal(previous_high)
    row[f"structure_prev_low_{window}_bar"] = _format_decimal(previous_low)
    row[f"structure_distance_to_high_{window}_bar"] = _format_decimal(
      _ratio(_difference(close_price, previous_high), previous_high)
    )
    row[f"structure_distance_to_low_{window}_bar"] = _format_decimal(
      _ratio(_difference(close_price, previous_low), previous_low)
    )
    row[f"structure_range_position_{window}_bar"] = _format_decimal(
      _ratio(
        _difference(close_price, previous_low),
        _difference(previous_high, previous_low),
      )
    )
    row[f"structure_breakout_up_{window}_bar"] = _format_int(1 if close_price > previous_high else 0)
    row[f"structure_breakout_down_{window}_bar"] = _format_int(1 if close_price < previous_low else 0)

  zones: dict[tuple[int, str], StructureZone] = {}
  for window in ZONE_LOOKBACK_WINDOWS:
    if len(bar_history) < window:
      continue
    window_bars = bar_history[-window:]
    for side in ZONE_SIDE_NAMES:
      zone = _detect_structure_zone(
        window_bars,
        side=side,
        reference_price=close_price,
      )
      if zone is None:
        continue
      zones[(window, side)] = zone
      row[f"structure_{side}_zone_low_{window}_bar"] = _format_decimal(zone.lower)
      row[f"structure_{side}_zone_high_{window}_bar"] = _format_decimal(zone.upper)
      row[f"structure_{side}_zone_center_{window}_bar"] = _format_decimal(zone.center)
      row[f"structure_{side}_zone_width_pct_{window}_bar"] = _format_decimal(zone.width_pct)
      row[f"structure_{side}_zone_distance_{window}_bar"] = _format_decimal(
        _ratio(_difference(close_price, zone.center), zone.center)
      )
      row[f"structure_{side}_zone_age_{window}_bar"] = _format_int(zone.age_bars)
      row[f"structure_{side}_zone_freshness_{window}_bar"] = _format_decimal(zone.freshness)
      row[f"structure_{side}_zone_touch_count_{window}_bar"] = _format_int(zone.touch_count)
      row[f"structure_{side}_zone_rejection_count_{window}_bar"] = _format_int(zone.rejection_count)
      row[f"structure_{side}_zone_volume_share_{window}_bar"] = _format_decimal(zone.volume_share)

  for short_window, long_window in ZONE_ALIGNMENT_WINDOWS:
    for side in ZONE_SIDE_NAMES:
      row[f"structure_{side}_zone_alignment_{short_window}_{long_window}_bar"] = _format_decimal(
        _zone_alignment_score(
          zones.get((short_window, side)),
          zones.get((long_window, side)),
        )
      )

  return row


def _blank_layered_context_row() -> dict[str, str]:
  row = {
    "structure_session_vwap": "",
    "structure_close_vs_session_vwap_pct": "",
    "trend_ema_spread_6_24_pct": "",
    "trend_ema_spread_24_96_pct": "",
    "trend_slope_acceleration_6_24_bar": "",
    "trend_slope_acceleration_24_96_bar": "",
  }
  for window in TREND_LOOKBACK_WINDOWS:
    row[f"trend_return_{window}_bar"] = ""
    row[f"trend_ema_{window}"] = ""
    row[f"trend_close_vs_ema_{window}_pct"] = ""
  for window in TREND_SHAPE_WINDOWS:
    row[f"trend_log_slope_{window}_bar"] = ""
    row[f"trend_log_slope_tstat_{window}_bar"] = ""
    row[f"trend_linear_r2_{window}_bar"] = ""
    row[f"trend_efficiency_{window}_bar"] = ""
    row[f"trend_signed_efficiency_{window}_bar"] = ""
    row[f"trend_chop_score_{window}_bar"] = ""
    row[f"trend_max_drawdown_{window}_bar"] = ""
    row[f"trend_max_runup_{window}_bar"] = ""
  for window in TREND_VOLUME_WINDOWS:
    row[f"trend_volume_ratio_{window}_bar"] = ""
    row[f"trend_volume_zscore_{window}_bar"] = ""
  for window in TREND_VOLATILITY_WINDOWS:
    row[f"trend_realized_volatility_{window}_bar"] = ""
  for window in STRUCTURE_LOOKBACK_WINDOWS:
    row[f"structure_prev_high_{window}_bar"] = ""
    row[f"structure_prev_low_{window}_bar"] = ""
    row[f"structure_range_position_{window}_bar"] = ""
    row[f"structure_distance_to_high_{window}_bar"] = ""
    row[f"structure_distance_to_low_{window}_bar"] = ""
    row[f"structure_breakout_up_{window}_bar"] = ""
    row[f"structure_breakout_down_{window}_bar"] = ""
  for window in ZONE_LOOKBACK_WINDOWS:
    for side in ZONE_SIDE_NAMES:
      row[f"structure_{side}_zone_low_{window}_bar"] = ""
      row[f"structure_{side}_zone_high_{window}_bar"] = ""
      row[f"structure_{side}_zone_center_{window}_bar"] = ""
      row[f"structure_{side}_zone_width_pct_{window}_bar"] = ""
      row[f"structure_{side}_zone_distance_{window}_bar"] = ""
      row[f"structure_{side}_zone_age_{window}_bar"] = ""
      row[f"structure_{side}_zone_freshness_{window}_bar"] = ""
      row[f"structure_{side}_zone_touch_count_{window}_bar"] = ""
      row[f"structure_{side}_zone_rejection_count_{window}_bar"] = ""
      row[f"structure_{side}_zone_volume_share_{window}_bar"] = ""
  for short_window, long_window in ZONE_ALIGNMENT_WINDOWS:
    for side in ZONE_SIDE_NAMES:
      row[f"structure_{side}_zone_alignment_{short_window}_{long_window}_bar"] = ""
  return row


def _detect_structure_zone(
  window_bars: list[CandleWindow],
  *,
  side: str,
  reference_price: Decimal,
) -> StructureZone | None:
  tolerance = _zone_tolerance(window_bars, reference_price)
  if tolerance <= 0:
    return None

  pivots = _swing_pivots(window_bars, side=side)
  if not pivots:
    fallback = _fallback_pivot(window_bars, side=side)
    if fallback is None:
      return None
    pivots = [fallback]

  pivot_clusters = _cluster_pivots_by_price(pivots, tolerance)
  candidates = [
    _build_structure_zone(window_bars, cluster=cluster, tolerance=tolerance, side=side)
    for cluster in pivot_clusters
  ]
  candidates = [candidate for candidate in candidates if candidate is not None]
  if not candidates:
    return None

  if side == "supply":
    preferred = [candidate for candidate in candidates if candidate.center >= reference_price]
    if preferred:
      return min(
        preferred,
        key=lambda candidate: (
          candidate.center - reference_price,
          -candidate.rejection_count,
          -candidate.touch_count,
          candidate.age_bars,
        ),
      )
  else:
    preferred = [candidate for candidate in candidates if candidate.center <= reference_price]
    if preferred:
      return min(
        preferred,
        key=lambda candidate: (
          reference_price - candidate.center,
          -candidate.rejection_count,
          -candidate.touch_count,
          candidate.age_bars,
        ),
      )

  return min(
    candidates,
    key=lambda candidate: (
      abs(candidate.center - reference_price),
      -candidate.rejection_count,
      -candidate.touch_count,
      candidate.age_bars,
    ),
  )


def _zone_tolerance(window_bars: list[CandleWindow], reference_price: Decimal) -> Decimal:
  ranges = [
    bar.high - bar.low
    for bar in window_bars
    if bar.high is not None and bar.low is not None and bar.high >= bar.low
  ]
  mean_range = _mean_decimal(ranges) or Decimal("0")
  range_tolerance = mean_range * ZONE_TOLERANCE_RANGE_FACTOR
  price_tolerance = abs(reference_price) * ZONE_TOLERANCE_FLOOR_BPS / Decimal("10000")
  tolerance = max(range_tolerance, price_tolerance)
  return tolerance if tolerance > 0 else Decimal("0")


def _swing_pivots(
  window_bars: list[CandleWindow],
  *,
  side: str,
) -> list[tuple[int, Decimal]]:
  values = [
    (bar.high if side == "supply" else bar.low) or bar.close
    for bar in window_bars
  ]
  pivots: list[tuple[int, Decimal]] = []
  for index, current in enumerate(values):
    if current is None:
      continue
    if index < ZONE_PIVOT_HALFWINDOW or index + ZONE_PIVOT_HALFWINDOW >= len(values):
      continue
    left = values[index - ZONE_PIVOT_HALFWINDOW:index]
    right = values[index + 1:index + 1 + ZONE_PIVOT_HALFWINDOW]
    if any(value is None for value in [*left, *right]):
      continue
    neighborhood = [*left, *right]
    if side == "supply":
      if all(current >= value for value in neighborhood) and any(current > value for value in neighborhood):
        pivots.append((index, current))
    else:
      if all(current <= value for value in neighborhood) and any(current < value for value in neighborhood):
        pivots.append((index, current))
  return pivots


def _fallback_pivot(
  window_bars: list[CandleWindow],
  *,
  side: str,
) -> tuple[int, Decimal] | None:
  values = [
    (
      index,
      (bar.high if side == "supply" else bar.low) or bar.close,
    )
    for index, bar in enumerate(window_bars)
  ]
  valid_values = [(index, value) for index, value in values if value is not None]
  if not valid_values:
    return None
  if side == "supply":
    return max(valid_values, key=lambda item: (item[1], item[0]))
  return min(valid_values, key=lambda item: (item[1], item[0]))


def _cluster_pivots_by_price(
  pivots: list[tuple[int, Decimal]],
  tolerance: Decimal,
) -> list[list[tuple[int, Decimal]]]:
  if not pivots:
    return []

  ordered = sorted(pivots, key=lambda item: item[1])
  clusters: list[list[tuple[int, Decimal]]] = [[ordered[0]]]
  for pivot in ordered[1:]:
    cluster = clusters[-1]
    if abs(pivot[1] - cluster[-1][1]) <= tolerance:
      cluster.append(pivot)
      continue
    clusters.append([pivot])
  return clusters


def _build_structure_zone(
  window_bars: list[CandleWindow],
  *,
  cluster: list[tuple[int, Decimal]],
  tolerance: Decimal,
  side: str,
) -> StructureZone | None:
  if not cluster:
    return None

  cluster_indices = [index for index, _price in cluster]
  cluster_prices = [price for _index, price in cluster]
  if not cluster_prices:
    return None

  half_tolerance = tolerance / Decimal("2")
  zone_lower = min(cluster_prices) - half_tolerance
  zone_upper = max(cluster_prices) + half_tolerance
  center = (zone_lower + zone_upper) / Decimal("2")
  width_abs = zone_upper - zone_lower
  width_pct = _ratio(width_abs, center)
  total_volume = Decimal("0")
  touched_volume = Decimal("0")
  touch_count = 0
  rejection_count = 0

  for bar in window_bars:
    if bar.volume is not None:
      total_volume += bar.volume
    if not _bar_touches_zone(bar, zone_lower, zone_upper):
      continue
    touch_count += 1
    if bar.volume is not None:
      touched_volume += bar.volume
    if _is_zone_rejection_bar(bar, side=side, zone_center=center):
      rejection_count += 1

  age_bars = len(window_bars) - 1 - max(cluster_indices)
  freshness = Decimal(len(window_bars) - age_bars) / Decimal(len(window_bars))
  return StructureZone(
    lower=zone_lower,
    upper=zone_upper,
    center=center,
    width_abs=width_abs,
    width_pct=width_pct,
    touch_count=touch_count,
    rejection_count=rejection_count,
    age_bars=age_bars,
    freshness=freshness,
    volume_share=_ratio(touched_volume, total_volume),
  )


def _bar_touches_zone(
  bar: CandleWindow,
  zone_lower: Decimal,
  zone_upper: Decimal,
) -> bool:
  bar_high = bar.high or bar.close
  bar_low = bar.low or bar.close
  if bar_high is None or bar_low is None:
    return False
  return bar_high >= zone_lower and bar_low <= zone_upper


def _is_zone_rejection_bar(
  bar: CandleWindow,
  *,
  side: str,
  zone_center: Decimal,
) -> bool:
  open_price = bar.open or bar.close
  close_price = bar.close
  if open_price is None or close_price is None:
    return False
  if side == "supply":
    return close_price <= zone_center and close_price < open_price
  return close_price >= zone_center and close_price > open_price


def _zone_alignment_score(
  primary_zone: StructureZone | None,
  secondary_zone: StructureZone | None,
) -> Decimal | None:
  if primary_zone is None or secondary_zone is None:
    return None
  if primary_zone.upper >= secondary_zone.lower and secondary_zone.upper >= primary_zone.lower:
    return Decimal("1")

  width_normalizer = max(
    primary_zone.width_abs,
    secondary_zone.width_abs,
    abs(primary_zone.center) * Decimal("0.0005"),
  )
  if width_normalizer <= 0:
    return Decimal("0")
  score = Decimal("1") - (abs(primary_zone.center - secondary_zone.center) / (width_normalizer * Decimal("2")))
  if score < 0:
    return Decimal("0")
  if score > 1:
    return Decimal("1")
  return score


def _linear_log_trend_stats(values: list[Decimal]) -> dict[str, float] | None:
  if len(values) < 3 or any(value <= 0 for value in values):
    return None

  log_values = [math.log(float(value)) for value in values]
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
  signed_efficiency = max(-1.0, min(1.0, net_change / path_length)) if path_length > 0 else 0.0

  return {
    "slope": slope,
    "slope_tstat": slope_tstat,
    "r2": max(0.0, min(1.0, r2)),
    "efficiency": efficiency,
    "signed_efficiency": signed_efficiency,
    "max_drawdown": _max_drawdown(values),
    "max_runup": _max_runup(values),
  }


def _max_drawdown(values: list[Decimal]) -> float:
  peak: Decimal | None = None
  max_drawdown = Decimal("0")
  for value in values:
    if value <= 0:
      continue
    peak = value if peak is None else max(peak, value)
    if peak > 0:
      max_drawdown = max(max_drawdown, (peak - value) / peak)
  return float(max_drawdown)


def _max_runup(values: list[Decimal]) -> float:
  trough: Decimal | None = None
  max_runup = Decimal("0")
  for value in values:
    if value <= 0:
      continue
    trough = value if trough is None else min(trough, value)
    if trough > 0:
      max_runup = max(max_runup, (value - trough) / trough)
  return float(max_runup)


def _compute_regime_features(
  context_row: dict[str, str],
  regime_model: OnlineRegimeFeatureModel,
) -> dict[str, str]:
  observation = build_regime_observation(context_row)
  snapshot = regime_model.update(observation)
  return _format_regime_snapshot(snapshot)


def _format_regime_snapshot(snapshot: RegimeFeatureSnapshot) -> dict[str, str]:
  return {
    "regime_hmm_p_bear": _format_float(snapshot.hmm_p_bear),
    "regime_hmm_p_chop": _format_float(snapshot.hmm_p_chop),
    "regime_hmm_p_bull": _format_float(snapshot.hmm_p_bull),
    "regime_hmm_confidence": _format_float(snapshot.hmm_confidence),
    "regime_hmm_entropy": _format_float(snapshot.hmm_entropy),
    "regime_hmm_trend_score": _format_float(snapshot.hmm_trend_score),
    "regime_hmm_state_index": _format_int(snapshot.hmm_state_index),
    "regime_hmm_transition_risk": _format_float(snapshot.hmm_transition_risk),
    "regime_hmm_expected_duration": _format_float(snapshot.hmm_expected_duration),
    "regime_bocpd_change_prob": _format_float(snapshot.bocpd_change_prob),
    "regime_bocpd_expected_run_length": _format_float(snapshot.bocpd_expected_run_length),
    "regime_bocpd_stability_score": _format_float(snapshot.bocpd_stability_score),
    "regime_bocpd_change_flag": _format_int(snapshot.bocpd_change_flag),
  }


def _session_vwap(
  *,
  base_volume: Decimal,
  quote_volume: Decimal,
  current_base_volume: Decimal | None,
  current_quote_volume: Decimal | None,
) -> Decimal | None:
  total_base = base_volume + (current_base_volume or Decimal("0"))
  total_quote = quote_volume + (current_quote_volume or Decimal("0"))
  if total_base == 0:
    return None
  return total_quote / total_base


def _ema_decimal(values: list[Decimal], *, span: int) -> Decimal | None:
  if not values:
    return None
  alpha = Decimal("2") / Decimal(span + 1)
  ema = values[0]
  for value in values[1:]:
    ema = (value * alpha) + (ema * (Decimal("1") - alpha))
  return ema


def _aggregate_trade_window(window: list[TradeEvent]) -> dict[str, str]:
  if not window:
    return {
      "trades_window_count": "0",
      "trades_window_base_volume": "",
      "trades_window_quote_volume": "",
      "trades_window_vwap": "",
      "trades_window_avg_size": "",
      "trades_window_buy_count": "0",
      "trades_window_sell_count": "0",
      "trades_window_buy_base_volume": "",
      "trades_window_sell_base_volume": "",
      "trades_window_imbalance": "",
    }

  base_volume = Decimal("0")
  quote_volume = Decimal("0")
  buy_count = 0
  sell_count = 0
  buy_volume = Decimal("0")
  sell_volume = Decimal("0")

  for trade in window:
    if trade.size is not None:
      base_volume += trade.size
    if trade.quote_notional is not None:
      quote_volume += trade.quote_notional
    elif trade.price is not None and trade.size is not None:
      quote_volume += trade.price * trade.size

    if trade.aggressor_side == "BUY":
      buy_count += 1
      if trade.size is not None:
        buy_volume += trade.size
    elif trade.aggressor_side == "SELL":
      sell_count += 1
      if trade.size is not None:
        sell_volume += trade.size

  vwap = quote_volume / base_volume if base_volume != 0 else None
  avg_size = base_volume / Decimal(len(window)) if window else None
  imbalance = _ratio(buy_volume - sell_volume, base_volume)

  return {
    "trades_window_count": str(len(window)),
    "trades_window_base_volume": _format_decimal(base_volume),
    "trades_window_quote_volume": _format_decimal(quote_volume),
    "trades_window_vwap": _format_decimal(vwap),
    "trades_window_avg_size": _format_decimal(avg_size),
    "trades_window_buy_count": str(buy_count),
    "trades_window_sell_count": str(sell_count),
    "trades_window_buy_base_volume": _format_decimal(buy_volume),
    "trades_window_sell_base_volume": _format_decimal(sell_volume),
    "trades_window_imbalance": _format_decimal(imbalance),
  }


def _aggregate_top_window(window: list[TopEvent]) -> dict[str, str]:
  if not window:
    return {
      "top_window_observation_count": "0",
      "top_window_mid_mean": "",
      "top_window_mid_last": "",
      "top_window_spread_mean": "",
      "top_window_spread_mean_bps": "",
      "top_window_bid_size_mean": "",
      "top_window_ask_size_mean": "",
      "top_window_size_imbalance_mean": "",
    }

  mids = [event.mid for event in window if event.mid is not None]
  spreads = [event.spread for event in window if event.spread is not None]
  bid_sizes = [event.bid_size for event in window if event.bid_size is not None]
  ask_sizes = [event.ask_size for event in window if event.ask_size is not None]

  spread_bps_values = [
    _ratio(event.spread, event.mid) * Decimal("10000")
    for event in window
    if event.spread is not None and event.mid not in (None, Decimal("0"))
  ]
  size_imbalance_values = [
    _ratio(event.bid_size - event.ask_size, event.bid_size + event.ask_size)
    for event in window
    if event.bid_size is not None and event.ask_size is not None and event.bid_size + event.ask_size != 0
  ]

  return {
    "top_window_observation_count": str(len(window)),
    "top_window_mid_mean": _format_decimal(_mean_decimal(mids)),
    "top_window_mid_last": _format_decimal(mids[-1] if mids else None),
    "top_window_spread_mean": _format_decimal(_mean_decimal(spreads)),
    "top_window_spread_mean_bps": _format_decimal(_mean_decimal(spread_bps_values)),
    "top_window_bid_size_mean": _format_decimal(_mean_decimal(bid_sizes)),
    "top_window_ask_size_mean": _format_decimal(_mean_decimal(ask_sizes)),
    "top_window_size_imbalance_mean": _format_decimal(_mean_decimal(size_imbalance_values)),
  }


def _advance_index(events: list[Any], index: int, start_dt: datetime) -> int:
  while index < len(events) and events[index].event_dt < start_dt:
    index += 1
  return index


def _window_end_index(events: list[Any], index: int, end_dt: datetime) -> int:
  while index < len(events) and events[index].event_dt < end_dt:
    index += 1
  return index


def _prefer_candle_row(existing: dict[str, str], candidate: dict[str, str]) -> bool:
  existing_closed = _parse_bool_optional(existing.get("is_closed")) is True
  candidate_closed = _parse_bool_optional(candidate.get("is_closed")) is True
  if candidate_closed and not existing_closed:
    return True
  if existing_closed and not candidate_closed:
    return False
  return _parse_dt(candidate.get("event_time") or candidate.get("candle_start")) >= _parse_dt(
    existing.get("event_time") or existing.get("candle_start")
  )


def _infer_candle_delta(rows: list[dict[str, str]], start_times: list[datetime]) -> timedelta:
  if len(start_times) >= 2:
    deltas = [next_time - current_time for current_time, next_time in zip(start_times, start_times[1:])]
    positive_deltas = [delta for delta in deltas if delta > timedelta(0)]
    if positive_deltas:
      return min(positive_deltas)

  interval = rows[0].get("interval", "")
  parsed_interval = _parse_interval(interval)
  return parsed_interval or DEFAULT_CANDLE_DELTA


def _resolve_window_end(
  start_dt: datetime,
  explicit_end: datetime | None,
  next_start: datetime | None,
  inferred_delta: timedelta,
) -> datetime:
  if next_start is not None and next_start > start_dt:
    return next_start
  if explicit_end is not None and explicit_end > start_dt:
    return explicit_end + timedelta(milliseconds=1)
  return start_dt + inferred_delta


def _difference(left: Decimal | None, right: Decimal | None) -> Decimal | None:
  if left is None or right is None:
    return None
  return left - right


def _ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
  if numerator is None or denominator in (None, Decimal("0")):
    return None
  return numerator / denominator


def _mean_decimal(values: list[Decimal | None]) -> Decimal | None:
  filtered = [value for value in values if value is not None]
  if not filtered:
    return None
  return sum(filtered, Decimal("0")) / Decimal(len(filtered))


def _decimal_or_none(value: Any) -> Decimal | None:
  if value in (None, ""):
    return None
  try:
    return Decimal(str(value))
  except (InvalidOperation, ValueError):
    return None


def _int_or_none(value: Any) -> int | None:
  if value in (None, ""):
    return None
  try:
    return int(str(value))
  except ValueError:
    return None


def _parse_dt(value: str | None) -> datetime:
  if value is None:
    raise ValueError("datetime value is required")
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _parse_dt_optional(value: str | None) -> datetime | None:
  if value in (None, ""):
    return None
  return _parse_dt(value)


def _parse_bool_optional(value: str | None) -> bool | None:
  if value in (None, ""):
    return None
  lowered = value.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  return None


def _parse_interval(value: str) -> timedelta | None:
  if not value:
    return None
  unit = value[-1]
  try:
    count = int(value[:-1])
  except ValueError:
    return None

  if unit == "m":
    return timedelta(minutes=count)
  if unit == "h":
    return timedelta(hours=count)
  if unit == "d":
    return timedelta(days=count)
  return None


def _format_dt(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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


def _format_int(value: int | None) -> str:
  if value is None:
    return ""
  return str(value)


def _format_bool(value: bool | None) -> str:
  if value is None:
    return ""
  return "true" if value else "false"


def _decimal_to_float(value: Decimal | None) -> float | None:
  if value is None:
    return None
  return float(value)

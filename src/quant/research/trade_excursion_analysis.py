from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from quant.risk.sizing import PositionSizingConfig


@dataclass(frozen=True, slots=True)
class TradeExcursionAnalysisConfig:
  backtest_dir: Path
  output_dir: Path | None = None
  base_sizing: PositionSizingConfig | None = None


@dataclass(frozen=True, slots=True)
class TradeExcursion:
  direction: int
  entry_candle_start: str
  exit_candle_start: str
  bars: int
  terminal_open_return: float
  segment_net_return: float
  max_favorable_return: float
  max_adverse_return: float
  peak_giveback_return: float
  entry_stop_distance_rate: float
  time_to_peak_bars: int
  exit_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class TradeExcursionAnalysisSummary:
  trade_count: int
  winning_trades: int
  losing_trades: int
  average_hold_bars: float
  median_hold_bars: float
  mfe_bps_quantiles: dict[str, float]
  mae_bps_quantiles: dict[str, float]
  peak_giveback_bps_quantiles: dict[str, float]
  time_to_peak_bars_quantiles: dict[str, float]
  entry_stop_bps_quantiles: dict[str, float]
  recommendations: dict[str, Any]
  top_trades: list[TradeExcursion]
  worst_trades: list[TradeExcursion]


def run_trade_excursion_analysis(
  config: TradeExcursionAnalysisConfig,
) -> TradeExcursionAnalysisSummary:
  periods = _load_period_rows(config.backtest_dir)
  trades = extract_trade_excursions(periods)
  recommendations = (
    suggest_excursion_informed_exit_parameters(
      trades=trades,
      base_sizing=config.base_sizing,
    )
    if config.base_sizing is not None
    else {}
  )
  summary = summarize_trade_excursions(
    trades=trades,
    recommendations=recommendations,
  )
  output_dir = config.output_dir or config.backtest_dir
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_summary(output_dir / "trade_excursion_summary.json", summary)
  _write_trades(output_dir / "trade_excursions.csv", trades)
  return summary


def extract_trade_excursions(periods: list[dict[str, str]]) -> list[TradeExcursion]:
  excursions: list[TradeExcursion] = []
  current_rows: list[dict[str, str]] = []
  current_direction = 0
  for index, row in enumerate(periods):
    direction = _sign(_float(row, "target_position"))
    if direction == 0:
      if current_rows:
        excursions.append(_build_excursion(current_rows, periods, index))
        current_rows = []
        current_direction = 0
      continue
    if current_rows and direction != current_direction:
      excursions.append(_build_excursion(current_rows, periods, index))
      current_rows = []
    current_rows.append(row)
    current_direction = direction
  if current_rows:
    excursions.append(_build_excursion(current_rows, periods, len(periods)))
  return excursions


def summarize_trade_excursions(
  *,
  trades: list[TradeExcursion],
  recommendations: dict[str, Any] | None = None,
) -> TradeExcursionAnalysisSummary:
  winners = [trade for trade in trades if trade.terminal_open_return > 0]
  losers = [trade for trade in trades if trade.terminal_open_return <= 0]
  average_hold_bars = statistics.mean(trade.bars for trade in trades) if trades else 0.0
  median_hold_bars = statistics.median(trade.bars for trade in trades) if trades else 0.0
  return TradeExcursionAnalysisSummary(
    trade_count=len(trades),
    winning_trades=len(winners),
    losing_trades=len(losers),
    average_hold_bars=average_hold_bars,
    median_hold_bars=median_hold_bars,
    mfe_bps_quantiles=_quantiles([trade.max_favorable_return * 10_000 for trade in trades]),
    mae_bps_quantiles=_quantiles([abs(trade.max_adverse_return) * 10_000 for trade in trades]),
    peak_giveback_bps_quantiles=_quantiles([trade.peak_giveback_return * 10_000 for trade in trades]),
    time_to_peak_bars_quantiles=_quantiles([float(trade.time_to_peak_bars) for trade in trades]),
    entry_stop_bps_quantiles=_quantiles([trade.entry_stop_distance_rate * 10_000 for trade in trades]),
    recommendations=recommendations or {},
    top_trades=sorted(trades, key=lambda trade: trade.terminal_open_return, reverse=True)[:5],
    worst_trades=sorted(trades, key=lambda trade: trade.terminal_open_return)[:5],
  )


def suggest_excursion_informed_exit_parameters(
  *,
  trades: list[TradeExcursion],
  base_sizing: PositionSizingConfig,
) -> dict[str, Any]:
  if not trades:
    return {
      "stop_loss_floor_bps_candidates": [base_sizing.stop_loss_floor_bps],
      "profit_take_bps_candidates": [],
      "trailing_activation_bps_candidates": [],
      "trailing_retrace_bps_candidates": [],
    }

  winners = [trade for trade in trades if trade.terminal_open_return > 0]
  reference_trades = winners or trades
  winner_mfe_bps = [trade.max_favorable_return * 10_000 for trade in reference_trades]
  all_mae_bps = [abs(trade.max_adverse_return) * 10_000 for trade in trades]
  giveback_bps = [trade.peak_giveback_return * 10_000 for trade in reference_trades if trade.peak_giveback_return > 0]

  winner_mfe_p25 = _percentile(winner_mfe_bps, 0.25)
  winner_mfe_p50 = _percentile(winner_mfe_bps, 0.50)
  mae_p50 = _percentile(all_mae_bps, 0.50)
  mae_p75 = _percentile(all_mae_bps, 0.75)
  giveback_p50 = _percentile(giveback_bps, 0.50) if giveback_bps else 0.0
  giveback_p75 = _percentile(giveback_bps, 0.75) if giveback_bps else 0.0

  stop_candidates = _bps_candidates(
    [
      winner_mfe_p25 * 0.50,
      winner_mfe_p25 * 0.75,
      mae_p50 * 1.10,
      mae_p75 * 1.10,
    ],
    minimum_bps=5.0,
    maximum_bps=min(base_sizing.stop_loss_floor_bps, max(10.0, winner_mfe_p50)),
  )
  if not stop_candidates:
    stop_candidates = [base_sizing.stop_loss_floor_bps]

  profit_take_bps_candidates = _bps_candidates(
    [
      winner_mfe_p25 * 0.85,
      winner_mfe_p50 * 0.85,
    ],
    minimum_bps=5.0,
    maximum_bps=max(winner_mfe_p50, winner_mfe_p25),
  )
  trailing_activation_bps_candidates = _bps_candidates(
    [
      winner_mfe_p25 * 0.60,
      winner_mfe_p25 * 0.85,
    ],
    minimum_bps=5.0,
    maximum_bps=max(winner_mfe_p25, 5.0),
  )
  trailing_retrace_bps_candidates = _bps_candidates(
    [
      giveback_p50,
      giveback_p75,
    ],
    minimum_bps=2.5,
    maximum_bps=max(giveback_p75, giveback_p50, 5.0),
  )

  profit_take_multiple_candidates = sorted(
    {
      round(profit_bps / stop_bps, 3)
      for stop_bps in stop_candidates
      for profit_bps in profit_take_bps_candidates
      if stop_bps > 0 and profit_bps > 0
    }
  )
  trailing_multiple_candidates = sorted(
    {
      (
        round(activation_bps / stop_bps, 3),
        round(retrace_bps / stop_bps, 3),
      )
      for stop_bps in stop_candidates
      for activation_bps in trailing_activation_bps_candidates
      for retrace_bps in trailing_retrace_bps_candidates
      if stop_bps > 0 and activation_bps >= retrace_bps > 0
    }
  )
  candidate_grid = build_excursion_informed_sizing_candidates(
    base_sizing=base_sizing,
    stop_loss_floor_bps_candidates=stop_candidates,
    profit_take_multiple_candidates=profit_take_multiple_candidates,
    trailing_multiple_candidates=trailing_multiple_candidates,
  )
  return {
    "stop_loss_floor_bps_candidates": stop_candidates,
    "profit_take_bps_candidates": profit_take_bps_candidates,
    "trailing_activation_bps_candidates": trailing_activation_bps_candidates,
    "trailing_retrace_bps_candidates": trailing_retrace_bps_candidates,
    "profit_take_multiple_candidates": profit_take_multiple_candidates,
    "trailing_multiple_candidates": [list(candidate) for candidate in trailing_multiple_candidates],
    "candidate_grid_size": len(candidate_grid),
  }


def build_excursion_informed_sizing_candidates(
  *,
  base_sizing: PositionSizingConfig,
  stop_loss_floor_bps_candidates: list[float],
  profit_take_multiple_candidates: list[float],
  trailing_multiple_candidates: list[tuple[float, float]],
) -> list[PositionSizingConfig]:
  candidates: list[PositionSizingConfig] = []
  seen: set[tuple[float, float, float, float]] = set()

  def append_candidate(
    *,
    stop_loss_floor_bps: float,
    profit_take_exit_multiple: float = 0.0,
    trailing_profit_activation_multiple: float = 0.0,
    trailing_profit_retrace_multiple: float = 0.0,
  ) -> None:
    key = (
      round(stop_loss_floor_bps, 6),
      round(profit_take_exit_multiple, 6),
      round(trailing_profit_activation_multiple, 6),
      round(trailing_profit_retrace_multiple, 6),
    )
    if key in seen:
      return
    seen.add(key)
    candidates.append(
      replace(
        base_sizing,
        stop_loss_floor_bps=stop_loss_floor_bps,
        profit_take_exit_multiple=profit_take_exit_multiple,
        trailing_profit_activation_multiple=trailing_profit_activation_multiple,
        trailing_profit_retrace_multiple=trailing_profit_retrace_multiple,
      )
    )

  append_candidate(stop_loss_floor_bps=base_sizing.stop_loss_floor_bps)
  for stop_bps in stop_loss_floor_bps_candidates:
    append_candidate(stop_loss_floor_bps=stop_bps)
    for profit_take_multiple in profit_take_multiple_candidates:
      append_candidate(
        stop_loss_floor_bps=stop_bps,
        profit_take_exit_multiple=profit_take_multiple,
      )
    for trailing_activation_multiple, trailing_retrace_multiple in trailing_multiple_candidates:
      append_candidate(
        stop_loss_floor_bps=stop_bps,
        trailing_profit_activation_multiple=trailing_activation_multiple,
        trailing_profit_retrace_multiple=trailing_retrace_multiple,
      )
      for profit_take_multiple in profit_take_multiple_candidates:
        append_candidate(
          stop_loss_floor_bps=stop_bps,
          profit_take_exit_multiple=profit_take_multiple,
          trailing_profit_activation_multiple=trailing_activation_multiple,
          trailing_profit_retrace_multiple=trailing_retrace_multiple,
        )
  return candidates


def _build_excursion(
  rows: list[dict[str, str]],
  periods: list[dict[str, str]],
  exit_index: int,
) -> TradeExcursion:
  max_favorable_return = max(_float(row, "max_favorable_return") for row in rows)
  max_adverse_return = min(_float(row, "max_adverse_return") for row in rows)
  time_to_peak_bars = 1
  for index, row in enumerate(rows, start=1):
    if _float(row, "max_favorable_return") >= max_favorable_return:
      time_to_peak_bars = index
      break
  terminal_open_return = _float(rows[-1], "open_return")
  peak_giveback_return = max(0.0, max_favorable_return - terminal_open_return)
  exit_flags: tuple[str, ...] = ()
  exit_candle_start = rows[-1].get("candle_start", "")
  if exit_index < len(periods):
    exit_row = periods[exit_index]
    exit_candle_start = exit_row.get("candle_start", exit_candle_start)
    exit_flags = tuple(
      flag
      for flag in str(exit_row.get("constraint_flags", "")).split("|")
      if flag
    )
  return TradeExcursion(
    direction=_sign(_float(rows[0], "target_position")),
    entry_candle_start=rows[0].get("candle_start", ""),
    exit_candle_start=exit_candle_start,
    bars=len(rows),
    terminal_open_return=terminal_open_return,
    segment_net_return=sum(_float(row, "net_return") for row in rows),
    max_favorable_return=max_favorable_return,
    max_adverse_return=max_adverse_return,
    peak_giveback_return=peak_giveback_return,
    entry_stop_distance_rate=_float(rows[0], "stop_distance_rate"),
    time_to_peak_bars=time_to_peak_bars,
    exit_flags=exit_flags,
  )


def _load_period_rows(backtest_dir: Path) -> list[dict[str, str]]:
  periods_path = backtest_dir / "periods.csv"
  with periods_path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


def _write_summary(path: Path, summary: TradeExcursionAnalysisSummary) -> None:
  payload = {
    "trade_count": summary.trade_count,
    "winning_trades": summary.winning_trades,
    "losing_trades": summary.losing_trades,
    "average_hold_bars": summary.average_hold_bars,
    "median_hold_bars": summary.median_hold_bars,
    "mfe_bps_quantiles": summary.mfe_bps_quantiles,
    "mae_bps_quantiles": summary.mae_bps_quantiles,
    "peak_giveback_bps_quantiles": summary.peak_giveback_bps_quantiles,
    "time_to_peak_bars_quantiles": summary.time_to_peak_bars_quantiles,
    "entry_stop_bps_quantiles": summary.entry_stop_bps_quantiles,
    "recommendations": summary.recommendations,
    "top_trades": [asdict(trade) for trade in summary.top_trades],
    "worst_trades": [asdict(trade) for trade in summary.worst_trades],
  }
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_trades(path: Path, trades: list[TradeExcursion]) -> None:
  fieldnames = [
    "direction",
    "entry_candle_start",
    "exit_candle_start",
    "bars",
    "terminal_open_return",
    "segment_net_return",
    "max_favorable_return",
    "max_adverse_return",
    "peak_giveback_return",
    "entry_stop_distance_rate",
    "time_to_peak_bars",
    "exit_flags",
  ]
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for trade in trades:
      payload = asdict(trade)
      payload["exit_flags"] = "|".join(trade.exit_flags)
      writer.writerow(payload)


def _quantiles(values: list[float]) -> dict[str, float]:
  if not values:
    return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
  return {
    "p25": _percentile(values, 0.25),
    "p50": _percentile(values, 0.50),
    "p75": _percentile(values, 0.75),
    "p90": _percentile(values, 0.90),
  }


def _percentile(values: list[float], quantile: float) -> float:
  if not values:
    return 0.0
  if len(values) == 1:
    return values[0]
  ordered = sorted(values)
  index = (len(ordered) - 1) * min(1.0, max(0.0, quantile))
  lower = int(index)
  upper = min(lower + 1, len(ordered) - 1)
  weight = index - lower
  return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bps_candidates(
  raw_values: list[float],
  *,
  minimum_bps: float,
  maximum_bps: float,
) -> list[float]:
  if maximum_bps <= 0:
    return []
  output = []
  seen: set[float] = set()
  for value in raw_values:
    if value <= 0:
      continue
    rounded = _round_bps(min(maximum_bps, max(minimum_bps, value)))
    if rounded in seen:
      continue
    seen.add(rounded)
    output.append(rounded)
  return sorted(output)


def _round_bps(value: float) -> float:
  if value <= 10:
    return round(value * 2) / 2
  return round(value / 5) * 5


def _float(row: dict[str, str], key: str) -> float:
  value = row.get(key, "")
  if value in ("", None):
    return 0.0
  return float(value)


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0

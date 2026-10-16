from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelWeaknessDiagnosticsConfig:
  backtest_dir: Path
  feature_root: Path
  provider: str
  symbol: str
  dataset_name: str = "market_supervised"
  trend_column: str = "trend_return_24_bar"
  trend_threshold_bps: float = 20.0
  trend_state_target_column: str = "future_trend_state_label_24_bar"
  trend_state_prediction_column: str = "oof_slow_trend_state_prediction"
  regime_change_column: str = "regime_bocpd_change_prob"
  regime_change_threshold: float = 0.35


@dataclass(slots=True)
class HoldSegmentSummary:
  direction: int
  start: str
  bars: int
  hours: float
  net_return: float
  gross_return: float
  turnover: float
  average_abs_position: float
  resize_rate: float
  mean_trend_bps: float | None


@dataclass(slots=True)
class ModelWeaknessDiagnosticsSummary:
  overall_directional_accuracy: float
  trade_bar_directional_accuracy: float
  active_bar_directional_accuracy: float
  long_bars: int
  short_bars: int
  flat_bars: int
  entries: int
  segment_count: int
  average_hold_bars: float
  median_hold_bars: float
  average_hold_hours: float
  winning_segments: int
  average_resize_rate: float
  cost_share_of_gross: float
  turnover_cap_share: float
  regime_breakdown: dict[str, dict[str, float]]
  trend_state_diagnostics: dict[str, object]
  entry_timing: dict[str, float]
  worst_segments: list[HoldSegmentSummary]
  best_segments: list[HoldSegmentSummary]
  primary_weakness: str
  secondary_weakness: str


def run_model_weakness_diagnostics(
  config: ModelWeaknessDiagnosticsConfig,
) -> ModelWeaknessDiagnosticsSummary:
  periods = _load_period_rows(config)
  if not periods:
    raise ValueError(f"No backtest period rows found in {config.backtest_dir}")

  overall_directional_accuracy = _directional_accuracy(periods)
  trade_rows = [row for row in periods if _float(row, "turnover") > 0]
  active_rows = [row for row in periods if abs(_float(row, "target_position")) > 0]
  trade_bar_directional_accuracy = _directional_accuracy(trade_rows)
  active_bar_directional_accuracy = _directional_accuracy(active_rows)

  long_bars = sum(1 for row in periods if _sign(_float(row, "target_position")) > 0)
  short_bars = sum(1 for row in periods if _sign(_float(row, "target_position")) < 0)
  flat_bars = len(periods) - long_bars - short_bars

  segments = _build_hold_segments(periods, trend_column=config.trend_column)
  entries = len(segments)
  average_hold_bars = statistics.mean(segment.bars for segment in segments) if segments else 0.0
  median_hold_bars = statistics.median(segment.bars for segment in segments) if segments else 0.0
  average_hold_hours = statistics.mean(segment.hours for segment in segments) if segments else 0.0
  winning_segments = sum(1 for segment in segments if segment.net_return > 0)
  average_resize_rate = statistics.mean(segment.resize_rate for segment in segments) if segments else 0.0

  backtest_summary = json.loads((config.backtest_dir / "summary.json").read_text(encoding="utf-8"))
  aggregate_metrics = backtest_summary["aggregate_metrics"]
  cost_share_of_gross = (
    aggregate_metrics["total_transaction_cost"] / aggregate_metrics["total_gross_return"]
    if aggregate_metrics["total_gross_return"] > 0
    else 0.0
  )
  turnover_cap_share = aggregate_metrics["turnover_cap_periods"] / aggregate_metrics["periods"]

  regime_breakdown = _regime_breakdown(
    periods=periods,
    trend_column=config.trend_column,
    trend_threshold_bps=config.trend_threshold_bps,
  )
  trend_state_diagnostics = _trend_state_diagnostics(
    periods=periods,
    target_column=config.trend_state_target_column,
    prediction_column=config.trend_state_prediction_column,
    regime_change_column=config.regime_change_column,
    regime_change_threshold=config.regime_change_threshold,
  )
  entry_timing = _entry_timing(periods)

  primary_weakness, secondary_weakness = _rank_weaknesses(
    long_bars=long_bars,
    short_bars=short_bars,
    trade_bar_directional_accuracy=trade_bar_directional_accuracy,
    overall_directional_accuracy=overall_directional_accuracy,
    average_resize_rate=average_resize_rate,
    regime_breakdown=regime_breakdown,
  )
  if (
    float(trend_state_diagnostics.get("rows", 0.0)) >= 10
    and float(trend_state_diagnostics.get("macro_recall", 1.0)) < 0.45
  ):
    primary_weakness, secondary_weakness = "trend_recognition", primary_weakness

  summary = ModelWeaknessDiagnosticsSummary(
    overall_directional_accuracy=overall_directional_accuracy,
    trade_bar_directional_accuracy=trade_bar_directional_accuracy,
    active_bar_directional_accuracy=active_bar_directional_accuracy,
    long_bars=long_bars,
    short_bars=short_bars,
    flat_bars=flat_bars,
    entries=entries,
    segment_count=len(segments),
    average_hold_bars=average_hold_bars,
    median_hold_bars=median_hold_bars,
    average_hold_hours=average_hold_hours,
    winning_segments=winning_segments,
    average_resize_rate=average_resize_rate,
    cost_share_of_gross=cost_share_of_gross,
    turnover_cap_share=turnover_cap_share,
    regime_breakdown=regime_breakdown,
    trend_state_diagnostics=trend_state_diagnostics,
    entry_timing=entry_timing,
    worst_segments=sorted(segments, key=lambda segment: segment.net_return)[:3],
    best_segments=sorted(segments, key=lambda segment: segment.net_return, reverse=True)[:3],
    primary_weakness=primary_weakness,
    secondary_weakness=secondary_weakness,
  )
  (config.backtest_dir / "weakness_diagnostics.json").write_text(
    json.dumps(_summary_payload(summary), indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return summary


def _load_period_rows(config: ModelWeaknessDiagnosticsConfig) -> list[dict[str, str]]:
  feature_rows = _load_feature_rows(
    config.feature_root,
    provider=config.provider,
    symbol=config.symbol,
    dataset_name=config.dataset_name,
  )
  periods_path = config.backtest_dir / "periods.csv"
  rows: list[dict[str, str]] = []
  with periods_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
      feature_row = feature_rows.get(row.get("candle_start", ""), {})
      rows.append({**feature_row, **row})
  return rows


def _load_feature_rows(
  feature_root: Path,
  *,
  provider: str,
  symbol: str,
  dataset_name: str,
) -> dict[str, dict[str, str]]:
  dataset_root = feature_root / "training_datasets" / dataset_name / provider / symbol
  rows: dict[str, dict[str, str]] = {}
  for path in sorted(dataset_root.rglob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        rows[row.get("candle_start", "")] = row
  return rows


def _directional_accuracy(rows: list[dict[str, str]]) -> float:
  if not rows:
    return 0.0
  return sum(
    1
    for row in rows
    if _sign(_float(row, "prediction")) == _sign(_float(row, "realized_return"))
  ) / len(rows)


def _build_hold_segments(
  periods: list[dict[str, str]],
  *,
  trend_column: str,
) -> list[HoldSegmentSummary]:
  segments: list[list[dict[str, str]]] = []
  current: list[dict[str, str]] = []
  for row in periods:
    direction = _sign(_float(row, "target_position"))
    if direction == 0:
      if current:
        segments.append(current)
        current = []
      continue
    if current and _sign(_float(current[-1], "target_position")) != direction:
      segments.append(current)
      current = []
    current.append(row)
  if current:
    segments.append(current)

  output: list[HoldSegmentSummary] = []
  for segment in segments:
    trend_values = [_float(row, trend_column) for row in segment if row.get(trend_column, "") != ""]
    output.append(
      HoldSegmentSummary(
        direction=_sign(_float(segment[0], "target_position")),
        start=segment[0]["candle_start"],
        bars=len(segment),
        hours=len(segment) * 5 / 60,
        net_return=sum(_float(row, "net_return") for row in segment),
        gross_return=sum(_float(row, "gross_return") for row in segment),
        turnover=sum(_float(row, "turnover") for row in segment),
        average_abs_position=statistics.mean(abs(_float(row, "target_position")) for row in segment),
        resize_rate=statistics.mean(1.0 if _float(row, "turnover") > 0 else 0.0 for row in segment),
        mean_trend_bps=statistics.mean(trend_values) * 10_000 if trend_values else None,
      )
    )
  return output


def _regime_breakdown(
  *,
  periods: list[dict[str, str]],
  trend_column: str,
  trend_threshold_bps: float,
) -> dict[str, dict[str, float]]:
  threshold_rate = trend_threshold_bps / 10_000
  regime_rows = {
    "uptrend": [],
    "downtrend": [],
    "range": [],
  }
  for row in periods:
    if row.get(trend_column, "") == "":
      continue
    trend_value = _float(row, trend_column)
    if trend_value >= threshold_rate:
      regime_rows["uptrend"].append(row)
    elif trend_value <= -threshold_rate:
      regime_rows["downtrend"].append(row)
    else:
      regime_rows["range"].append(row)

  breakdown: dict[str, dict[str, float]] = {}
  for regime_name, rows in regime_rows.items():
    if not rows:
      continue
    active_rows = [row for row in rows if abs(_float(row, "target_position")) > 0]
    aligned_rows = [
      row
      for row in active_rows
      if _sign(_float(row, "target_position")) == _sign(_float(row, trend_column))
    ]
    breakdown[regime_name] = {
      "rows": float(len(rows)),
      "directional_accuracy": _directional_accuracy(rows),
      "net_return": sum(_float(row, "net_return") for row in rows),
      "active_rows": float(len(active_rows)),
      "trend_alignment_rate": len(aligned_rows) / len(active_rows) if active_rows else 0.0,
    }
  return breakdown


def _trend_state_diagnostics(
  *,
  periods: list[dict[str, str]],
  target_column: str,
  prediction_column: str,
  regime_change_column: str,
  regime_change_threshold: float,
) -> dict[str, object]:
  rows = [
    row
    for row in periods
    if row.get(target_column, "") != "" and row.get(prediction_column, "") != ""
  ]
  if not rows:
    return {
      "rows": 0.0,
      "accuracy": 0.0,
      "macro_recall": 0.0,
      "transition_rows": 0.0,
      "transition_accuracy": 0.0,
      "confusion": {},
    }

  targets = [int(float(row[target_column])) for row in rows]
  predictions = [int(round(float(row[prediction_column]))) for row in rows]
  transition_rows = [
    row
    for row in rows
    if _float(row, regime_change_column) >= regime_change_threshold
  ]
  transition_targets = [int(float(row[target_column])) for row in transition_rows]
  transition_predictions = [int(round(float(row[prediction_column]))) for row in transition_rows]
  return {
    "rows": float(len(rows)),
    "accuracy": _classification_accuracy(targets, predictions),
    "macro_recall": _classification_macro_recall(targets, predictions),
    "transition_rows": float(len(transition_rows)),
    "transition_accuracy": _classification_accuracy(transition_targets, transition_predictions),
    "uptrend_recall": _class_recall(targets, predictions, 1),
    "downtrend_recall": _class_recall(targets, predictions, -1),
    "chop_recall": _class_recall(targets, predictions, 0),
    "confusion": _confusion_matrix(targets, predictions),
  }


def _entry_timing(periods: list[dict[str, str]]) -> dict[str, float]:
  entries: list[dict[str, float]] = []
  previous_position = 0.0
  for index, row in enumerate(periods):
    position = _float(row, "target_position")
    if previous_position == 0 and position != 0:
      forward_rows = periods[index : min(index + 6, len(periods))]
      entries.append(
        {
          "net_next_3": sum(_float(candidate, "net_return") for candidate in forward_rows[:3]),
          "net_next_6": sum(_float(candidate, "net_return") for candidate in forward_rows),
        }
      )
    previous_position = position

  if not entries:
    return {
      "count": 0.0,
      "positive_next_3_rate": 0.0,
      "positive_next_6_rate": 0.0,
      "average_net_next_3": 0.0,
      "average_net_next_6": 0.0,
    }
  return {
    "count": float(len(entries)),
    "positive_next_3_rate": sum(1 for entry in entries if entry["net_next_3"] > 0) / len(entries),
    "positive_next_6_rate": sum(1 for entry in entries if entry["net_next_6"] > 0) / len(entries),
    "average_net_next_3": statistics.mean(entry["net_next_3"] for entry in entries),
    "average_net_next_6": statistics.mean(entry["net_next_6"] for entry in entries),
  }


def _rank_weaknesses(
  *,
  long_bars: int,
  short_bars: int,
  trade_bar_directional_accuracy: float,
  overall_directional_accuracy: float,
  average_resize_rate: float,
  regime_breakdown: dict[str, dict[str, float]],
) -> tuple[str, str]:
  uptrend_alignment = regime_breakdown.get("uptrend", {}).get("trend_alignment_rate", 0.0)
  if short_bars > long_bars * 20 and uptrend_alignment < 0.1:
    return "trend_recognition", "timing_churn"
  if average_resize_rate > 0.8 and trade_bar_directional_accuracy < 0.5:
    return "timing_churn", "trend_recognition"
  if overall_directional_accuracy < 0.5:
    return "timing_churn", "hold_management"
  return "hold_management", "timing_churn"


def _summary_payload(summary: ModelWeaknessDiagnosticsSummary) -> dict[str, object]:
  return {
    "overall_directional_accuracy": summary.overall_directional_accuracy,
    "trade_bar_directional_accuracy": summary.trade_bar_directional_accuracy,
    "active_bar_directional_accuracy": summary.active_bar_directional_accuracy,
    "long_bars": summary.long_bars,
    "short_bars": summary.short_bars,
    "flat_bars": summary.flat_bars,
    "entries": summary.entries,
    "segment_count": summary.segment_count,
    "average_hold_bars": summary.average_hold_bars,
    "median_hold_bars": summary.median_hold_bars,
    "average_hold_hours": summary.average_hold_hours,
    "winning_segments": summary.winning_segments,
    "average_resize_rate": summary.average_resize_rate,
    "cost_share_of_gross": summary.cost_share_of_gross,
    "turnover_cap_share": summary.turnover_cap_share,
    "regime_breakdown": summary.regime_breakdown,
    "trend_state_diagnostics": summary.trend_state_diagnostics,
    "entry_timing": summary.entry_timing,
    "worst_segments": [asdict(segment) for segment in summary.worst_segments],
    "best_segments": [asdict(segment) for segment in summary.best_segments],
    "primary_weakness": summary.primary_weakness,
    "secondary_weakness": summary.secondary_weakness,
  }


def _float(row: dict[str, str], key: str) -> float:
  value = row.get(key, "")
  if value == "":
    return 0.0
  return float(value)


def _classification_accuracy(targets: list[int], predictions: list[int]) -> float:
  if not targets:
    return 0.0
  return sum(1 for target, prediction in zip(targets, predictions) if target == prediction) / len(targets)


def _classification_macro_recall(targets: list[int], predictions: list[int]) -> float:
  if not targets:
    return 0.0
  recalls = [
    _class_recall(targets, predictions, label)
    for label in sorted(set(targets))
  ]
  return sum(recalls) / len(recalls) if recalls else 0.0


def _class_recall(targets: list[int], predictions: list[int], label: int) -> float:
  label_predictions = [prediction for target, prediction in zip(targets, predictions) if target == label]
  if not label_predictions:
    return 0.0
  return sum(1 for prediction in label_predictions if prediction == label) / len(label_predictions)


def _confusion_matrix(targets: list[int], predictions: list[int]) -> dict[str, dict[str, int]]:
  labels = sorted(set(targets) | set(predictions))
  matrix = {
    str(target): {str(prediction): 0 for prediction in labels}
    for target in labels
  }
  for target, prediction in zip(targets, predictions):
    matrix[str(target)][str(prediction)] += 1
  return matrix


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0

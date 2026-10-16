from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant.models.base import utc_run_slug
from quant.training.datasets import discover_series, load_series_rows_for_dataset


@dataclass(frozen=True, slots=True)
class TrendLabelDiagnosticsConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("reports/trend_label_diagnostics")
  dataset_name: str = "market_supervised"
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  horizons: tuple[int, ...] = (6, 24, 96)
  label_columns: tuple[str, ...] = ()
  slope_dead_zone_bps_per_bar_candidates: tuple[float, ...] = (0.25, 0.5, 1.0)
  slope_tstat_threshold_candidates: tuple[float, ...] = (0.5, 1.0, 1.5)


@dataclass(frozen=True, slots=True)
class TrendLabelDiagnosticsSummary:
  output_path: Path
  series_processed: int
  rows: int


def run_trend_label_diagnostics(config: TrendLabelDiagnosticsConfig) -> TrendLabelDiagnosticsSummary:
  output: dict[str, Any] = {
    "dataset_name": config.dataset_name,
    "horizons": list(config.horizons),
    "current_labels": {},
    "candidate_state_labels": {},
    "series": {},
  }
  series_processed = 0
  total_rows = 0
  all_rows: list[dict[str, str]] = []

  for series_key in discover_series(
    config.input_root,
    dataset_name=config.dataset_name,
    providers=config.providers,
    symbols=config.symbols,
  ):
    rows = load_series_rows_for_dataset(config.input_root, series_key, dataset_name=config.dataset_name)
    if not rows:
      continue
    series_processed += 1
    total_rows += len(rows)
    all_rows.extend(rows)
    output["series"][f"{series_key.provider}:{series_key.symbol}"] = _diagnostics_payload(rows, config)

  output["current_labels"] = _current_label_summaries(all_rows, config)
  output["candidate_state_labels"] = _candidate_state_label_summaries(all_rows, config)
  output["series_processed"] = series_processed
  output["rows"] = total_rows

  output_path = config.output_root / f"{utc_run_slug()}.json"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
  return TrendLabelDiagnosticsSummary(output_path=output_path, series_processed=series_processed, rows=total_rows)


def _diagnostics_payload(rows: list[dict[str, str]], config: TrendLabelDiagnosticsConfig) -> dict[str, Any]:
  return {
    "rows": len(rows),
    "current_labels": _current_label_summaries(rows, config),
    "candidate_state_labels": _candidate_state_label_summaries(rows, config),
  }


def _current_label_summaries(rows: list[dict[str, str]], config: TrendLabelDiagnosticsConfig) -> dict[str, Any]:
  columns = config.label_columns or _default_label_columns(config.horizons)
  return {
    column: _label_summary(rows, column)
    for column in columns
  }


def _candidate_state_label_summaries(rows: list[dict[str, str]], config: TrendLabelDiagnosticsConfig) -> dict[str, Any]:
  output: dict[str, Any] = {}
  for horizon in config.horizons:
    slope_column = f"future_trend_slope_bps_per_bar_{horizon}_bar"
    tstat_column = f"future_trend_slope_tstat_{horizon}_bar"
    if not any(row.get(slope_column, "") != "" and row.get(tstat_column, "") != "" for row in rows):
      continue
    horizon_rows: list[dict[str, Any]] = []
    for dead_zone in config.slope_dead_zone_bps_per_bar_candidates:
      for tstat_threshold in config.slope_tstat_threshold_candidates:
        labels = [
          _candidate_state_label(row, slope_column, tstat_column, dead_zone, tstat_threshold)
          for row in rows
        ]
        labels = [label for label in labels if label is not None]
        horizon_rows.append(
          {
            "slope_dead_zone_bps_per_bar": dead_zone,
            "slope_tstat_threshold": tstat_threshold,
            "summary": _value_summary(labels),
          }
        )
    output[f"{horizon}_bar"] = horizon_rows
  return output


def _candidate_state_label(
  row: dict[str, str],
  slope_column: str,
  tstat_column: str,
  dead_zone: float,
  tstat_threshold: float,
) -> int | None:
  if row.get(slope_column, "") == "" or row.get(tstat_column, "") == "":
    return None
  slope = float(row[slope_column])
  tstat = float(row[tstat_column])
  if abs(tstat) < tstat_threshold:
    return 0
  if slope > dead_zone:
    return 1
  if slope < -dead_zone:
    return -1
  return 0


def _label_summary(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
  values = [
    int(float(row[column]))
    for row in rows
    if row.get(column, "") != ""
  ]
  summary = _value_summary(values)
  summary["missing"] = len(rows) - len(values)
  summary["coverage"] = len(values) / len(rows) if rows else 0.0
  return summary


def _value_summary(values: list[int]) -> dict[str, Any]:
  counts: dict[str, int] = {}
  for value in values:
    counts[str(value)] = counts.get(str(value), 0) + 1
  total = len(values)
  shares = {
    label: count / total
    for label, count in counts.items()
  } if total else {}
  max_share = max(shares.values()) if shares else 0.0
  return {
    "rows": total,
    "counts": counts,
    "shares": shares,
    "max_class_share": max_share,
    "minority_class_share": min(shares.values()) if shares else 0.0,
    "imbalanced": max_share >= 0.7 if shares else False,
  }


def _default_label_columns(horizons: tuple[int, ...]) -> tuple[str, ...]:
  columns: list[str] = []
  for horizon in horizons:
    columns.extend(
      (
        f"future_trend_state_label_{horizon}_bar",
        f"future_trend_break_label_{horizon}_bar",
        f"future_trend_persistence_label_{horizon}_bar",
        f"triple_barrier_label_{horizon}_bar",
      )
    )
  return tuple(columns)

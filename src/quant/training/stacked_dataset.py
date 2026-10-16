from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from quant.training.supervised_dataset import (
  DatasetKey,
  dataset_file_path,
  read_rows,
  write_rows,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PredictionSource:
  family: str
  model_name: str
  alias: str
  source_columns: tuple[str, ...] = ("prediction",)
  output_columns: tuple[str, ...] = ()

  @property
  def prediction_column(self) -> str:
    return f"oof_{self.alias}_prediction"

  @property
  def column_mappings(self) -> tuple[tuple[str, str], ...]:
    if self.output_columns:
      if len(self.output_columns) != len(self.source_columns):
        raise ValueError(
          f"PredictionSource {self.alias} has mismatched source/output columns."
        )
      return tuple(zip(self.source_columns, self.output_columns))
    if self.source_columns == ("prediction",):
      return ((self.source_columns[0], self.prediction_column),)
    return tuple(
      (
        source_column,
        f"oof_{self.alias}_{source_column.removeprefix('prediction_')}",
      )
      for source_column in self.source_columns
    )


@dataclass(frozen=True, slots=True)
class StackedDatasetConfig:
  input_root: Path = Path("data/features")
  prediction_root: Path = Path("data/models/walk_forward")
  output_root: Path = Path("data/features")
  input_dataset_name: str = "market_supervised"
  output_dataset_name: str = "market_stacked"
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  limit_datasets: int | None = None
  prediction_sources: tuple[PredictionSource, ...] = (
    PredictionSource("tabular", "ridge_return", "ridge_return"),
    PredictionSource("tabular", "histgb_return", "histgb_return"),
    PredictionSource("tabular", "logistic_direction", "logistic_direction"),
    PredictionSource("tabular", "histgb_direction", "histgb_direction"),
    PredictionSource("transformers", "return_forecaster", "transformer_return_forecaster"),
  )


@dataclass(slots=True)
class StackedDatasetSummary:
  datasets_processed: int = 0
  output_files_written: int = 0
  stacked_rows_written: int = 0
  prediction_runs_used: int = 0


def build_stacked_datasets(config: StackedDatasetConfig) -> StackedDatasetSummary:
  summary = StackedDatasetSummary()
  prediction_cache: dict[tuple[str, str], tuple[dict[str, dict[str, str]], dict[str, str]]] = {}

  for index, dataset in enumerate(
    discover_supervised_training_datasets(
      config.input_root,
      dataset_name=config.input_dataset_name,
      providers=config.providers,
      symbols=config.symbols,
    ),
    start=1,
  ):
    if config.limit_datasets is not None and index > config.limit_datasets:
      break

    key = (dataset.provider, dataset.symbol)
    if key not in prediction_cache:
      prediction_cache[key] = load_prediction_lookups(
        config.prediction_root,
        provider=dataset.provider,
        symbol=dataset.symbol,
        prediction_sources=config.prediction_sources,
      )

    prediction_lookup, run_manifest = prediction_cache[key]
    if not prediction_lookup:
      LOGGER.info("Skipping stacked dataset for %s %s because no walk-forward predictions were found", dataset.provider, dataset.symbol)
      continue

    supervised_path = dataset_file_path(
      config.input_root,
      f"training_datasets/{config.input_dataset_name}",
      dataset,
    )
    if not supervised_path.exists():
      continue

    stacked_rows, fieldnames = join_supervised_and_prediction_rows(
      supervised_path,
      prediction_lookup,
      prediction_sources=config.prediction_sources,
    )
    if not stacked_rows:
      LOGGER.info("No stacked rows were available for %s %s on %s", dataset.provider, dataset.symbol, dataset.feature_date)
      continue

    output_path = dataset_file_path(
      config.output_root,
      f"training_datasets/{config.output_dataset_name}",
      dataset,
    )
    write_rows(output_path, fieldnames, stacked_rows)
    _write_manifest(output_path.with_suffix(".manifest.json"), run_manifest)
    LOGGER.info("Wrote %s stacked rows to %s", len(stacked_rows), output_path)

    summary.datasets_processed += 1
    summary.output_files_written += 2
    summary.stacked_rows_written += len(stacked_rows)
    summary.prediction_runs_used = max(summary.prediction_runs_used, len(run_manifest))

  return summary


def discover_supervised_training_datasets(
  input_root: Path,
  *,
  dataset_name: str = "market_supervised",
  providers: tuple[str, ...] = (),
  symbols: tuple[str, ...] = (),
) -> list[DatasetKey]:
  root = input_root / "training_datasets" / dataset_name
  if not root.exists():
    return []

  provider_filter = set(providers)
  symbol_filter = set(symbols)
  datasets: dict[tuple[str, str, str, str, str], DatasetKey] = {}

  for path in sorted(root.rglob("*.csv")):
    parts = path.relative_to(root).parts
    if len(parts) != 5:
      continue
    provider, symbol, year, month, filename = parts
    day = Path(filename).stem
    if provider_filter and provider not in provider_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue
    key = (provider, symbol, year, month, day)
    datasets[key] = DatasetKey(provider=provider, symbol=symbol, year=year, month=month, day=day)

  return [datasets[key] for key in sorted(datasets)]


def load_prediction_lookups(
  prediction_root: Path,
  *,
  provider: str,
  symbol: str,
  prediction_sources: tuple[PredictionSource, ...],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
  combined_lookup: dict[str, dict[str, str]] = {}
  run_manifest: dict[str, str] = {}

  for source in prediction_sources:
    run_dir = latest_prediction_run(
      prediction_root,
      family=source.family,
      model_name=source.model_name,
      provider=provider,
      symbol=symbol,
    )
    if run_dir is None:
      LOGGER.info("Missing walk-forward run for %s/%s %s %s", source.family, source.model_name, provider, symbol)
      return {}, {}

    run_manifest[source.alias] = str(run_dir)
    prediction_path = run_dir / "validation_predictions.csv"
    if not prediction_path.exists():
      LOGGER.info("Missing validation predictions at %s", prediction_path)
      return {}, {}

    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        candle_start = row.get("candle_start", "")
        if candle_start == "":
          continue
        combined_lookup.setdefault(candle_start, {})
        for source_column, output_column in source.column_mappings:
          combined_lookup[candle_start][output_column] = row.get(source_column, "")

  return combined_lookup, run_manifest


def latest_prediction_run(
  prediction_root: Path,
  *,
  family: str,
  model_name: str,
  provider: str,
  symbol: str,
) -> Path | None:
  root = prediction_root / family / model_name / provider / symbol
  if not root.exists():
    return None

  run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
  if not run_dirs:
    return None
  return run_dirs[-1]


def join_supervised_and_prediction_rows(
  supervised_path: Path,
  prediction_lookup: dict[str, dict[str, str]],
  *,
  prediction_sources: tuple[PredictionSource, ...],
) -> tuple[list[dict[str, str]], list[str]]:
  supervised_rows, supervised_fields = read_rows(supervised_path)
  prediction_fields = [
    output_column
    for source in prediction_sources
    for _source_column, output_column in source.column_mappings
  ]
  fieldnames = [
    *supervised_fields,
    *[field for field in prediction_fields if field not in supervised_fields],
    "has_complete_oof_predictions",
  ]

  output_rows: list[dict[str, str]] = []
  for row in supervised_rows:
    candle_start = row.get("candle_start", "")
    prediction_row = prediction_lookup.get(candle_start)
    if prediction_row is None:
      continue
    if any(prediction_row.get(field, "") == "" for field in prediction_fields):
      continue

    merged = dict(row)
    for field in prediction_fields:
      merged[field] = prediction_row[field]
    merged["has_complete_oof_predictions"] = "true"
    output_rows.append(merged)

  return output_rows, fieldnames


def _write_manifest(path: Path, manifest: dict[str, str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"prediction_runs": manifest}, indent=2, sort_keys=True), encoding="utf-8")

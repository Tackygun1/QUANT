from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

JOIN_KEYS = ("provider", "symbol", "feature_date", "candle_start", "candle_end")


@dataclass(frozen=True, slots=True)
class SupervisedDatasetConfig:
  input_root: Path = Path("data/features")
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
class SupervisedDatasetSummary:
  datasets_processed: int = 0
  output_files_written: int = 0
  joined_rows_written: int = 0


def build_supervised_datasets(config: SupervisedDatasetConfig) -> SupervisedDatasetSummary:
  summary = SupervisedDatasetSummary()
  for index, dataset in enumerate(discover_datasets(config), start=1):
    if config.limit_datasets is not None and index > config.limit_datasets:
      break

    feature_path = dataset_file_path(config.input_root, "market_features", dataset)
    label_path = dataset_file_path(config.input_root, "market_labels", dataset)
    if not feature_path.exists() or not label_path.exists():
      LOGGER.info("Skipping %s %s because feature or label file is missing", dataset.provider, dataset.symbol)
      continue

    joined_rows, fieldnames = join_feature_and_label_rows(feature_path, label_path)
    if not joined_rows:
      continue

    output_path = dataset_file_path(
      config.output_root,
      "training_datasets/market_supervised",
      dataset,
    )
    write_rows(output_path, fieldnames, joined_rows)
    LOGGER.info("Wrote %s joined rows to %s", len(joined_rows), output_path)

    summary.datasets_processed += 1
    summary.output_files_written += 1
    summary.joined_rows_written += len(joined_rows)

  return summary


def discover_datasets(config: SupervisedDatasetConfig) -> list[DatasetKey]:
  features_root = config.input_root / "market_features"
  labels_root = config.input_root / "market_labels"
  if not features_root.exists() or not labels_root.exists():
    return []

  provider_filter = set(config.providers)
  symbol_filter = set(config.symbols)
  feature_keys: dict[tuple[str, str, str, str, str], DatasetKey] = {}
  label_keys: set[tuple[str, str, str, str, str]] = set()

  for path in sorted(features_root.rglob("*.csv")):
    parts = path.relative_to(features_root).parts
    if len(parts) != 5:
      continue
    provider, symbol, year, month, filename = parts
    day = Path(filename).stem
    if provider_filter and provider not in provider_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue
    key = (provider, symbol, year, month, day)
    feature_keys[key] = DatasetKey(provider, symbol, year, month, day)

  for path in sorted(labels_root.rglob("*.csv")):
    parts = path.relative_to(labels_root).parts
    if len(parts) != 5:
      continue
    provider, symbol, year, month, filename = parts
    day = Path(filename).stem
    label_keys.add((provider, symbol, year, month, day))

  shared_keys = sorted(set(feature_keys) & label_keys)
  return [feature_keys[key] for key in shared_keys]


def join_feature_and_label_rows(
  feature_path: Path,
  label_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
  feature_rows, feature_fields = read_rows(feature_path)
  label_rows, label_fields = read_rows(label_path)
  label_lookup = {
    tuple(row.get(key, "") for key in JOIN_KEYS): row
    for row in label_rows
  }

  joined_rows: list[dict[str, str]] = []
  fieldnames = [
    *feature_fields,
    *[field for field in label_fields if field not in feature_fields],
    "has_any_forward_label",
    "has_1_bar_label",
  ]

  for feature_row in feature_rows:
    key = tuple(feature_row.get(column, "") for column in JOIN_KEYS)
    label_row = label_lookup.get(key)
    if label_row is None:
      continue

    merged = dict(feature_row)
    for field in label_fields:
      if field not in merged:
        merged[field] = label_row.get(field, "")

    merged["has_any_forward_label"] = _format_bool(
      any(
        label_row.get(field, "")
        for field in label_fields
        if field.startswith("forward_return_") and field.endswith("_bar")
      )
    )
    merged["has_1_bar_label"] = _format_bool(bool(label_row.get("forward_return_1_bar", "")))
    joined_rows.append(merged)

  return joined_rows, fieldnames


def dataset_file_path(root: Path, table_name: str, dataset: DatasetKey) -> Path:
  return root / table_name / dataset.provider / dataset.symbol / dataset.year / dataset.month / f"{dataset.day}.csv"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _format_bool(value: bool) -> str:
  return "true" if value else "false"

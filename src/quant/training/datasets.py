from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


METADATA_COLUMNS = {
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
  "has_any_forward_label",
  "has_1_bar_label",
}

LABEL_PREFIXES = (
  "forward_return_",
  "forward_log_return_",
  "direction_label_",
  "tradeable_label_",
  "legacy_tradeable_label_",
  "future_trend_",
  "triple_barrier_",
  "realized_volatility_",
  "future_abs_return_",
  "oracle_position_label_",
  "oracle_tradeability_",
)

FEATURE_GROUP_DEFINITIONS: dict[str, dict[str, tuple[str, ...]]] = {
  "price_core": {
    "columns": (
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
    ),
    "prefixes": (),
  },
  "baseline_core": {
    "columns": (
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
    ),
    "prefixes": (
      "trades_window_",
      "top_window_",
    ),
  },
  "trend_layer": {
    "columns": (),
    "prefixes": ("trend_",),
  },
  "trend_regime_layer": {
    "columns": (
      "trend_return_24_bar",
      "trend_return_96_bar",
      "trend_close_vs_ema_24_pct",
      "trend_close_vs_ema_96_pct",
      "trend_ema_spread_6_24_pct",
      "trend_ema_spread_24_96_pct",
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
    ),
    "prefixes": (),
  },
  "regime_layer": {
    "columns": (),
    "prefixes": ("regime_",),
  },
  "structure_layer": {
    "columns": (),
    "prefixes": ("structure_",),
  },
  "layered_context": {
    "columns": (),
    "prefixes": (
      "trend_",
      "structure_",
      "regime_",
    ),
  },
  "trade_flow_layer": {
    "columns": (),
    "prefixes": ("trades_window_",),
  },
  "order_book_layer": {
    "columns": (),
    "prefixes": ("top_window_",),
  },
  "orderflow_layer": {
    "columns": (),
    "prefixes": (
      "trades_window_",
      "top_window_",
    ),
  },
  "hybrid_transformer_oof": {
    "columns": (),
    "prefixes": ("oof_hybrid_transformer_",),
  },
  "trend_state_oof": {
    "columns": (),
    "prefixes": ("oof_trend_state_",),
  },
  "fast_trend_state_oof": {
    "columns": (),
    "prefixes": ("oof_fast_trend_state_",),
  },
  "slow_trend_state_oof": {
    "columns": (),
    "prefixes": ("oof_slow_trend_state_",),
  },
  "fast_triple_barrier_oof": {
    "columns": (),
    "prefixes": ("oof_fast_triple_barrier_",),
  },
  "all_oof": {
    "columns": (),
    "prefixes": ("oof_",),
  },
}


@dataclass(frozen=True, slots=True)
class SeriesKey:
  provider: str
  symbol: str


@dataclass(slots=True)
class TabularDataset:
  feature_names: list[str]
  target_column: str
  rows: list[dict[str, str]]
  features: list[list[float]]
  targets: list[float | int]


@dataclass(slots=True)
class SequenceDataset:
  feature_names: list[str]
  target_column: str
  rows: list[dict[str, str]]
  sequences: list[list[list[float]]]
  targets: list[float]


@dataclass(slots=True)
class MultiTargetSequenceDataset:
  feature_names: list[str]
  target_columns: list[str]
  auxiliary_target_columns: list[str]
  rows: list[dict[str, str]]
  sequences: list[list[list[float]]]
  targets: list[list[float]]
  auxiliary_targets: list[list[int]]


def discover_series(
  input_root: Path,
  *,
  dataset_name: str = "market_supervised",
  providers: tuple[str, ...] = (),
  symbols: tuple[str, ...] = (),
) -> list[SeriesKey]:
  root = input_root / "training_datasets" / dataset_name
  if not root.exists():
    return []

  provider_filter = set(providers)
  symbol_filter = set(symbols)
  series: dict[tuple[str, str], SeriesKey] = {}

  for path in sorted(root.rglob("*.csv")):
    parts = path.relative_to(root).parts
    if len(parts) != 5:
      continue
    provider, symbol, _year, _month, _filename = parts
    if provider_filter and provider not in provider_filter:
      continue
    if symbol_filter and symbol not in symbol_filter:
      continue
    series[(provider, symbol)] = SeriesKey(provider=provider, symbol=symbol)

  return list(series.values())


def load_series_rows(input_root: Path, series_key: SeriesKey) -> list[dict[str, str]]:
  return load_series_rows_for_dataset(input_root, series_key, dataset_name="market_supervised")


def load_series_rows_for_dataset(
  input_root: Path,
  series_key: SeriesKey,
  *,
  dataset_name: str,
) -> list[dict[str, str]]:
  root = input_root / "training_datasets" / dataset_name / series_key.provider / series_key.symbol
  if not root.exists():
    return []

  rows: list[dict[str, str]] = []
  for path in sorted(root.rglob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      rows.extend(csv.DictReader(handle))
  rows.sort(key=lambda row: row.get("candle_start", ""))
  return rows


def select_feature_columns(
  rows: list[dict[str, str]],
  *,
  feature_groups: tuple[str, ...] = (),
  exclude_feature_groups: tuple[str, ...] = (),
  feature_prefixes: tuple[str, ...] = (),
  exclude_feature_prefixes: tuple[str, ...] = (),
  include_feature_columns: tuple[str, ...] = (),
  exclude_feature_columns: tuple[str, ...] = (),
) -> list[str]:
  if not rows:
    return []

  feature_names: list[str] = []
  for column in rows[0].keys():
    if column in METADATA_COLUMNS:
      continue
    if any(column.startswith(prefix) for prefix in LABEL_PREFIXES):
      continue
    if _is_numeric_column(rows, column):
      feature_names.append(column)
  return filter_feature_columns(
    feature_names,
    feature_groups=feature_groups,
    exclude_feature_groups=exclude_feature_groups,
    feature_prefixes=feature_prefixes,
    exclude_feature_prefixes=exclude_feature_prefixes,
    include_feature_columns=include_feature_columns,
    exclude_feature_columns=exclude_feature_columns,
  )


def filter_feature_columns(
  feature_names: list[str],
  *,
  feature_groups: tuple[str, ...] = (),
  exclude_feature_groups: tuple[str, ...] = (),
  feature_prefixes: tuple[str, ...] = (),
  exclude_feature_prefixes: tuple[str, ...] = (),
  include_feature_columns: tuple[str, ...] = (),
  exclude_feature_columns: tuple[str, ...] = (),
) -> list[str]:
  _validate_feature_groups(feature_groups)
  _validate_feature_groups(exclude_feature_groups)

  include_active = bool(feature_groups or feature_prefixes or include_feature_columns)
  include_columns = set(include_feature_columns)
  exclude_columns = set(exclude_feature_columns)

  filtered: list[str] = []
  for column in feature_names:
    if include_active and not _column_matches(
      column,
      feature_groups=feature_groups,
      feature_prefixes=feature_prefixes,
      feature_columns=include_columns,
    ):
      continue
    if _column_matches(
      column,
      feature_groups=exclude_feature_groups,
      feature_prefixes=exclude_feature_prefixes,
      feature_columns=exclude_columns,
    ):
      continue
    filtered.append(column)
  return filtered


def build_tabular_dataset(
  rows: list[dict[str, str]],
  *,
  target_column: str,
  feature_columns: list[str] | None = None,
  classification: bool,
  require_closed_rows: bool = True,
  feature_groups: tuple[str, ...] = (),
  exclude_feature_groups: tuple[str, ...] = (),
  feature_prefixes: tuple[str, ...] = (),
  exclude_feature_prefixes: tuple[str, ...] = (),
  include_feature_columns: tuple[str, ...] = (),
  exclude_feature_columns: tuple[str, ...] = (),
) -> TabularDataset:
  selected_features = feature_columns if feature_columns is not None else select_feature_columns(
    rows,
    feature_groups=feature_groups,
    exclude_feature_groups=exclude_feature_groups,
    feature_prefixes=feature_prefixes,
    exclude_feature_prefixes=exclude_feature_prefixes,
    include_feature_columns=include_feature_columns,
    exclude_feature_columns=exclude_feature_columns,
  )
  output_rows: list[dict[str, str]] = []
  features: list[list[float]] = []
  targets: list[float | int] = []

  for row in rows:
    if require_closed_rows and row.get("is_closed") != "true":
      continue
    target_value = row.get(target_column, "")
    if target_value == "":
      continue

    target = int(target_value) if classification else float(target_value)
    feature_vector = [_numeric_value(row.get(column, "")) for column in selected_features]

    output_rows.append(row)
    features.append(feature_vector)
    targets.append(target)

  return TabularDataset(
    feature_names=selected_features,
    target_column=target_column,
    rows=output_rows,
    features=features,
    targets=targets,
  )


def build_sequence_dataset(
  rows: list[dict[str, str]],
  *,
  target_column: str,
  lookback: int,
  feature_columns: list[str] | None = None,
  require_closed_rows: bool = True,
  feature_groups: tuple[str, ...] = (),
  exclude_feature_groups: tuple[str, ...] = (),
  feature_prefixes: tuple[str, ...] = (),
  exclude_feature_prefixes: tuple[str, ...] = (),
  include_feature_columns: tuple[str, ...] = (),
  exclude_feature_columns: tuple[str, ...] = (),
) -> SequenceDataset:
  selected_features = feature_columns if feature_columns is not None else select_feature_columns(
    rows,
    feature_groups=feature_groups,
    exclude_feature_groups=exclude_feature_groups,
    feature_prefixes=feature_prefixes,
    exclude_feature_prefixes=exclude_feature_prefixes,
    include_feature_columns=include_feature_columns,
    exclude_feature_columns=exclude_feature_columns,
  )
  usable_rows = [
    row
    for row in rows
    if not require_closed_rows or row.get("is_closed") == "true"
  ]

  output_rows: list[dict[str, str]] = []
  sequences: list[list[list[float]]] = []
  targets: list[float] = []

  for index in range(lookback - 1, len(usable_rows)):
    row = usable_rows[index]
    target_value = row.get(target_column, "")
    if target_value == "":
      continue

    window = usable_rows[index - lookback + 1 : index + 1]
    sequence = [
      [_numeric_value(window_row.get(column, "")) for column in selected_features]
      for window_row in window
    ]
    output_rows.append(row)
    sequences.append(sequence)
    targets.append(float(target_value))

  return SequenceDataset(
    feature_names=selected_features,
    target_column=target_column,
    rows=output_rows,
    sequences=sequences,
    targets=targets,
  )


def build_multi_target_sequence_dataset(
  rows: list[dict[str, str]],
  *,
  target_columns: tuple[str, ...] | list[str],
  auxiliary_target_columns: tuple[str, ...] | list[str] = (),
  lookback: int,
  feature_columns: list[str] | None = None,
  require_closed_rows: bool = True,
  feature_groups: tuple[str, ...] = (),
  exclude_feature_groups: tuple[str, ...] = (),
  feature_prefixes: tuple[str, ...] = (),
  exclude_feature_prefixes: tuple[str, ...] = (),
  include_feature_columns: tuple[str, ...] = (),
  exclude_feature_columns: tuple[str, ...] = (),
) -> MultiTargetSequenceDataset:
  selected_features = feature_columns if feature_columns is not None else select_feature_columns(
    rows,
    feature_groups=feature_groups,
    exclude_feature_groups=exclude_feature_groups,
    feature_prefixes=feature_prefixes,
    exclude_feature_prefixes=exclude_feature_prefixes,
    include_feature_columns=include_feature_columns,
    exclude_feature_columns=exclude_feature_columns,
  )
  selected_targets = list(target_columns)
  usable_rows = [
    row
    for row in rows
    if not require_closed_rows or row.get("is_closed") == "true"
  ]

  output_rows: list[dict[str, str]] = []
  sequences: list[list[list[float]]] = []
  targets: list[list[float]] = []
  auxiliary_targets: list[list[int]] = []

  for index in range(lookback - 1, len(usable_rows)):
    row = usable_rows[index]
    target_values = [row.get(column, "") for column in selected_targets]
    auxiliary_target_values = [row.get(column, "") for column in auxiliary_target_columns]
    if any(value == "" for value in target_values) or any(value == "" for value in auxiliary_target_values):
      continue

    window = usable_rows[index - lookback + 1 : index + 1]
    sequence = [
      [_numeric_value(window_row.get(column, "")) for column in selected_features]
      for window_row in window
    ]
    output_rows.append(row)
    sequences.append(sequence)
    targets.append([float(value) for value in target_values])
    auxiliary_targets.append([int(float(value)) for value in auxiliary_target_values])

  return MultiTargetSequenceDataset(
    feature_names=selected_features,
    target_columns=selected_targets,
    auxiliary_target_columns=list(auxiliary_target_columns),
    rows=output_rows,
    sequences=sequences,
    targets=targets,
    auxiliary_targets=auxiliary_targets,
  )


def chronological_split(length: int, validation_fraction: float) -> tuple[list[int], list[int]]:
  if length == 0:
    return [], []
  if length == 1:
    return [0], []

  validation_rows = max(1, int(length * validation_fraction))
  split_index = max(1, length - validation_rows)
  train_indices = list(range(split_index))
  validation_indices = list(range(split_index, length))
  if not validation_indices:
    return train_indices, []
  return train_indices, validation_indices


def subset_tabular_dataset(dataset: TabularDataset, indices: list[int]) -> TabularDataset:
  return TabularDataset(
    feature_names=list(dataset.feature_names),
    target_column=dataset.target_column,
    rows=[dataset.rows[index] for index in indices],
    features=[dataset.features[index] for index in indices],
    targets=[dataset.targets[index] for index in indices],
  )


def subset_sequence_dataset(dataset: SequenceDataset, indices: list[int]) -> SequenceDataset:
  return SequenceDataset(
    feature_names=list(dataset.feature_names),
    target_column=dataset.target_column,
    rows=[dataset.rows[index] for index in indices],
    sequences=[dataset.sequences[index] for index in indices],
    targets=[dataset.targets[index] for index in indices],
  )


def subset_multi_target_sequence_dataset(
  dataset: MultiTargetSequenceDataset,
  indices: list[int],
) -> MultiTargetSequenceDataset:
  return MultiTargetSequenceDataset(
    feature_names=list(dataset.feature_names),
    target_columns=list(dataset.target_columns),
    auxiliary_target_columns=list(dataset.auxiliary_target_columns),
    rows=[dataset.rows[index] for index in indices],
    sequences=[dataset.sequences[index] for index in indices],
    targets=[dataset.targets[index] for index in indices],
    auxiliary_targets=[dataset.auxiliary_targets[index] for index in indices],
  )


def _is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
  for row in rows:
    value = row.get(column, "")
    if value == "":
      continue
    try:
      float(value)
      return True
    except ValueError:
      return False
  return False


def _numeric_value(value: str) -> float:
  if value == "":
    return 0.0
  return float(value)


def _validate_feature_groups(groups: tuple[str, ...]) -> None:
  unknown = sorted(set(groups) - set(FEATURE_GROUP_DEFINITIONS))
  if unknown:
    raise ValueError(f"Unknown feature group(s): {', '.join(unknown)}")


def _column_matches(
  column: str,
  *,
  feature_groups: tuple[str, ...],
  feature_prefixes: tuple[str, ...],
  feature_columns: set[str],
) -> bool:
  if column in feature_columns:
    return True
  if any(column.startswith(prefix) for prefix in feature_prefixes):
    return True
  for group in feature_groups:
    definition = FEATURE_GROUP_DEFINITIONS[group]
    if column in definition["columns"]:
      return True
    if any(column.startswith(prefix) for prefix in definition["prefixes"]):
      return True
  return False

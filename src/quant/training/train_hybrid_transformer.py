from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.transformers.hybrid_forecaster import (
  HybridQuantileTransformer,
  HybridTransformerConfig,
  build_auxiliary_prediction_columns,
  build_prediction_columns,
  build_target_columns,
)
from quant.training.datasets import (
  MultiTargetSequenceDataset,
  build_multi_target_sequence_dataset,
  chronological_split,
  discover_series,
  load_series_rows_for_dataset,
  subset_multi_target_sequence_dataset,
)
from quant.training.train_meta import build_meta_sample_weights


@dataclass(frozen=True, slots=True)
class HybridTransformerTrainingConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  minimum_rows: int = 128
  dataset_name: str = "market_supervised"
  target_columns: tuple[str, ...] = ()
  auxiliary_target_columns: tuple[str, ...] = ()
  feature_groups: tuple[str, ...] = ()
  exclude_feature_groups: tuple[str, ...] = ()
  feature_prefixes: tuple[str, ...] = ()
  exclude_feature_prefixes: tuple[str, ...] = ()
  include_feature_columns: tuple[str, ...] = ()
  exclude_feature_columns: tuple[str, ...] = ()
  config_path: Path | None = None
  sample_weight_scheme: str = ""
  sample_weight_target_column: str = ""
  trend_column: str = "trend_return_24_bar"
  trend_threshold_bps: float = 20.0
  regime_column: str = "regime_hmm_state_index"
  regime_change_column: str = "regime_bocpd_change_prob"
  regime_change_threshold: float = 0.35
  regime_change_boost: float = 1.0
  sample_weight_floor: float = 0.5
  sample_weight_cap: float = 3.0
  transformer: HybridTransformerConfig = HybridTransformerConfig()


@dataclass(slots=True)
class HybridTransformerTrainingSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  artifacts_written: int = 0


def train_hybrid_transformers(config: HybridTransformerTrainingConfig) -> HybridTransformerTrainingSummary:
  summary = HybridTransformerTrainingSummary()

  for series_key in discover_series(
    config.input_root,
    dataset_name=config.dataset_name,
    providers=config.providers,
    symbols=config.symbols,
  ):
    summary.runs_attempted += 1
    rows = load_series_rows_for_dataset(
      config.input_root,
      series_key,
      dataset_name=config.dataset_name,
    )
    target_columns = list(config.target_columns or tuple(build_target_columns(config.transformer.horizons)))
    dataset = build_multi_target_sequence_dataset(
      rows,
      target_columns=target_columns,
      auxiliary_target_columns=config.auxiliary_target_columns,
      lookback=config.transformer.lookback,
      feature_groups=config.feature_groups,
      exclude_feature_groups=config.exclude_feature_groups,
      feature_prefixes=config.feature_prefixes,
      exclude_feature_prefixes=config.exclude_feature_prefixes,
      include_feature_columns=config.include_feature_columns,
      exclude_feature_columns=config.exclude_feature_columns,
    )
    if len(dataset.rows) < config.minimum_rows:
      continue

    train_indices, validation_indices = chronological_split(
      len(dataset.rows),
      config.transformer.validation_fraction,
    )
    train_dataset = subset_multi_target_sequence_dataset(dataset, train_indices)
    validation_dataset = subset_multi_target_sequence_dataset(dataset, validation_indices)

    model = HybridQuantileTransformer(config.transformer)
    sample_weights = _build_sample_weights(config, train_dataset)
    metrics = model.fit(
      train_dataset.sequences,
      train_dataset.targets,
      feature_names=train_dataset.feature_names,
      target_columns=train_dataset.target_columns,
      auxiliary_targets=train_dataset.auxiliary_targets,
      auxiliary_target_columns=train_dataset.auxiliary_target_columns,
      validation_sequences=validation_dataset.sequences,
      validation_targets=validation_dataset.targets,
      validation_auxiliary_targets=validation_dataset.auxiliary_targets,
      sample_weights=sample_weights,
    )

    artifact_root = (
      config.output_root
      / "hybrid_transformers"
      / "quantile_forecaster"
      / series_key.provider
      / series_key.symbol
      / utc_run_slug()
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_path = artifact_root / "model.pt"
    metadata_path = artifact_root / "metadata.json"
    predictions_path = artifact_root / "validation_predictions.csv"

    model.save(model_path)
    prediction_columns = [
      *build_prediction_columns(train_dataset.target_columns, config.transformer.quantiles),
      *build_auxiliary_prediction_columns(model.auxiliary_target_columns, model.auxiliary_class_values),
    ]
    TrainingArtifactMetadata(
      model_name=model.model_name,
      task_type=model.task_type,
      target_column=train_dataset.target_columns[0],
      provider=series_key.provider,
      symbol=series_key.symbol,
      feature_columns=tuple(dataset.feature_names),
      train_rows=len(train_dataset.rows),
      validation_rows=len(validation_dataset.rows),
      metrics=metrics,
      extra={
        "config_path": str(config.config_path) if config.config_path else "",
        "transformer_config": asdict(config.transformer),
          "dataset_name": config.dataset_name,
          "target_columns": train_dataset.target_columns,
          "auxiliary_target_columns": train_dataset.auxiliary_target_columns,
          "auxiliary_class_values": model.auxiliary_class_values,
          "prediction_columns": prediction_columns,
          "primary_prediction_column": model.primary_prediction_column,
          "feature_groups": config.feature_groups,
          "exclude_feature_groups": config.exclude_feature_groups,
          "feature_prefixes": config.feature_prefixes,
          "exclude_feature_prefixes": config.exclude_feature_prefixes,
          "include_feature_columns": config.include_feature_columns,
          "exclude_feature_columns": config.exclude_feature_columns,
          "sample_weight_scheme": config.sample_weight_scheme,
          "sample_weight_target_column": _sample_weight_target_column(config, train_dataset),
          "regime_change_boost": config.regime_change_boost,
        },
      ).write_json(metadata_path)

    validation_predictions = (
      model.predict_output_rows(validation_dataset.sequences)
      if validation_dataset.sequences
      else []
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
      fieldnames = [
        "provider",
        "symbol",
        "candle_start",
        *[f"target_{column}" for column in train_dataset.target_columns],
        *[f"target_{column}" for column in train_dataset.auxiliary_target_columns],
        *prediction_columns,
      ]
      writer = csv.DictWriter(handle, fieldnames=fieldnames)
      writer.writeheader()
      for row, target_values, auxiliary_target_values, prediction_row in zip(
        validation_dataset.rows,
        validation_dataset.targets,
        validation_dataset.auxiliary_targets,
        validation_predictions,
      ):
        payload = {
          "provider": row.get("provider", ""),
          "symbol": row.get("symbol", ""),
          "candle_start": row.get("candle_start", ""),
        }
        for target_column, target_value in zip(train_dataset.target_columns, target_values):
          payload[f"target_{target_column}"] = target_value
        for target_column, target_value in zip(train_dataset.auxiliary_target_columns, auxiliary_target_values):
          payload[f"target_{target_column}"] = target_value
        for prediction_column in prediction_columns:
          payload[prediction_column] = prediction_row.get(prediction_column, "")
        writer.writerow(payload)

    (artifact_root / "training_config.json").write_text(
      json.dumps(
        {
          "input_root": str(config.input_root),
            "dataset_name": config.dataset_name,
            "minimum_rows": config.minimum_rows,
            "target_columns": train_dataset.target_columns,
            "auxiliary_target_columns": train_dataset.auxiliary_target_columns,
            "feature_groups": config.feature_groups,
            "exclude_feature_groups": config.exclude_feature_groups,
            "feature_prefixes": config.feature_prefixes,
            "exclude_feature_prefixes": config.exclude_feature_prefixes,
            "include_feature_columns": config.include_feature_columns,
            "exclude_feature_columns": config.exclude_feature_columns,
            "sample_weight_scheme": config.sample_weight_scheme,
            "sample_weight_target_column": _sample_weight_target_column(config, train_dataset),
            "regime_change_boost": config.regime_change_boost,
            "transformer": asdict(config.transformer),
          },
        indent=2,
        sort_keys=True,
      ),
      encoding="utf-8",
    )

    summary.runs_completed += 1
    summary.artifacts_written += 3

  return summary


def _build_sample_weights(
  config: HybridTransformerTrainingConfig,
  dataset: MultiTargetSequenceDataset,
) -> list[float]:
  if not config.sample_weight_scheme:
    return []
  target_column = _sample_weight_target_column(config, dataset)
  targets = _sample_weight_targets(target_column, dataset)
  return build_meta_sample_weights(
    rows=dataset.rows,
    targets=targets,
    scheme=config.sample_weight_scheme,
    trend_column=config.trend_column,
    trend_threshold_bps=config.trend_threshold_bps,
    regime_column=config.regime_column,
    regime_change_column=config.regime_change_column,
    regime_change_threshold=config.regime_change_threshold,
    regime_change_boost=config.regime_change_boost,
    floor=config.sample_weight_floor,
    cap=config.sample_weight_cap,
  )


def _sample_weight_target_column(
  config: HybridTransformerTrainingConfig,
  dataset: MultiTargetSequenceDataset,
) -> str:
  if config.sample_weight_target_column:
    return config.sample_weight_target_column
  if dataset.auxiliary_target_columns:
    return dataset.auxiliary_target_columns[0]
  return dataset.target_columns[0] if dataset.target_columns else ""


def _sample_weight_targets(
  target_column: str,
  dataset: MultiTargetSequenceDataset,
) -> list[float | int]:
  if target_column in dataset.auxiliary_target_columns:
    column_index = dataset.auxiliary_target_columns.index(target_column)
    return [row[column_index] for row in dataset.auxiliary_targets]
  if target_column in dataset.target_columns:
    column_index = dataset.target_columns.index(target_column)
    return [row[column_index] for row in dataset.targets]
  if dataset.auxiliary_target_columns:
    return [row[0] for row in dataset.auxiliary_targets]
  return [row[0] for row in dataset.targets]

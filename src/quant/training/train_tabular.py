from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant.models.classification_outputs import (
  build_classification_prediction_outputs,
  classification_probability_metadata,
  is_supported_ternary_classification,
)
from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.registry import TABULAR_MODEL_REGISTRY
from quant.training.datasets import (
  build_tabular_dataset,
  chronological_split,
  discover_series,
  load_series_rows,
  subset_tabular_dataset,
)
from quant.validation.metrics import task_metrics


@dataclass(frozen=True, slots=True)
class TabularTrainingConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  model_names: tuple[str, ...] = ("ridge_return", "logistic_direction", "histgb_return", "histgb_direction")
  validation_fraction: float = 0.34
  minimum_rows: int = 10
  target_overrides: dict[str, str] | None = None
  model_params: dict[str, dict[str, Any]] | None = None
  config_path: Path | None = None


@dataclass(slots=True)
class TabularTrainingSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  artifacts_written: int = 0


def train_tabular_models(config: TabularTrainingConfig) -> TabularTrainingSummary:
  summary = TabularTrainingSummary()

  for series_key in discover_series(config.input_root, providers=config.providers, symbols=config.symbols):
    rows = load_series_rows(config.input_root, series_key)
    for model_name in config.model_names:
      summary.runs_attempted += 1
      if model_name not in TABULAR_MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

      model_spec = TABULAR_MODEL_REGISTRY[model_name]
      task_type = model_spec["task_type"]
      target_column = (config.target_overrides or {}).get(model_name, model_spec["target_column"])
      params = (config.model_params or {}).get(model_name, {})
      dataset = build_tabular_dataset(
        rows,
        target_column=target_column,
        classification=task_type == "classification",
      )

      if len(dataset.rows) < config.minimum_rows:
        continue

      train_indices, validation_indices = chronological_split(
        len(dataset.rows),
        config.validation_fraction,
      )
      train_dataset = subset_tabular_dataset(dataset, train_indices)
      validation_dataset = subset_tabular_dataset(dataset, validation_indices)
      if task_type == "classification" and len(set(train_dataset.targets)) < 2:
        continue

      model = model_spec["builder"](params)
      model.fit(train_dataset.features, train_dataset.targets)

      train_predictions = model.predict(train_dataset.features)
      validation_predictions = model.predict(validation_dataset.features) if validation_dataset.features else []
      metrics = build_metrics(
        task_type=task_type,
        train_targets=train_dataset.targets,
        train_predictions=train_predictions,
        validation_targets=validation_dataset.targets,
        validation_predictions=validation_predictions,
      )
      prediction_outputs = build_prediction_outputs(
        task_type=task_type,
        target_column=target_column,
        model=model,
        features=validation_dataset.features,
        predictions=validation_predictions,
      )
      artifact_root = (
        config.output_root
        / "tabular"
        / model_name
        / series_key.provider
        / series_key.symbol
        / utc_run_slug()
      )
      artifact_root.mkdir(parents=True, exist_ok=True)
      model_path = artifact_root / "model.pkl"
      metadata_path = artifact_root / "metadata.json"
      predictions_path = artifact_root / "validation_predictions.csv"

      model.save(model_path)
      TrainingArtifactMetadata(
        model_name=model_name,
        task_type=task_type,
        target_column=target_column,
        provider=series_key.provider,
        symbol=series_key.symbol,
        feature_columns=tuple(dataset.feature_names),
        train_rows=len(train_dataset.rows),
        validation_rows=len(validation_dataset.rows),
        metrics=metrics,
        extra={
          "config_path": str(config.config_path) if config.config_path else "",
          "model_params": params,
        },
      ).write_json(metadata_path)
      write_validation_predictions(
        predictions_path,
        validation_dataset.rows,
        validation_dataset.targets,
        validation_predictions,
        prediction_outputs=prediction_outputs,
      )
      (artifact_root / "training_config.json").write_text(
        json.dumps(
          {
            "input_root": str(config.input_root),
            "model_name": model_name,
            "target_column": target_column,
            "validation_fraction": config.validation_fraction,
            "minimum_rows": config.minimum_rows,
            "model_params": params,
          },
          indent=2,
          sort_keys=True,
        ),
        encoding="utf-8",
      )

      summary.runs_completed += 1
      summary.artifacts_written += 3

  return summary


def build_metrics(
  *,
  task_type: str,
  train_targets: list[float | int],
  train_predictions: list[float],
  validation_targets: list[float | int],
  validation_predictions: list[float],
) -> dict[str, float]:
  metrics = task_metrics(task_type, train_targets, train_predictions, prefix="train_")
  metrics.update(task_metrics(task_type, validation_targets, validation_predictions, prefix="validation_"))
  return metrics


def write_validation_predictions(
  path: Path,
  rows: list[dict[str, str]],
  targets: list[float | int],
  predictions: list[float],
  *,
  prediction_outputs: list[dict[str, float]] | None = None,
) -> None:
  extra_fieldnames: list[str] = []
  for output in prediction_outputs or []:
    for fieldname in output:
      if fieldname != "prediction" and fieldname not in extra_fieldnames:
        extra_fieldnames.append(fieldname)

  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=["provider", "symbol", "candle_start", "target", "prediction", *extra_fieldnames],
    )
    writer.writeheader()
    output_rows = prediction_outputs or [{} for _prediction in predictions]
    for row, target, prediction, output in zip(rows, targets, predictions, output_rows):
      prediction_value = output.get("prediction", prediction)
      payload: dict[str, Any] = {
        "provider": row.get("provider", ""),
        "symbol": row.get("symbol", ""),
        "candle_start": row.get("candle_start", ""),
        "target": target,
        "prediction": prediction_value,
      }
      for fieldname in extra_fieldnames:
        payload[fieldname] = output.get(fieldname, "")
      writer.writerow(
        payload
      )


def build_prediction_outputs(
  *,
  task_type: str,
  target_column: str,
  model: Any,
  features: list[list[float]],
  predictions: list[float],
) -> list[dict[str, float]]:
  if is_supported_ternary_classification(task_type=task_type, target_column=target_column):
    return build_classification_prediction_outputs(
      model=model,
      features=features,
      target_column=target_column,
      predictions=predictions,
    )
  return [{"prediction": float(prediction)} for prediction in predictions]


def build_prediction_output_metadata(
  *,
  task_type: str,
  target_column: str,
  prediction_outputs: list[dict[str, float]],
) -> dict[str, Any]:
  if (
    is_supported_ternary_classification(task_type=task_type, target_column=target_column)
    and prediction_outputs
    and "prediction_confidence" in prediction_outputs[0]
  ):
    return classification_probability_metadata(target_column=target_column)
  return {}

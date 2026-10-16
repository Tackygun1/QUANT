from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.models.base import utc_run_slug
from quant.models.registry import TABULAR_MODEL_REGISTRY
from quant.models.transformers.return_forecaster import (
  ReturnSequenceTransformer,
  TransformerForecasterConfig,
)
from quant.training.datasets import (
  build_sequence_dataset,
  build_tabular_dataset,
  discover_series,
  load_series_rows,
  subset_sequence_dataset,
  subset_tabular_dataset,
)
from quant.training.train_tabular import build_prediction_outputs
from quant.validation.leakage import effective_purge_rows
from quant.validation.metrics import task_metrics


@dataclass(frozen=True, slots=True)
class TabularWalkForwardConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  model_names: tuple[str, ...] = ("ridge_return", "logistic_direction", "histgb_return", "histgb_direction")
  initial_train_rows: int = 256
  validation_rows: int = 64
  step_rows: int | None = None
  embargo_rows: int = 0
  purge_rows: int | None = None
  minimum_rows: int = 320
  target_overrides: dict[str, str] | None = None
  model_params: dict[str, dict[str, Any]] | None = None
  config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TransformerWalkForwardConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  target_column: str = "forward_return_1_bar"
  initial_train_rows: int = 256
  validation_rows: int = 64
  step_rows: int | None = None
  embargo_rows: int = 0
  purge_rows: int | None = None
  minimum_rows: int = 320
  config_path: Path | None = None
  transformer: TransformerForecasterConfig = TransformerForecasterConfig()


@dataclass(slots=True)
class WalkForwardRunSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  artifacts_written: int = 0
  folds_completed: int = 0


def expanding_window_splits(
  length: int,
  *,
  initial_train_rows: int,
  validation_rows: int,
  step_rows: int | None = None,
  embargo_rows: int = 0,
  purge_rows: int = 0,
) -> list[tuple[list[int], list[int]]]:
  if initial_train_rows < 1:
    raise ValueError("initial_train_rows must be at least 1.")
  if validation_rows < 1:
    raise ValueError("validation_rows must be at least 1.")
  if embargo_rows < 0:
    raise ValueError("embargo_rows cannot be negative.")
  if purge_rows < 0:
    raise ValueError("purge_rows cannot be negative.")

  step = step_rows or validation_rows
  if step < 1:
    raise ValueError("step_rows must be at least 1.")

  splits: list[tuple[list[int], list[int]]] = []
  train_end = initial_train_rows
  while True:
    validation_start = train_end + embargo_rows
    validation_end = validation_start + validation_rows
    if validation_end > length:
      break
    purged_train_end = max(0, train_end - purge_rows)
    if purged_train_end <= 0:
      break
    splits.append(
      (
        list(range(0, purged_train_end)),
        list(range(validation_start, validation_end)),
      )
    )
    train_end += step
  return splits


def evaluate_tabular_walk_forward(config: TabularWalkForwardConfig) -> WalkForwardRunSummary:
  summary = WalkForwardRunSummary()

  for series_key in discover_series(config.input_root, providers=config.providers, symbols=config.symbols):
    rows = load_series_rows(config.input_root, series_key)
    for model_name in config.model_names:
      summary.runs_attempted += 1
      if model_name not in TABULAR_MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

      model_spec = TABULAR_MODEL_REGISTRY[model_name]
      task_type = model_spec["task_type"]
      target_column = (config.target_overrides or {}).get(model_name, model_spec["target_column"])
      purge_rows = effective_purge_rows(config.purge_rows, target_column)
      params = (config.model_params or {}).get(model_name, {})
      dataset = build_tabular_dataset(
        rows,
        target_column=target_column,
        classification=task_type == "classification",
      )

      required_rows = max(
        config.minimum_rows,
        config.initial_train_rows + config.embargo_rows + config.validation_rows,
      )
      artifact_root = _artifact_root(
        config.output_root,
        family="tabular",
        model_name=model_name,
        provider=series_key.provider,
        symbol=series_key.symbol,
      )

      run_status = "completed"
      skip_reason = ""
      folds_attempted = 0
      folds_skipped = 0
      fold_rows: list[dict[str, Any]] = []
      prediction_rows: list[dict[str, Any]] = []

      if len(dataset.rows) < required_rows:
        run_status = "skipped"
        skip_reason = "not_enough_rows"
      else:
        splits = expanding_window_splits(
          len(dataset.rows),
          initial_train_rows=config.initial_train_rows,
          validation_rows=config.validation_rows,
          step_rows=config.step_rows,
          embargo_rows=config.embargo_rows,
          purge_rows=purge_rows,
        )
        folds_attempted = len(splits)
        for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
          train_dataset = subset_tabular_dataset(dataset, train_indices)
          validation_dataset = subset_tabular_dataset(dataset, validation_indices)
          if task_type == "classification" and len(set(train_dataset.targets)) < 2:
            folds_skipped += 1
            continue

          model = model_spec["builder"](params)
          model.fit(train_dataset.features, train_dataset.targets)
          train_predictions = model.predict(train_dataset.features)
          validation_predictions = model.predict(validation_dataset.features)
          validation_prediction_outputs = build_prediction_outputs(
            task_type=task_type,
            target_column=target_column,
            model=model,
            features=validation_dataset.features,
            predictions=validation_predictions,
          )
          metrics = {}
          metrics.update(task_metrics(task_type, train_dataset.targets, train_predictions, prefix="train_"))
          metrics.update(task_metrics(task_type, validation_dataset.targets, validation_predictions, prefix="validation_"))
          fold_rows.append(
            _fold_metadata(
              fold_index=fold_index,
              task_type=task_type,
              train_rows=train_dataset.rows,
              validation_rows=validation_dataset.rows,
              metrics=metrics,
            )
          )
          prediction_rows.extend(
            _prediction_rows(
              model_name=model_name,
              task_type=task_type,
              fold_index=fold_index,
              rows=validation_dataset.rows,
              targets=validation_dataset.targets,
              predictions=validation_predictions,
              prediction_outputs=validation_prediction_outputs,
            )
          )
        if not fold_rows and skip_reason == "":
          skip_reason = "no_usable_folds"

      artifacts_written = _write_walk_forward_artifacts(
        artifact_root=artifact_root,
        family="tabular",
        model_name=model_name,
        task_type=task_type,
        provider=series_key.provider,
        symbol=series_key.symbol,
        target_column=target_column,
        feature_columns=dataset.feature_names,
        total_rows=len(dataset.rows),
        fold_rows=fold_rows,
        prediction_rows=prediction_rows,
        folds_attempted=folds_attempted,
        folds_skipped=folds_skipped,
        window_config={
          "initial_train_rows": config.initial_train_rows,
          "validation_rows": config.validation_rows,
          "step_rows": config.step_rows or config.validation_rows,
          "embargo_rows": config.embargo_rows,
          "purge_rows": purge_rows,
          "minimum_rows": config.minimum_rows,
        },
        run_config={
          "input_root": str(config.input_root),
          "model_name": model_name,
          "target_column": target_column,
          "model_params": params,
        },
        config_path=config.config_path,
        run_status=run_status if fold_rows else "skipped",
        skip_reason=skip_reason if not fold_rows else "",
      )
      summary.artifacts_written += artifacts_written
      if fold_rows:
        summary.runs_completed += 1
        summary.folds_completed += len(fold_rows)

  return summary


def evaluate_transformer_walk_forward(config: TransformerWalkForwardConfig) -> WalkForwardRunSummary:
  summary = WalkForwardRunSummary()

  for series_key in discover_series(config.input_root, providers=config.providers, symbols=config.symbols):
    summary.runs_attempted += 1
    rows = load_series_rows(config.input_root, series_key)
    dataset = build_sequence_dataset(
      rows,
      target_column=config.target_column,
      lookback=config.transformer.lookback,
    )
    purge_rows = effective_purge_rows(config.purge_rows, config.target_column)

    required_rows = max(
      config.minimum_rows,
      config.initial_train_rows + config.embargo_rows + config.validation_rows,
    )
    artifact_root = _artifact_root(
      config.output_root,
      family="transformers",
      model_name="return_forecaster",
      provider=series_key.provider,
      symbol=series_key.symbol,
    )

    run_status = "completed"
    skip_reason = ""
    folds_attempted = 0
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    if len(dataset.rows) < required_rows:
      run_status = "skipped"
      skip_reason = "not_enough_rows"
    else:
      splits = expanding_window_splits(
        len(dataset.rows),
        initial_train_rows=config.initial_train_rows,
          validation_rows=config.validation_rows,
          step_rows=config.step_rows,
          embargo_rows=config.embargo_rows,
          purge_rows=purge_rows,
        )
      folds_attempted = len(splits)
      for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_dataset = subset_sequence_dataset(dataset, train_indices)
        validation_dataset = subset_sequence_dataset(dataset, validation_indices)
        model = ReturnSequenceTransformer(config.transformer)
        metrics = model.fit(
          train_dataset.sequences,
          train_dataset.targets,
          feature_names=train_dataset.feature_names,
          validation_sequences=validation_dataset.sequences,
          validation_targets=validation_dataset.targets,
        )
        validation_predictions = model.predict(validation_dataset.sequences)
        fold_rows.append(
          _fold_metadata(
            fold_index=fold_index,
            task_type=model.task_type,
            train_rows=train_dataset.rows,
            validation_rows=validation_dataset.rows,
            metrics=metrics,
          )
        )
        prediction_rows.extend(
          _prediction_rows(
            model_name=model.model_name,
            task_type=model.task_type,
            fold_index=fold_index,
            rows=validation_dataset.rows,
            targets=validation_dataset.targets,
            predictions=validation_predictions,
            )
          )
      if not fold_rows and skip_reason == "":
        skip_reason = "no_usable_folds"

    artifacts_written = _write_walk_forward_artifacts(
      artifact_root=artifact_root,
      family="transformers",
      model_name="transformer_return_forecaster",
      task_type="regression",
      provider=series_key.provider,
      symbol=series_key.symbol,
      target_column=config.target_column,
      feature_columns=dataset.feature_names,
      total_rows=len(dataset.rows),
      fold_rows=fold_rows,
      prediction_rows=prediction_rows,
      folds_attempted=folds_attempted,
      folds_skipped=0,
      window_config={
        "initial_train_rows": config.initial_train_rows,
        "validation_rows": config.validation_rows,
        "step_rows": config.step_rows or config.validation_rows,
        "embargo_rows": config.embargo_rows,
        "purge_rows": purge_rows,
        "minimum_rows": config.minimum_rows,
        "lookback": config.transformer.lookback,
      },
      run_config={
        "input_root": str(config.input_root),
        "target_column": config.target_column,
        "transformer": asdict(config.transformer),
      },
      config_path=config.config_path,
      run_status=run_status if fold_rows else "skipped",
      skip_reason=skip_reason if not fold_rows else "",
    )
    summary.artifacts_written += artifacts_written
    if fold_rows:
      summary.runs_completed += 1
      summary.folds_completed += len(fold_rows)

  return summary


def _artifact_root(
  output_root: Path,
  *,
  family: str,
  model_name: str,
  provider: str,
  symbol: str,
) -> Path:
  root = output_root / "walk_forward" / family / model_name / provider / symbol / utc_run_slug()
  root.mkdir(parents=True, exist_ok=True)
  return root


def _fold_metadata(
  *,
  fold_index: int,
  task_type: str,
  train_rows: list[dict[str, str]],
  validation_rows: list[dict[str, str]],
  metrics: dict[str, float],
) -> dict[str, Any]:
  row: dict[str, Any] = {
    "fold_index": fold_index,
    "task_type": task_type,
    "train_rows": len(train_rows),
    "validation_rows": len(validation_rows),
    "train_start": train_rows[0].get("candle_start", "") if train_rows else "",
    "train_end": train_rows[-1].get("candle_start", "") if train_rows else "",
    "validation_start": validation_rows[0].get("candle_start", "") if validation_rows else "",
    "validation_end": validation_rows[-1].get("candle_start", "") if validation_rows else "",
  }
  row.update(metrics)
  return row


def _prediction_rows(
  *,
  model_name: str,
  task_type: str,
  fold_index: int,
  rows: list[dict[str, str]],
  targets: list[float | int],
  predictions: list[float],
  prediction_outputs: list[dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
  output: list[dict[str, Any]] = []
  extra_outputs = prediction_outputs or [{} for _prediction in predictions]
  for row, target, prediction, extra_output in zip(rows, targets, predictions, extra_outputs):
    payload = {
      "model_name": model_name,
      "task_type": task_type,
      "fold_index": fold_index,
      "provider": row.get("provider", ""),
      "symbol": row.get("symbol", ""),
      "candle_start": row.get("candle_start", ""),
      "target": target,
      "prediction": extra_output.get("prediction", prediction),
    }
    for key, value in extra_output.items():
      if key != "prediction":
        payload[key] = value
    output.append(payload)
  return output


def _write_walk_forward_artifacts(
  *,
  artifact_root: Path,
  family: str,
  model_name: str,
  task_type: str,
  provider: str,
  symbol: str,
  target_column: str,
  feature_columns: list[str],
  total_rows: int,
  fold_rows: list[dict[str, Any]],
  prediction_rows: list[dict[str, Any]],
  folds_attempted: int,
  folds_skipped: int,
  window_config: dict[str, Any],
  run_config: dict[str, Any],
  config_path: Path | None,
  run_status: str,
  skip_reason: str,
) -> int:
  fold_metrics_path = artifact_root / "fold_metrics.csv"
  predictions_path = artifact_root / "validation_predictions.csv"
  summary_path = artifact_root / "summary.json"
  run_config_path = artifact_root / "run_config.json"

  _write_csv(
    fold_metrics_path,
    fold_rows,
    fieldnames=[
      "fold_index",
      "task_type",
      "train_rows",
      "validation_rows",
      "train_start",
      "train_end",
      "validation_start",
      "validation_end",
      "train_mae",
      "train_rmse",
      "train_directional_accuracy",
      "validation_mae",
      "validation_rmse",
      "validation_directional_accuracy",
      "train_accuracy",
      "train_macro_recall",
      "validation_accuracy",
      "validation_macro_recall",
    ],
  )
  _write_csv(
    predictions_path,
    prediction_rows,
    fieldnames=[
      "model_name",
      "task_type",
      "fold_index",
      "provider",
      "symbol",
      "candle_start",
      "target",
      "prediction",
      *_extra_prediction_fieldnames(prediction_rows),
    ],
  )

  aggregate_metrics = task_metrics(
    task_type,
    [row["target"] for row in prediction_rows],
    [float(row["prediction"]) for row in prediction_rows],
    prefix="overall_validation_",
  )
  aggregate_metrics.update(_mean_fold_metrics(fold_rows))
  summary_payload = {
    "family": family,
    "model_name": model_name,
    "task_type": task_type,
    "provider": provider,
    "symbol": symbol,
    "target_column": target_column,
    "feature_columns": feature_columns,
    "total_rows": total_rows,
    "folds_attempted": folds_attempted,
    "folds_completed": len(fold_rows),
    "folds_skipped": folds_skipped,
    "window": window_config,
    "aggregate_metrics": aggregate_metrics,
    "status": run_status,
    "skip_reason": skip_reason,
    "config_path": str(config_path) if config_path else "",
  }
  summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
  run_config_path.write_text(
    json.dumps(run_config, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return 4


def _extra_prediction_fieldnames(prediction_rows: list[dict[str, Any]]) -> list[str]:
  base_fields = {
    "model_name",
    "task_type",
    "fold_index",
    "provider",
    "symbol",
    "candle_start",
    "target",
    "prediction",
  }
  output: list[str] = []
  for row in prediction_rows:
    for key in row:
      if key not in base_fields and key not in output:
        output.append(key)
  return output


def _mean_fold_metrics(fold_rows: list[dict[str, Any]]) -> dict[str, float]:
  metric_names = sorted(
    {
      key
      for row in fold_rows
      for key in row.keys()
      if key in {
        "validation_mae",
        "validation_rmse",
        "validation_directional_accuracy",
        "validation_accuracy",
        "validation_macro_recall",
      }
    }
  )
  output: dict[str, float] = {}
  for metric_name in metric_names:
    values = [float(row[metric_name]) for row in fold_rows if metric_name in row]
    if not values:
      continue
    output[f"mean_fold_{metric_name}"] = sum(values) / len(values)
  return output


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})

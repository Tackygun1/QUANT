from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.ensembles.stacked_meta import STACKED_META_MODEL_REGISTRY
from quant.training.datasets import (
  build_tabular_dataset,
  discover_series,
  load_series_rows_for_dataset,
  subset_tabular_dataset,
)
from quant.training.train_meta import (
  build_meta_sample_weights,
  resolve_stacked_supporting_artifact_metadata,
  split_meta_training_datasets,
)
from quant.training.train_tabular import (
  build_prediction_output_metadata,
  build_prediction_outputs,
  write_validation_predictions,
)
from quant.validation.leakage import effective_purge_rows
from quant.validation.metrics import task_metrics
from quant.validation.walk_forward import expanding_window_splits


@dataclass(frozen=True, slots=True)
class MetaWalkForwardConfig:
  input_root: Path
  output_root: Path
  dataset_name: str
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  model_names: tuple[str, ...] = ("lightgbm_hybrid_direction_meta",)
  initial_train_rows: int = 8192
  validation_rows: int = 1024
  step_rows: int | None = None
  embargo_rows: int = 1
  purge_rows: int | None = None
  minimum_rows: int = 9217
  target_overrides: dict[str, str] | None = None
  model_params: dict[str, dict[str, Any]] | None = None
  feature_groups: tuple[str, ...] = ()
  exclude_feature_groups: tuple[str, ...] = ()
  feature_prefixes: tuple[str, ...] = ()
  exclude_feature_prefixes: tuple[str, ...] = ()
  include_feature_columns: tuple[str, ...] = ()
  exclude_feature_columns: tuple[str, ...] = ()
  sample_weight_scheme: str = ""
  trend_column: str = "trend_return_24_bar"
  trend_threshold_bps: float = 20.0
  regime_column: str = "regime_hmm_state_index"
  regime_change_column: str = "regime_bocpd_change_prob"
  regime_change_threshold: float = 0.35
  regime_change_boost: float = 1.0
  sample_weight_floor: float = 0.5
  sample_weight_cap: float = 3.0
  probability_calibration_method: str = "temperature"
  probability_calibration_fraction: float = 0.1
  minimum_calibration_rows: int = 128
  calibration_model_names: tuple[str, ...] = ("lightgbm_hybrid_direction_meta",)
  config_path: Path | None = None


@dataclass(slots=True)
class MetaWalkForwardSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  folds_completed: int = 0
  prediction_rows: int = 0
  output_dirs: list[Path] = field(default_factory=list)


def evaluate_meta_walk_forward(config: MetaWalkForwardConfig) -> MetaWalkForwardSummary:
  summary = MetaWalkForwardSummary()
  for series_key in discover_series(
    config.input_root,
    dataset_name=config.dataset_name,
    providers=config.providers,
    symbols=config.symbols,
  ):
    source_rows = load_series_rows_for_dataset(
      config.input_root,
      series_key,
      dataset_name=config.dataset_name,
    )
    for model_name in config.model_names:
      summary.runs_attempted += 1
      if model_name not in STACKED_META_MODEL_REGISTRY:
        raise ValueError(f"Unknown meta-model name: {model_name}")
      model_spec = STACKED_META_MODEL_REGISTRY[model_name]
      task_type = str(model_spec["task_type"])
      target_column = (config.target_overrides or {}).get(model_name, str(model_spec["target_column"]))
      params = (config.model_params or {}).get(model_name, {})
      dataset = build_tabular_dataset(
        source_rows,
        target_column=target_column,
        classification=task_type == "classification",
        feature_groups=config.feature_groups,
        exclude_feature_groups=config.exclude_feature_groups,
        feature_prefixes=config.feature_prefixes,
        exclude_feature_prefixes=config.exclude_feature_prefixes,
        include_feature_columns=config.include_feature_columns,
        exclude_feature_columns=config.exclude_feature_columns,
      )
      purge_rows = effective_purge_rows(config.purge_rows, target_column)
      required_rows = max(
        config.minimum_rows,
        config.initial_train_rows + config.embargo_rows + config.validation_rows,
      )
      if len(dataset.rows) < required_rows:
        continue

      splits = expanding_window_splits(
        len(dataset.rows),
        initial_train_rows=config.initial_train_rows,
        validation_rows=config.validation_rows,
        step_rows=config.step_rows,
        embargo_rows=config.embargo_rows,
        purge_rows=purge_rows,
      )
      fold_rows: list[dict[str, Any]] = []
      oof_rows: list[dict[str, str]] = []
      oof_targets: list[float | int] = []
      oof_predictions: list[float] = []
      oof_prediction_outputs: list[dict[str, float]] = []

      for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_dataset = subset_tabular_dataset(dataset, train_indices)
        validation_dataset = subset_tabular_dataset(dataset, validation_indices)
        if task_type == "classification" and len(set(train_dataset.targets)) < 2:
          continue
        fit_dataset, calibration_dataset = split_meta_training_datasets(
          model_name=model_name,
          task_type=task_type,
          target_column=target_column,
          train_dataset=train_dataset,
          calibration_method=config.probability_calibration_method,
          calibration_fraction=config.probability_calibration_fraction,
          minimum_calibration_rows=config.minimum_calibration_rows,
          calibration_model_names=config.calibration_model_names,
        )
        sample_weights = build_meta_sample_weights(
          rows=fit_dataset.rows,
          targets=fit_dataset.targets,
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
        model = model_spec["builder"](params)
        model.fit(fit_dataset.features, fit_dataset.targets, sample_weights=sample_weights or None)
        calibration: dict[str, Any] = {}
        if calibration_dataset is not None and hasattr(model, "fit_temperature_calibrator"):
          calibration = model.fit_temperature_calibrator(
            calibration_dataset.features,
            calibration_dataset.targets,
          )
        train_predictions = model.predict(fit_dataset.features)
        validation_predictions = model.predict(validation_dataset.features)
        prediction_outputs = build_prediction_outputs(
          task_type=task_type,
          target_column=target_column,
          model=model,
          features=validation_dataset.features,
          predictions=validation_predictions,
        )
        for output in prediction_outputs:
          output["fold_index"] = fold_index
        metrics = {
          **task_metrics(task_type, fit_dataset.targets, train_predictions, prefix="train_"),
          **task_metrics(task_type, validation_dataset.targets, validation_predictions, prefix="validation_"),
        }
        prediction_distribution = Counter(int(float(value)) for value in validation_predictions)
        fold_rows.append({
          "fold_index": fold_index,
          "fit_rows": len(fit_dataset.rows),
          "calibration_rows": len(calibration_dataset.rows) if calibration_dataset is not None else 0,
          "validation_rows": len(validation_dataset.rows),
          "fit_start": fit_dataset.rows[0].get("candle_start", ""),
          "fit_end": fit_dataset.rows[-1].get("candle_start", ""),
          "calibration_start": calibration_dataset.rows[0].get("candle_start", "") if calibration_dataset is not None else "",
          "calibration_end": calibration_dataset.rows[-1].get("candle_start", "") if calibration_dataset is not None else "",
          "validation_start": validation_dataset.rows[0].get("candle_start", ""),
          "validation_end": validation_dataset.rows[-1].get("candle_start", ""),
          "calibration_temperature": calibration.get("temperature", ""),
          "prediction_short": prediction_distribution.get(-1, 0),
          "prediction_flat": prediction_distribution.get(0, 0),
          "prediction_long": prediction_distribution.get(1, 0),
          **metrics,
        })
        oof_rows.extend(validation_dataset.rows)
        oof_targets.extend(validation_dataset.targets)
        oof_predictions.extend(float(value) for value in validation_predictions)
        oof_prediction_outputs.extend(prediction_outputs)

      if not fold_rows:
        continue
      output_dir = (
        config.output_root
        / "walk_forward"
        / "meta_models"
        / model_name
        / series_key.provider
        / series_key.symbol
        / utc_run_slug()
      )
      output_dir.mkdir(parents=True, exist_ok=True)
      prediction_metadata = build_prediction_output_metadata(
        task_type=task_type,
        target_column=target_column,
        prediction_outputs=oof_prediction_outputs,
      )
      aggregate_metrics = task_metrics(
        task_type,
        oof_targets,
        oof_predictions,
        prefix="overall_oof_",
      )
      supporting_metadata = resolve_stacked_supporting_artifact_metadata(
        config.input_root,
        dataset_name=config.dataset_name,
        series_key=series_key,
      )
      write_validation_predictions(
        output_dir / "validation_predictions.csv",
        oof_rows,
        oof_targets,
        oof_predictions,
        prediction_outputs=oof_prediction_outputs,
      )
      TrainingArtifactMetadata(
        model_name=model_name,
        task_type=task_type,
        target_column=target_column,
        provider=series_key.provider,
        symbol=series_key.symbol,
        feature_columns=tuple(dataset.feature_names),
        train_rows=0,
        validation_rows=len(oof_rows),
        metrics=aggregate_metrics,
        extra={
          "config_path": str(config.config_path) if config.config_path else "",
          "dataset_name": config.dataset_name,
          "evaluation_type": "purged_expanding_window_oof",
          "folds_completed": len(fold_rows),
          "model_params": params,
          "sample_weight_scheme": config.sample_weight_scheme,
          "probability_calibration_method": config.probability_calibration_method,
          "probability_calibration_fraction": config.probability_calibration_fraction,
          **supporting_metadata,
          **prediction_metadata,
        },
      ).write_json(output_dir / "metadata.json")
      write_fold_metrics(output_dir / "fold_metrics.csv", fold_rows)
      (output_dir / "summary.json").write_text(
        json.dumps({
          "status": "completed",
          "model_name": model_name,
          "provider": series_key.provider,
          "symbol": series_key.symbol,
          "dataset_name": config.dataset_name,
          "total_rows": len(dataset.rows),
          "oof_rows": len(oof_rows),
          "folds_completed": len(fold_rows),
          "window": {
            "initial_train_rows": config.initial_train_rows,
            "validation_rows": config.validation_rows,
            "step_rows": config.step_rows or config.validation_rows,
            "embargo_rows": config.embargo_rows,
            "purge_rows": purge_rows,
          },
          "aggregate_metrics": aggregate_metrics,
          "folds": fold_rows,
          "config_path": str(config.config_path) if config.config_path else "",
        }, indent=2, sort_keys=True),
        encoding="utf-8",
      )
      summary.runs_completed += 1
      summary.folds_completed += len(fold_rows)
      summary.prediction_rows += len(oof_rows)
      summary.output_dirs.append(output_dir)
  return summary


def write_fold_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
  fieldnames = list(rows[0]) if rows else []
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

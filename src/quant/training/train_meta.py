from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.classification_outputs import is_supported_ternary_classification
from quant.models.ensembles.stacked_meta import STACKED_META_MODEL_REGISTRY
from quant.training.datasets import (
  TabularDataset,
  build_tabular_dataset,
  chronological_split,
  discover_series,
  load_series_rows_for_dataset,
  subset_tabular_dataset,
)
from quant.training.train_tabular import (
  build_metrics,
  build_prediction_output_metadata,
  build_prediction_outputs,
  write_validation_predictions,
)


@dataclass(frozen=True, slots=True)
class MetaTrainingConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  model_names: tuple[str, ...] = ("lightgbm_return_meta", "lightgbm_direction_meta")
  validation_fraction: float = 0.25
  minimum_rows: int = 64
  dataset_name: str = "market_stacked"
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
class MetaTrainingSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  artifacts_written: int = 0


def train_meta_models(config: MetaTrainingConfig) -> MetaTrainingSummary:
  summary = MetaTrainingSummary()

  for series_key in discover_series(
    config.input_root,
    dataset_name=config.dataset_name,
    providers=config.providers,
    symbols=config.symbols,
  ):
    rows = load_series_rows_for_dataset(
      config.input_root,
      series_key,
      dataset_name=config.dataset_name,
    )
    for model_name in config.model_names:
      summary.runs_attempted += 1
      if model_name not in STACKED_META_MODEL_REGISTRY:
        raise ValueError(f"Unknown meta-model name: {model_name}")

      model_spec = STACKED_META_MODEL_REGISTRY[model_name]
      task_type = model_spec["task_type"]
      target_column = (config.target_overrides or {}).get(model_name, model_spec["target_column"])
      params = (config.model_params or {}).get(model_name, {})
      dataset = build_tabular_dataset(
        rows,
        target_column=target_column,
        classification=task_type == "classification",
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
        config.validation_fraction,
      )
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
      model.fit(
        fit_dataset.features,
        fit_dataset.targets,
        sample_weights=sample_weights or None,
      )
      probability_calibration: dict[str, Any] = {}
      if calibration_dataset is not None and hasattr(model, "fit_temperature_calibrator"):
        probability_calibration = model.fit_temperature_calibrator(
          calibration_dataset.features,
          calibration_dataset.targets,
        )
        probability_calibration["calibration_fraction"] = config.probability_calibration_fraction
        probability_calibration["calibration_rows"] = len(calibration_dataset.rows)
        probability_calibration["fit_rows"] = len(fit_dataset.rows)

      train_predictions = model.predict(fit_dataset.features)
      validation_predictions = model.predict(validation_dataset.features) if validation_dataset.features else []
      validation_prediction_outputs = build_prediction_outputs(
        task_type=task_type,
        target_column=target_column,
        model=model,
        features=validation_dataset.features,
        predictions=validation_predictions,
      )
      prediction_metadata = (
        build_prediction_output_metadata(
          task_type=task_type,
          target_column=target_column,
          prediction_outputs=validation_prediction_outputs,
        )
        if model_name == "lightgbm_hybrid_direction_meta"
        else {}
      )
      supporting_artifact_metadata = resolve_stacked_supporting_artifact_metadata(
        config.input_root,
        dataset_name=config.dataset_name,
        series_key=series_key,
      )
      metrics = build_metrics(
        task_type=task_type,
        train_targets=fit_dataset.targets,
        train_predictions=train_predictions,
        validation_targets=validation_dataset.targets,
        validation_predictions=validation_predictions,
      )

      artifact_root = (
        config.output_root
        / "ensembles"
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
        train_rows=len(fit_dataset.rows),
        validation_rows=len(validation_dataset.rows),
        metrics=metrics,
        extra={
          "config_path": str(config.config_path) if config.config_path else "",
          "model_params": params,
          "dataset_name": config.dataset_name,
          "feature_groups": config.feature_groups,
          "exclude_feature_groups": config.exclude_feature_groups,
          "feature_prefixes": config.feature_prefixes,
          "exclude_feature_prefixes": config.exclude_feature_prefixes,
          "include_feature_columns": config.include_feature_columns,
          "exclude_feature_columns": config.exclude_feature_columns,
          "sample_weight_scheme": config.sample_weight_scheme,
          "trend_column": config.trend_column,
          "trend_threshold_bps": config.trend_threshold_bps,
          "regime_column": config.regime_column,
          "regime_change_column": config.regime_change_column,
          "regime_change_threshold": config.regime_change_threshold,
          "regime_change_boost": config.regime_change_boost,
          "sample_weight_floor": config.sample_weight_floor,
          "sample_weight_cap": config.sample_weight_cap,
          "probability_calibration_method": config.probability_calibration_method,
          "probability_calibration_fraction": config.probability_calibration_fraction,
          "minimum_calibration_rows": config.minimum_calibration_rows,
          "probability_calibration": probability_calibration,
          **supporting_artifact_metadata,
          **prediction_metadata,
        },
      ).write_json(metadata_path)
      write_validation_predictions(
        predictions_path,
        validation_dataset.rows,
        validation_dataset.targets,
        validation_predictions,
        prediction_outputs=validation_prediction_outputs,
      )
      (artifact_root / "training_config.json").write_text(
        json.dumps(
          {
            "input_root": str(config.input_root),
            "dataset_name": config.dataset_name,
            "model_name": model_name,
            "target_column": target_column,
            "validation_fraction": config.validation_fraction,
            "minimum_rows": config.minimum_rows,
            "model_params": params,
            "feature_groups": config.feature_groups,
            "exclude_feature_groups": config.exclude_feature_groups,
            "feature_prefixes": config.feature_prefixes,
            "exclude_feature_prefixes": config.exclude_feature_prefixes,
            "include_feature_columns": config.include_feature_columns,
            "exclude_feature_columns": config.exclude_feature_columns,
            "sample_weight_scheme": config.sample_weight_scheme,
            "trend_column": config.trend_column,
            "trend_threshold_bps": config.trend_threshold_bps,
            "regime_column": config.regime_column,
            "regime_change_column": config.regime_change_column,
            "regime_change_threshold": config.regime_change_threshold,
            "regime_change_boost": config.regime_change_boost,
            "sample_weight_floor": config.sample_weight_floor,
            "sample_weight_cap": config.sample_weight_cap,
            "probability_calibration_method": config.probability_calibration_method,
            "probability_calibration_fraction": config.probability_calibration_fraction,
            "minimum_calibration_rows": config.minimum_calibration_rows,
            "calibration_model_names": config.calibration_model_names,
          },
          indent=2,
          sort_keys=True,
        ),
        encoding="utf-8",
      )

      summary.runs_completed += 1
      summary.artifacts_written += 3

  return summary


VALID_SAMPLE_WEIGHT_SCHEMES = {
  "",
  "trend_regime_balance",
  "trend_regime_target_balance",
  "hmm_regime_balance",
  "hmm_regime_target_balance",
}


def resolve_stacked_supporting_artifact_metadata(
  input_root: Path,
  *,
  dataset_name: str,
  series_key: Any,
) -> dict[str, Any]:
  prediction_runs = resolve_stacked_prediction_runs(
    input_root,
    dataset_name=dataset_name,
    provider=str(series_key.provider),
    symbol=str(series_key.symbol),
  )
  if not prediction_runs:
    return {}

  metadata: dict[str, Any] = {
    "stacked_prediction_runs": prediction_runs,
  }
  hybrid_transformer_artifact_dir = prediction_runs.get("hybrid_transformer", "")
  if hybrid_transformer_artifact_dir:
    metadata["hybrid_transformer_artifact_dir"] = hybrid_transformer_artifact_dir
  return metadata


def resolve_stacked_prediction_runs(
  input_root: Path,
  *,
  dataset_name: str,
  provider: str,
  symbol: str,
) -> dict[str, str]:
  dataset_root = input_root / "training_datasets" / dataset_name / provider / symbol
  if not dataset_root.exists():
    return {}

  alias_counters: dict[str, Counter[str]] = {}
  for manifest_path in sorted(dataset_root.rglob("*.manifest.json")):
    try:
      payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except JSONDecodeError:
      continue
    for alias, run_dir in payload.get("prediction_runs", {}).items():
      alias_name = str(alias).strip()
      run_path = str(run_dir).strip()
      if not alias_name or not run_path:
        continue
      alias_counters.setdefault(alias_name, Counter())[run_path] += 1

  resolved_runs: dict[str, str] = {}
  for alias, counter in alias_counters.items():
    ranked_runs = sorted(counter.items(), key=lambda item: (item[1], item[0]))
    resolved_runs[alias] = ranked_runs[-1][0]
  return resolved_runs


def split_meta_training_datasets(
  *,
  model_name: str,
  task_type: str,
  target_column: str,
  train_dataset: TabularDataset,
  calibration_method: str,
  calibration_fraction: float,
  minimum_calibration_rows: int,
  calibration_model_names: tuple[str, ...],
) -> tuple[TabularDataset, TabularDataset | None]:
  if (
    not calibration_method
    or model_name not in calibration_model_names
    or not is_supported_ternary_classification(task_type=task_type, target_column=target_column)
    or len(train_dataset.rows) < max(2 * minimum_calibration_rows, 32)
  ):
    return train_dataset, None

  if calibration_fraction <= 0 or calibration_fraction >= 0.5:
    raise ValueError("probability_calibration_fraction must satisfy 0 < fraction < 0.5.")
  if minimum_calibration_rows < 1:
    raise ValueError("minimum_calibration_rows must be at least 1.")

  calibration_rows = max(minimum_calibration_rows, int(round(len(train_dataset.rows) * calibration_fraction)))
  calibration_rows = min(calibration_rows, len(train_dataset.rows) - minimum_calibration_rows)
  if calibration_rows < minimum_calibration_rows:
    return train_dataset, None

  split_index = len(train_dataset.rows) - calibration_rows
  fit_indices = list(range(split_index))
  calibration_indices = list(range(split_index, len(train_dataset.rows)))
  fit_dataset = subset_tabular_dataset(train_dataset, fit_indices)
  calibration_dataset = subset_tabular_dataset(train_dataset, calibration_indices)
  if len(set(int(target) for target in calibration_dataset.targets)) < 2:
    return train_dataset, None
  if len(fit_dataset.rows) < minimum_calibration_rows:
    return train_dataset, None
  return fit_dataset, calibration_dataset


def build_meta_sample_weights(
  *,
  rows: list[dict[str, str]],
  targets: list[float | int],
  scheme: str,
  trend_column: str,
  trend_threshold_bps: float,
  regime_column: str,
  regime_change_column: str,
  regime_change_threshold: float,
  regime_change_boost: float,
  floor: float,
  cap: float,
) -> list[float]:
  if scheme not in VALID_SAMPLE_WEIGHT_SCHEMES:
    raise ValueError(f"Unknown sample weight scheme: {scheme}")
  if not scheme or not rows or not targets:
    return []
  if len(rows) != len(targets):
    raise ValueError("rows and targets must have the same length for sample weight construction.")
  if floor <= 0 or cap <= 0 or floor > cap:
    raise ValueError("sample weight bounds must satisfy 0 < floor <= cap.")
  if regime_change_boost <= 0:
    raise ValueError("regime_change_boost must be positive.")

  threshold = trend_threshold_bps / 10_000
  buckets: list[str] = []
  for row, target in zip(rows, targets):
    if scheme.startswith("hmm_regime_"):
      regime = _hmm_regime_bucket(row.get(regime_column, ""))
    else:
      regime = _trend_bucket(row.get(trend_column, ""), threshold)
    if scheme in {"trend_regime_balance", "hmm_regime_balance"}:
      bucket = regime
    else:
      bucket = f"{regime}:{_target_bucket(target)}"
    buckets.append(bucket)

  counts = Counter(buckets)
  if not counts:
    return []

  bucket_count = len(counts)
  total_rows = len(buckets)
  raw_weights = [total_rows / (bucket_count * counts[bucket]) for bucket in buckets]
  if regime_change_boost != 1.0:
    raw_weights = [
      weight * (
        regime_change_boost
        if _is_regime_transition_row(
          row,
          regime_change_column=regime_change_column,
          threshold=regime_change_threshold,
        )
        else 1.0
      )
      for weight, row in zip(raw_weights, rows)
    ]
  mean_weight = sum(raw_weights) / len(raw_weights)
  if mean_weight <= 0:
    return [1.0 for _ in raw_weights]
  return [
    max(floor, min(cap, weight / mean_weight))
    for weight in raw_weights
  ]


def _trend_bucket(value: str, threshold: float) -> str:
  if value == "":
    return "missing"
  trend_value = float(value)
  if trend_value >= threshold:
    return "uptrend"
  if trend_value <= -threshold:
    return "downtrend"
  return "range"


def _target_bucket(target: float | int) -> str:
  numeric = float(target)
  if numeric > 0:
    return "positive"
  if numeric < 0:
    return "negative"
  return "flat"


def _hmm_regime_bucket(value: str) -> str:
  if value == "":
    return "missing"
  try:
    state_index = int(float(value))
  except ValueError:
    return "missing"
  if state_index <= 0:
    return "bear"
  if state_index >= 2:
    return "bull"
  return "chop"


def _is_regime_transition_row(
  row: dict[str, str],
  *,
  regime_change_column: str,
  threshold: float,
) -> bool:
  value = row.get(regime_change_column, "")
  if value == "":
    return False
  try:
    return float(value) >= threshold
  except ValueError:
    return False

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant.models.base import utc_run_slug
from quant.models.transformers.hybrid_forecaster import (
  HybridQuantileTransformer,
  HybridTransformerConfig,
  auxiliary_prediction_column,
  build_auxiliary_prediction_columns,
  build_prediction_columns,
  build_target_columns,
)
from quant.training.datasets import (
  build_multi_target_sequence_dataset,
  discover_series,
  load_series_rows_for_dataset,
  subset_multi_target_sequence_dataset,
)
from quant.training.train_hybrid_transformer import _build_sample_weights, _sample_weight_target_column
from quant.validation.leakage import effective_purge_rows
from quant.validation.metrics import task_metrics


@dataclass(frozen=True, slots=True)
class HybridTransformerWalkForwardConfig:
  input_root: Path = Path("data/features")
  output_root: Path = Path("data/models")
  providers: tuple[str, ...] = ()
  symbols: tuple[str, ...] = ()
  dataset_name: str = "market_supervised"
  target_columns: tuple[str, ...] = ()
  auxiliary_target_columns: tuple[str, ...] = ()
  feature_groups: tuple[str, ...] = ()
  exclude_feature_groups: tuple[str, ...] = ()
  feature_prefixes: tuple[str, ...] = ()
  exclude_feature_prefixes: tuple[str, ...] = ()
  include_feature_columns: tuple[str, ...] = ()
  exclude_feature_columns: tuple[str, ...] = ()
  initial_train_rows: int = 512
  validation_rows: int = 128
  step_rows: int | None = None
  embargo_rows: int = 0
  purge_rows: int | None = None
  minimum_rows: int = 768
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
class HybridWalkForwardSummary:
  runs_attempted: int = 0
  runs_completed: int = 0
  folds_completed: int = 0
  artifacts_written: int = 0


def purged_expanding_window_splits(
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
  if embargo_rows < 0 or purge_rows < 0:
    raise ValueError("embargo_rows and purge_rows cannot be negative.")

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


def evaluate_hybrid_transformer_walk_forward(
  config: HybridTransformerWalkForwardConfig,
) -> HybridWalkForwardSummary:
  summary = HybridWalkForwardSummary()

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
    purge_rows = effective_purge_rows(config.purge_rows, target_columns)
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

    required_rows = max(
      config.minimum_rows,
      config.initial_train_rows + config.embargo_rows + config.validation_rows,
    )
    artifact_root = _artifact_root(
      config.output_root,
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
      splits = purged_expanding_window_splits(
        len(dataset.rows),
        initial_train_rows=config.initial_train_rows,
        validation_rows=config.validation_rows,
        step_rows=config.step_rows,
        embargo_rows=config.embargo_rows,
        purge_rows=purge_rows,
      )
      folds_attempted = len(splits)
      for fold_index, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_dataset = subset_multi_target_sequence_dataset(dataset, train_indices)
        validation_dataset = subset_multi_target_sequence_dataset(dataset, validation_indices)
        if not train_dataset.rows or not validation_dataset.rows:
          folds_skipped += 1
          continue

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
        output_rows = model.predict_output_rows(validation_dataset.sequences)
        fold_rows.append(
          _fold_metadata(
            fold_index=fold_index,
            train_rows=train_dataset.rows,
            validation_rows=validation_dataset.rows,
            metrics=metrics,
          )
        )
        prediction_rows.extend(
          _prediction_rows(
            model_name=model.model_name,
            fold_index=fold_index,
            rows=validation_dataset.rows,
            target_columns=train_dataset.target_columns,
            targets=validation_dataset.targets,
            auxiliary_target_columns=train_dataset.auxiliary_target_columns,
            auxiliary_targets=validation_dataset.auxiliary_targets,
            predictions=output_rows,
            quantiles=config.transformer.quantiles,
            auxiliary_prediction_columns=build_auxiliary_prediction_columns(
              model.auxiliary_target_columns,
              model.auxiliary_class_values,
            ),
          )
        )
      if not fold_rows and skip_reason == "":
        skip_reason = "no_usable_folds"

    summary.artifacts_written += _write_artifacts(
      artifact_root=artifact_root,
      provider=series_key.provider,
      symbol=series_key.symbol,
      feature_columns=dataset.feature_names,
      target_columns=target_columns,
      auxiliary_target_columns=list(config.auxiliary_target_columns),
      total_rows=len(dataset.rows),
      prediction_rows=prediction_rows,
      fold_rows=fold_rows,
      folds_attempted=folds_attempted,
      folds_skipped=folds_skipped,
      quantiles=config.transformer.quantiles,
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
        "dataset_name": config.dataset_name,
        "target_columns": target_columns,
        "auxiliary_target_columns": config.auxiliary_target_columns,
        "feature_groups": config.feature_groups,
        "exclude_feature_groups": config.exclude_feature_groups,
        "feature_prefixes": config.feature_prefixes,
        "exclude_feature_prefixes": config.exclude_feature_prefixes,
        "include_feature_columns": config.include_feature_columns,
        "exclude_feature_columns": config.exclude_feature_columns,
        "sample_weight_scheme": config.sample_weight_scheme,
        "sample_weight_target_column": _sample_weight_target_column(config, dataset),
        "regime_change_boost": config.regime_change_boost,
        "transformer": asdict(config.transformer),
      },
      config_path=config.config_path,
      run_status=run_status if fold_rows else "skipped",
      skip_reason=skip_reason if not fold_rows else "",
    )
    if fold_rows:
      summary.runs_completed += 1
      summary.folds_completed += len(fold_rows)

  return summary


def _artifact_root(output_root: Path, *, provider: str, symbol: str) -> Path:
  root = output_root / "walk_forward" / "hybrid_transformers" / "quantile_forecaster" / provider / symbol / utc_run_slug()
  root.mkdir(parents=True, exist_ok=True)
  return root


def _fold_metadata(
  *,
  fold_index: int,
  train_rows: list[dict[str, str]],
  validation_rows: list[dict[str, str]],
  metrics: dict[str, float],
) -> dict[str, Any]:
  row: dict[str, Any] = {
    "fold_index": fold_index,
    "task_type": "regression",
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
  fold_index: int,
  rows: list[dict[str, str]],
  target_columns: list[str],
  targets: list[list[float]],
  auxiliary_target_columns: list[str],
  auxiliary_targets: list[list[int]],
  predictions: list[dict[str, float]],
  quantiles: tuple[float, ...],
  auxiliary_prediction_columns: list[str],
) -> list[dict[str, Any]]:
  output: list[dict[str, Any]] = []
  prediction_columns = [
    *build_prediction_columns(target_columns, quantiles),
    *auxiliary_prediction_columns,
  ]
  for row, target_values, auxiliary_target_values, prediction_row in zip(rows, targets, auxiliary_targets, predictions):
    payload: dict[str, Any] = {
      "model_name": model_name,
      "task_type": "regression",
      "fold_index": fold_index,
      "provider": row.get("provider", ""),
      "symbol": row.get("symbol", ""),
      "candle_start": row.get("candle_start", ""),
    }
    for target_column, target_value in zip(target_columns, target_values):
      payload[f"target_{target_column}"] = target_value
    for target_column, target_value in zip(auxiliary_target_columns, auxiliary_target_values):
      payload[f"target_{target_column}"] = target_value
    for prediction_column in prediction_columns:
      payload[prediction_column] = prediction_row.get(prediction_column, "")
    output.append(payload)
  return output


def _write_artifacts(
  *,
  artifact_root: Path,
  provider: str,
  symbol: str,
  feature_columns: list[str],
  target_columns: list[str],
  auxiliary_target_columns: list[str],
  total_rows: int,
  prediction_rows: list[dict[str, Any]],
  fold_rows: list[dict[str, Any]],
  folds_attempted: int,
  folds_skipped: int,
  quantiles: tuple[float, ...],
  window_config: dict[str, Any],
  run_config: dict[str, Any],
  config_path: Path | None,
  run_status: str,
  skip_reason: str,
) -> int:
  prediction_columns = [
    *build_prediction_columns(target_columns, quantiles),
    *build_auxiliary_prediction_columns(auxiliary_target_columns),
  ]
  fold_metrics_path = artifact_root / "fold_metrics.csv"
  predictions_path = artifact_root / "validation_predictions.csv"
  summary_path = artifact_root / "summary.json"
  run_config_path = artifact_root / "run_config.json"

  fold_fieldnames = sorted({key for row in fold_rows for key in row.keys()}) or [
    "fold_index",
    "task_type",
    "train_rows",
    "validation_rows",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
  ]
  prediction_fieldnames = [
    "model_name",
    "task_type",
    "fold_index",
    "provider",
    "symbol",
    "candle_start",
    *[f"target_{column}" for column in target_columns],
    *[f"target_{column}" for column in auxiliary_target_columns],
    *prediction_columns,
  ]
  _write_csv(fold_metrics_path, fold_rows, fieldnames=fold_fieldnames)
  _write_csv(predictions_path, prediction_rows, fieldnames=prediction_fieldnames)

  aggregate_metrics = _aggregate_prediction_metrics(
    prediction_rows=prediction_rows,
    target_columns=target_columns,
    auxiliary_target_columns=auxiliary_target_columns,
    quantiles=quantiles,
  )
  aggregate_metrics.update(_mean_fold_metrics(fold_rows))
  summary_payload = {
    "family": "hybrid_transformers",
    "model_name": "hybrid_quantile_transformer",
    "task_type": "regression",
    "provider": provider,
    "symbol": symbol,
    "target_columns": target_columns,
    "auxiliary_target_columns": auxiliary_target_columns,
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
  run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
  return 4


def _aggregate_prediction_metrics(
  *,
  prediction_rows: list[dict[str, Any]],
  target_columns: list[str],
  auxiliary_target_columns: list[str],
  quantiles: tuple[float, ...],
) -> dict[str, float]:
  if not prediction_rows:
    return {}
  metrics: dict[str, float] = {}
  median_label = _median_quantile_label(quantiles)
  for target_column in target_columns:
    target_values = [float(row[f"target_{target_column}"]) for row in prediction_rows]
    prediction_values = [
      float(row[f"prediction_return_q{median_label}_{target_column.removeprefix('forward_return_')}"])
      for row in prediction_rows
    ]
    metrics.update(
      task_metrics(
        "regression",
        target_values,
        prediction_values,
        prefix=f"overall_validation_{target_column}_",
      )
    )
  if target_columns:
    primary_target = target_columns[0]
    metrics.update(
      task_metrics(
        "regression",
        [float(row[f"target_{primary_target}"]) for row in prediction_rows],
        [
          float(row[f"prediction_return_q{median_label}_{primary_target.removeprefix('forward_return_')}"])
          for row in prediction_rows
        ],
        prefix="overall_validation_",
      )
    )
  for target_column in auxiliary_target_columns:
    prediction_column = auxiliary_prediction_column(target_column)
    rows = [
      row
      for row in prediction_rows
      if row.get(f"target_{target_column}", "") != "" and row.get(prediction_column, "") != ""
    ]
    if not rows:
      continue
    metrics.update(
      task_metrics(
        "classification",
        [int(float(row[f"target_{target_column}"])) for row in rows],
        [float(row[prediction_column]) for row in rows],
        prefix=f"overall_validation_{target_column}_",
      )
    )
  return metrics


def _median_quantile_label(quantiles: tuple[float, ...]) -> str:
  if 0.5 in quantiles:
    return "50"
  return f"{int(round(quantiles[len(quantiles) // 2] * 100)):02d}"


def _mean_fold_metrics(fold_rows: list[dict[str, Any]]) -> dict[str, float]:
  metric_names = sorted(
    {
      key
      for row in fold_rows
      for key in row.keys()
      if key.startswith("validation_")
      and key not in {"validation_start", "validation_end", "validation_rows"}
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

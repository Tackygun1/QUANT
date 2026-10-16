from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.execution.paper_trading import (
  META_MODEL_NAMES,
  load_model_from_artifact,
  predict_live_artifact,
  resolve_live_feature_values,
)
from quant.features.market_features import FeatureBuildConfig, build_market_features
from quant.ingestion.coinbase_historical import (
  CoinbaseHistoricalBackfillConfig,
  interval_to_seconds,
  run_coinbase_historical_backfill,
)
from quant.ingestion.staging import StageConfig, run_stager
from quant.labels.market_labels import LabelBuildConfig, build_market_labels
from quant.models.base import TrainingArtifactMetadata
from quant.risk.sizing import PositionSizingConfig
from quant.training.datasets import (
  build_sequence_dataset,
  build_tabular_dataset,
  load_series_rows_for_dataset,
)
from quant.training.supervised_dataset import SupervisedDatasetConfig, build_supervised_datasets
from quant.training.train_tabular import write_validation_predictions
from quant.validation.metrics import task_metrics

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrozenHoldoutConfig:
  source_workspace_root: Path
  artifact_dirs: tuple[Path, ...]
  holdout_root: Path
  provider: str = "coinbase"
  symbol: str = "BTC-USD"
  candle_interval: str = "5m"
  evaluation_start_time: datetime | None = None
  evaluation_end_time: datetime | None = None
  warmup_bars: int = 64
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.02
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  use_artifact_execution_sizing: bool = True
  sizing: PositionSizingConfig = PositionSizingConfig(
    signal_threshold_bps=0.5,
    signal_scale_bps=5.0,
  )
  clean_start: bool = True
  config_path: Path | None = None


@dataclass(slots=True)
class FrozenHoldoutSummary:
  holdout_root: Path
  evaluation_start_time: datetime
  evaluation_end_time: datetime
  evaluation_rows: int = 0
  results: dict[str, dict[str, Any]] = field(default_factory=dict)


def run_frozen_holdout_evaluation(config: FrozenHoldoutConfig) -> FrozenHoldoutSummary:
  holdout_root = config.holdout_root
  if config.clean_start and holdout_root.exists():
    shutil.rmtree(holdout_root)
  holdout_root.mkdir(parents=True, exist_ok=True)

  evaluation_start, evaluation_end = resolve_evaluation_window(config)
  build_holdout_dataset(
    config=config,
    evaluation_start=evaluation_start,
    evaluation_end=evaluation_end,
  )

  rows = load_series_rows_for_dataset(
    holdout_root / "features",
    _series_key(config.provider, config.symbol),
    dataset_name="market_supervised",
  )
  evaluation_rows = sum(
    1
    for row in rows
    if evaluation_start <= _parse_dt(row.get("candle_start", "")) < evaluation_end
  )

  results: dict[str, dict[str, Any]] = {}
  for artifact_dir in config.artifact_dirs:
    result = evaluate_frozen_artifact(
      config=config,
      artifact_dir=artifact_dir,
      evaluation_start=evaluation_start,
      evaluation_end=evaluation_end,
    )
    results[result["model_name"]] = result

  summary = FrozenHoldoutSummary(
    holdout_root=holdout_root,
    evaluation_start_time=evaluation_start,
    evaluation_end_time=evaluation_end,
    evaluation_rows=evaluation_rows,
    results=results,
  )
  (holdout_root / "state").mkdir(parents=True, exist_ok=True)
  (holdout_root / "state" / "holdout_summary.json").write_text(
    json.dumps(
      {
        "holdout_root": str(summary.holdout_root),
        "evaluation_start_time": _format_dt(summary.evaluation_start_time),
        "evaluation_end_time": _format_dt(summary.evaluation_end_time),
        "evaluation_rows": summary.evaluation_rows,
        "results": summary.results,
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return summary


def resolve_evaluation_window(config: FrozenHoldoutConfig) -> tuple[datetime, datetime]:
  state_path = config.source_workspace_root / "state" / "overnight_state.json"
  state = json.loads(state_path.read_text(encoding="utf-8"))
  source_end = _parse_dt(state["dataset_window"]["end_time"])
  evaluation_start = config.evaluation_start_time or align_to_next_bar(source_end, config.candle_interval)
  evaluation_end = config.evaluation_end_time or align_to_bar(datetime.now(UTC), config.candle_interval)
  if evaluation_end <= evaluation_start:
    raise ValueError(
      f"Evaluation end {evaluation_end} must be after evaluation start {evaluation_start}."
    )
  return evaluation_start, evaluation_end


def build_holdout_dataset(
  *,
  config: FrozenHoldoutConfig,
  evaluation_start: datetime,
  evaluation_end: datetime,
) -> None:
  warmup_start = evaluation_start - timedelta(seconds=interval_to_seconds(config.candle_interval) * config.warmup_bars)
  raw_root = config.holdout_root / "raw"
  staging_root = config.holdout_root / "staging"
  feature_root = config.holdout_root / "features"

  candle_summary = run_coinbase_historical_backfill(
    CoinbaseHistoricalBackfillConfig(
      products=(config.symbol,),
      channels=("candles",),
      start_time=warmup_start,
      end_time=evaluation_end,
      output_root=raw_root,
      candle_interval=config.candle_interval,
      request_timeout_seconds=config.request_timeout_seconds,
      request_pause_seconds=config.request_pause_seconds,
    )
  )
  trade_summary = run_coinbase_historical_backfill(
    CoinbaseHistoricalBackfillConfig(
      products=(config.symbol,),
      channels=("trades",),
      start_time=warmup_start,
      end_time=evaluation_end,
      output_root=raw_root,
      candle_interval=config.candle_interval,
      request_timeout_seconds=config.request_timeout_seconds,
      request_pause_seconds=config.request_pause_seconds,
    )
  )
  stage_summary = run_stager(
    StageConfig(
      input_root=raw_root,
      output_root=staging_root,
      providers=(config.provider,),
      symbols=(config.symbol,),
    )
  )
  feature_summary = build_market_features(
    FeatureBuildConfig(
      input_root=staging_root,
      output_root=feature_root,
      providers=(config.provider,),
      symbols=(config.symbol,),
    )
  )
  label_summary = build_market_labels(
    LabelBuildConfig(
      input_root=feature_root,
      output_root=feature_root,
      providers=(config.provider,),
      symbols=(config.symbol,),
    )
  )
  supervised_summary = build_supervised_datasets(
    SupervisedDatasetConfig(
      input_root=feature_root,
      output_root=feature_root,
      providers=(config.provider,),
      symbols=(config.symbol,),
    )
  )

  dataset_summary = {
    "warmup_start": _format_dt(warmup_start),
    "evaluation_start": _format_dt(evaluation_start),
    "evaluation_end": _format_dt(evaluation_end),
    "candle_backfill": asdict(candle_summary),
    "trade_backfill": asdict(trade_summary),
    "staging": {
      "files_processed": stage_summary.files_processed,
      "output_files_written": stage_summary.output_files_written,
      "rows_by_table": dict(stage_summary.rows_by_table),
    },
    "features": asdict(feature_summary),
    "labels": asdict(label_summary),
    "supervised": asdict(supervised_summary),
  }
  (config.holdout_root / "state").mkdir(parents=True, exist_ok=True)
  (config.holdout_root / "state" / "dataset_summary.json").write_text(
    json.dumps(dataset_summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )


def evaluate_frozen_artifact(
  *,
  config: FrozenHoldoutConfig,
  artifact_dir: Path,
  evaluation_start: datetime,
  evaluation_end: datetime,
) -> dict[str, Any]:
  source_metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
  model = load_model_from_artifact(artifact_dir)
  evaluation_sizing, execution_sizing_source = _resolve_artifact_execution_sizing(
    source_metadata=source_metadata,
    base_sizing=config.sizing,
    use_artifact_execution_sizing=config.use_artifact_execution_sizing,
  )
  rows = load_series_rows_for_dataset(
    config.holdout_root / "features",
    _series_key(config.provider, config.symbol),
    dataset_name="market_supervised",
  )
  prediction_rows, targets, predictions = build_frozen_predictions(
    rows=rows,
    source_metadata=source_metadata,
    model=model,
    evaluation_start=evaluation_start,
    evaluation_end=evaluation_end,
    artifact_dir=artifact_dir,
  )
  metrics = task_metrics(source_metadata["task_type"], targets, predictions, prefix="test_")

  evaluation_artifact_dir = (
    config.holdout_root
    / "evaluations"
    / source_metadata["model_name"]
    / artifact_dir.name
  )
  evaluation_artifact_dir.mkdir(parents=True, exist_ok=True)

  write_validation_predictions(
    evaluation_artifact_dir / "validation_predictions.csv",
    prediction_rows,
    targets,
    predictions,
  )
  TrainingArtifactMetadata(
    model_name=source_metadata["model_name"],
    task_type=source_metadata["task_type"],
    target_column=source_metadata["target_column"],
    provider=source_metadata["provider"],
    symbol=source_metadata["symbol"],
    feature_columns=tuple(source_metadata.get("feature_columns", ())),
    train_rows=0,
    validation_rows=len(prediction_rows),
    metrics=metrics,
    extra={
      "source_artifact_dir": str(artifact_dir),
      "source_metrics": source_metadata.get("metrics", {}),
      "evaluation_type": "untouched_holdout",
      "evaluation_start_time": _format_dt(evaluation_start),
      "evaluation_end_time": _format_dt(evaluation_end),
      "config_path": str(config.config_path) if config.config_path else "",
      "use_artifact_execution_sizing": config.use_artifact_execution_sizing,
      "execution_sizing_source": execution_sizing_source,
      "execution_sizing": asdict(evaluation_sizing),
    },
  ).write_json(evaluation_artifact_dir / "metadata.json")

  backtest_summary = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=evaluation_artifact_dir,
      input_root=config.holdout_root / "features",
      output_root=config.holdout_root / "reports" / "backtests",
      dataset_name="market_supervised",
      fee_bps=config.fee_bps,
      slippage_bps=config.slippage_bps,
      sizing=evaluation_sizing,
    )
  )
  result = {
    "model_name": source_metadata["model_name"],
    "source_artifact_dir": str(artifact_dir),
    "evaluation_artifact_dir": str(evaluation_artifact_dir),
    "evaluation_start_time": _format_dt(evaluation_start),
    "evaluation_end_time": _format_dt(evaluation_end),
    "evaluation_rows": len(prediction_rows),
    "use_artifact_execution_sizing": config.use_artifact_execution_sizing,
    "execution_sizing_source": execution_sizing_source,
    "execution_sizing": asdict(evaluation_sizing),
    "metrics": metrics,
    "backtest": {
      "joined_rows": backtest_summary.joined_rows,
      "active_periods": backtest_summary.active_periods,
      "trades": backtest_summary.trades,
      "total_net_return": backtest_summary.total_net_return,
      "max_drawdown": backtest_summary.max_drawdown,
      "sharpe_like": backtest_summary.sharpe_like,
      "total_transaction_cost": backtest_summary.total_transaction_cost,
      "total_turnover": backtest_summary.total_turnover,
      "active_hit_rate": backtest_summary.active_hit_rate,
      "average_net_return_per_trade": backtest_summary.average_net_return_per_trade,
      "profit_factor_after_cost": backtest_summary.profit_factor_after_cost,
      "after_cost_trade_quality": backtest_summary.after_cost_trade_quality,
      "output_dir": str(backtest_summary.output_dir) if backtest_summary.output_dir else "",
    },
  }
  (evaluation_artifact_dir / "frozen_result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def build_frozen_predictions(
  *,
  rows: list[dict[str, str]],
  source_metadata: dict[str, Any],
  model: Any,
  evaluation_start: datetime,
  evaluation_end: datetime,
  artifact_dir: Path | None = None,
) -> tuple[list[dict[str, str]], list[float | int], list[float]]:
  task_type = str(source_metadata["task_type"])
  target_column = str(source_metadata["target_column"])
  feature_columns = list(source_metadata.get("feature_columns", ()))
  model_name = str(source_metadata["model_name"])

  if model_name == "transformer_return_forecaster":
    lookback = int(getattr(getattr(model, "config", None), "lookback", 0))
    dataset = build_sequence_dataset(
      rows,
      target_column=target_column,
      lookback=lookback,
      feature_columns=feature_columns,
    )
    selected_rows: list[dict[str, str]] = []
    selected_sequences: list[list[list[float]]] = []
    selected_targets: list[float] = []
    for row, sequence, target in zip(dataset.rows, dataset.sequences, dataset.targets):
      candle_dt = _parse_dt(row.get("candle_start", ""))
      if evaluation_start <= candle_dt < evaluation_end:
        selected_rows.append(row)
        selected_sequences.append(sequence)
        selected_targets.append(target)
    predictions = model.predict(selected_sequences) if selected_sequences else []
    return selected_rows, selected_targets, predictions

  if model_name in META_MODEL_NAMES:
    if artifact_dir is None:
      raise ValueError("artifact_dir is required to score meta-models on frozen holdouts.")
    selected_rows: list[dict[str, str]] = []
    selected_targets: list[float | int] = []
    predictions: list[float] = []
    supporting_artifacts: dict[str, Any] = {}
    closed_rows = [row for row in rows if row.get("is_closed") == "true"]
    for index, row in enumerate(closed_rows):
      candle_dt = _parse_dt(row.get("candle_start", ""))
      if not (evaluation_start <= candle_dt < evaluation_end):
        continue
      target_value = row.get(target_column, "")
      if target_value == "":
        continue
      history = closed_rows[: index + 1]
      scoring_row, _scoring_context = resolve_live_feature_values(
        row=row,
        feature_rows=history,
        metadata=source_metadata,
        artifact_dir=artifact_dir,
        artifact_search_root=None,
        supporting_artifacts=supporting_artifacts,
      )
      prediction = float(
        predict_live_artifact(
          model=model,
          metadata=source_metadata,
          row=scoring_row,
          feature_rows=history,
        )
      )
      selected_rows.append(row)
      selected_targets.append(int(target_value) if task_type == "classification" else float(target_value))
      predictions.append(prediction)
    return selected_rows, selected_targets, predictions

  dataset = build_tabular_dataset(
    rows,
    target_column=target_column,
    feature_columns=feature_columns,
    classification=task_type == "classification",
  )
  selected_rows: list[dict[str, str]] = []
  selected_features: list[list[float]] = []
  selected_targets: list[float | int] = []
  for row, features, target in zip(dataset.rows, dataset.features, dataset.targets):
    candle_dt = _parse_dt(row.get("candle_start", ""))
    if evaluation_start <= candle_dt < evaluation_end:
      selected_rows.append(row)
      selected_features.append(features)
      selected_targets.append(target)
  predictions = model.predict(selected_features) if selected_features else []
  return selected_rows, selected_targets, predictions


def align_to_bar(value: datetime, interval: str) -> datetime:
  seconds = interval_to_seconds(interval)
  timestamp = int(value.astimezone(UTC).timestamp())
  floored = timestamp - (timestamp % seconds)
  return datetime.fromtimestamp(floored, tz=UTC)


def align_to_next_bar(value: datetime, interval: str) -> datetime:
  seconds = interval_to_seconds(interval)
  timestamp = int(value.astimezone(UTC).timestamp())
  remainder = timestamp % seconds
  aligned = timestamp if remainder == 0 else timestamp + (seconds - remainder)
  return datetime.fromtimestamp(aligned, tz=UTC)


def _resolve_artifact_execution_sizing(
  *,
  source_metadata: dict[str, Any],
  base_sizing: PositionSizingConfig,
  use_artifact_execution_sizing: bool,
) -> tuple[PositionSizingConfig, str]:
  if not use_artifact_execution_sizing:
    return base_sizing, "config"

  extra = source_metadata.get("extra")
  if not isinstance(extra, dict):
    return base_sizing, "config"

  execution_payload = extra.get("execution_sizing")
  execution_source = "artifact_execution_sizing"
  if not isinstance(execution_payload, dict):
    execution_payload = extra.get("execution_params")
    execution_source = "artifact_execution_params"
  if not isinstance(execution_payload, dict):
    return base_sizing, "config"

  payload = asdict(base_sizing)
  changed = False
  for key, value in execution_payload.items():
    if key in payload:
      payload[key] = value
      changed = True
  if not changed:
    return base_sizing, "config"
  return PositionSizingConfig(**payload), execution_source


def _series_key(provider: str, symbol: str) -> Any:
  from quant.training.datasets import SeriesKey

  return SeriesKey(provider=provider, symbol=symbol)


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_dt(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

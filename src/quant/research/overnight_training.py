from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, ModelBacktestSummary, run_model_backtest
from quant.features.market_features import FeatureBuildConfig, build_market_features
from quant.ingestion.coinbase_historical import (
  CoinbaseHistoricalBackfillConfig,
  run_coinbase_historical_backfill,
)
from quant.ingestion.staging import StageConfig, run_stager
from quant.labels.market_labels import LabelBuildConfig, build_market_labels
from quant.models.base import TrainingArtifactMetadata, utc_run_slug
from quant.models.ensembles.stacked_meta import STACKED_META_MODEL_REGISTRY
from quant.models.registry import TABULAR_MODEL_REGISTRY
from quant.models.transformers.hybrid_forecaster import (
  HybridQuantileTransformer,
  HybridTransformerConfig,
  build_prediction_columns,
  build_target_columns,
)
from quant.models.transformers.return_forecaster import (
  ReturnSequenceTransformer,
  TransformerForecasterConfig,
)
from quant.risk.sizing import PositionSizingConfig
from quant.training.datasets import (
  SeriesKey,
  build_multi_target_sequence_dataset,
  build_sequence_dataset,
  build_tabular_dataset,
  chronological_split,
  load_series_rows_for_dataset,
  subset_multi_target_sequence_dataset,
  subset_sequence_dataset,
  subset_tabular_dataset,
)
from quant.training.stacked_dataset import (
  PredictionSource,
  StackedDatasetConfig,
  build_stacked_datasets,
)
from quant.training.supervised_dataset import DatasetKey, SupervisedDatasetConfig, build_supervised_datasets, dataset_file_path, write_rows
from quant.training.train_meta import build_meta_sample_weights
from quant.training.train_tabular import (
  build_metrics,
  build_prediction_output_metadata,
  build_prediction_outputs,
  write_validation_predictions,
)
from quant.validation.hybrid_walk_forward import (
  HybridTransformerWalkForwardConfig,
  evaluate_hybrid_transformer_walk_forward,
)

LOGGER = logging.getLogger(__name__)

HYBRID_META_MODEL_NAMES = {
  "lightgbm_hybrid_return_meta",
  "lightgbm_hybrid_direction_meta",
  "lightgbm_hybrid_tradeability_meta",
}


@dataclass(frozen=True, slots=True)
class OvernightDataConfig:
  product: str = "BTC-USD"
  provider: str = "coinbase"
  candle_interval: str = "5m"
  candle_start_time: datetime | None = None
  trade_start_time: datetime | None = None
  end_time: datetime | None = None
  candle_lookback_days: int = 90
  trade_lookback_hours: int = 72
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.02


@dataclass(frozen=True, slots=True)
class EvolutionSearchConfig:
  duration_hours: float = 8.0
  validation_fraction: float = 0.2
  minimum_rows: int = 512
  transformer_recent_rows: int = 2048
  hybrid_recent_rows: int = 16384
  hybrid_horizons: tuple[int, ...] = (1, 6, 24, 96)
  hybrid_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
  hybrid_meta_validation_fraction: float = 0.2
  hybrid_output_dataset_name: str = "market_hybrid_stacked"
  hybrid_initial_train_rows: int = 2048
  hybrid_validation_rows: int = 512
  hybrid_step_rows: int = 512
  hybrid_embargo_rows: int = 1
  hybrid_purge_rows: int = 96
  hybrid_min_trade_count: int = 25
  hybrid_inactivity_penalty: float = 0.01
  initial_random_candidates: int = 3
  elite_count: int = 2
  mutation_power: float = 0.35
  retry_attempts: int = 3
  retry_pause_seconds: float = 20.0
  seed: int = 7
  model_cycle: tuple[str, ...] = (
    "histgb_return",
    "ridge_return",
    "lightgbm_hybrid_return_meta",
    "lightgbm_hybrid_direction_meta",
  )
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  sizing: PositionSizingConfig = PositionSizingConfig(
    signal_threshold_bps=0.5,
    signal_scale_bps=5.0,
  )


@dataclass(frozen=True, slots=True)
class OvernightTrainingConfig:
  workspace_root: Path
  data: OvernightDataConfig = OvernightDataConfig()
  search: EvolutionSearchConfig = EvolutionSearchConfig()
  clean_start: bool = False
  config_path: Path | None = None


@dataclass(slots=True)
class OvernightTrainingSummary:
  workspace_root: Path
  deadline: datetime
  dataset_rows: int = 0
  candidates_completed: int = 0
  best_results: dict[str, dict[str, Any]] = field(default_factory=dict)
  state_path: Path | None = None


def run_overnight_training(config: OvernightTrainingConfig) -> OvernightTrainingSummary:
  workspace = config.workspace_root
  workspace.mkdir(parents=True, exist_ok=True)
  (workspace / "logs").mkdir(parents=True, exist_ok=True)
  state_path = workspace / "state" / "overnight_state.json"
  state_path.parent.mkdir(parents=True, exist_ok=True)

  if config.clean_start:
    _clean_workspace(workspace)

  state = _load_state(state_path)
  started_at = _parse_dt_optional(state.get("started_at")) or datetime.now(UTC)
  deadline = _parse_dt_optional(state.get("deadline")) or (
    started_at + timedelta(hours=config.search.duration_hours)
  )
  state.setdefault("started_at", _format_dt(started_at))
  state.setdefault("deadline", _format_dt(deadline))
  state.setdefault("status", "running")
  state.setdefault("workspace_root", str(workspace))
  state.setdefault("config_path", str(config.config_path) if config.config_path else "")
  _write_state(state_path, state)

  build_dataset_with_retries(config, state_path)
  rows = load_series_rows_for_dataset(
    workspace / "features",
    series_key=SeriesKey(provider=config.data.provider, symbol=config.data.product),
    dataset_name="market_supervised",
  )
  if len(rows) < config.search.minimum_rows:
    raise RuntimeError(
      f"Overnight dataset is too small for training: {len(rows)} rows < {config.search.minimum_rows}"
    )

  completed = run_evolutionary_search(
    config=config,
    rows=rows,
    deadline=deadline,
    state_path=state_path,
  )
  candidates = _load_candidate_records(workspace / "search" / "candidates.jsonl")
  best_results = _best_results_by_model(candidates)

  final_state = _load_state(state_path)
  final_state["status"] = "completed"
  final_state["completed_at"] = _format_dt(datetime.now(UTC))
  final_state["dataset_rows"] = len(rows)
  final_state["candidates_completed"] = completed
  final_state["best_results"] = best_results
  _write_state(state_path, final_state)

  summary = OvernightTrainingSummary(
    workspace_root=workspace,
    deadline=deadline,
    dataset_rows=len(rows),
    candidates_completed=completed,
    best_results=best_results,
    state_path=state_path,
  )
  (workspace / "state" / "overnight_summary.json").write_text(
    json.dumps(
      {
        "workspace_root": str(summary.workspace_root),
        "deadline": _format_dt(summary.deadline),
        "dataset_rows": summary.dataset_rows,
        "candidates_completed": summary.candidates_completed,
        "best_results": summary.best_results,
        "state_path": str(summary.state_path) if summary.state_path else "",
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return summary


def build_dataset_with_retries(config: OvernightTrainingConfig, state_path: Path) -> None:
  state = _load_state(state_path)
  if state.get("dataset_status") == "completed":
    return

  attempts = 0
  while attempts < config.search.retry_attempts:
    attempts += 1
    try:
      build_large_dataset(config, state_path)
      return
    except Exception as exc:
      LOGGER.exception("Dataset build attempt %s failed: %s", attempts, exc)
      state = _load_state(state_path)
      state["dataset_status"] = "retrying"
      state["dataset_attempts"] = attempts
      state["dataset_last_error"] = str(exc)
      _write_state(state_path, state)
      if attempts >= config.search.retry_attempts:
        raise
      time.sleep(config.search.retry_pause_seconds)


def build_large_dataset(config: OvernightTrainingConfig, state_path: Path) -> None:
  workspace = config.workspace_root
  data_config = _materialize_data_times(config)

  raw_root = workspace / "raw"
  staging_root = workspace / "staging"
  feature_root = workspace / "features"
  for path in (raw_root, staging_root, feature_root):
    if path.exists():
      shutil.rmtree(path)

  state = _load_state(state_path)
  state["dataset_status"] = "building"
  state["dataset_window"] = {
    "candle_start_time": _format_dt(data_config.candle_start_time),
    "trade_start_time": _format_dt(data_config.trade_start_time),
    "end_time": _format_dt(data_config.end_time),
  }
  _write_state(state_path, state)

  candle_summary = run_coinbase_historical_backfill(
    CoinbaseHistoricalBackfillConfig(
      products=(data_config.product,),
      channels=("candles",),
      start_time=data_config.candle_start_time,
      end_time=data_config.end_time,
      output_root=raw_root,
      candle_interval=data_config.candle_interval,
      request_timeout_seconds=data_config.request_timeout_seconds,
      request_pause_seconds=data_config.request_pause_seconds,
    )
  )
  trade_summary = run_coinbase_historical_backfill(
    CoinbaseHistoricalBackfillConfig(
      products=(data_config.product,),
      channels=("trades",),
      start_time=data_config.trade_start_time,
      end_time=data_config.end_time,
      output_root=raw_root,
      candle_interval=data_config.candle_interval,
      request_timeout_seconds=data_config.request_timeout_seconds,
      request_pause_seconds=data_config.request_pause_seconds,
    )
  )
  stage_summary = run_stager(
    StageConfig(
      input_root=raw_root,
      output_root=staging_root,
      providers=(data_config.provider,),
      symbols=(data_config.product,),
    )
  )
  feature_summary = build_market_features(
    FeatureBuildConfig(
      input_root=staging_root,
      output_root=feature_root,
      providers=(data_config.provider,),
      symbols=(data_config.product,),
    )
  )
  label_summary = build_market_labels(
    LabelBuildConfig(
      input_root=feature_root,
      output_root=feature_root,
      providers=(data_config.provider,),
      symbols=(data_config.product,),
      horizons=config.search.hybrid_horizons,
    )
  )
  supervised_summary = build_supervised_datasets(
    SupervisedDatasetConfig(
      input_root=feature_root,
      output_root=feature_root,
      providers=(data_config.provider,),
      symbols=(data_config.product,),
    )
  )

  rows = load_series_rows_for_dataset(
    feature_root,
    series_key=SeriesKey(provider=data_config.provider, symbol=data_config.product),
    dataset_name="market_supervised",
  )
  dataset_summary = {
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
    "supervised_rows": len(rows),
  }

  state = _load_state(state_path)
  state["dataset_status"] = "completed"
  state["dataset_completed_at"] = _format_dt(datetime.now(UTC))
  state["dataset_summary"] = dataset_summary
  _write_state(state_path, state)
  (workspace / "state" / "dataset_summary.json").write_text(
    json.dumps(dataset_summary, indent=2, sort_keys=True),
    encoding="utf-8",
  )


def run_evolutionary_search(
  *,
  config: OvernightTrainingConfig,
  rows: list[dict[str, str]],
  deadline: datetime,
  state_path: Path,
) -> int:
  search_root = config.workspace_root / "search"
  search_root.mkdir(parents=True, exist_ok=True)
  candidates_path = search_root / "candidates.jsonl"
  rng = random.Random(config.search.seed)
  existing_records = _load_candidate_records(candidates_path)
  for _ in range(len(existing_records)):
    rng.random()

  completed = len([record for record in existing_records if record.get("status") == "completed"])
  candidate_index = len(existing_records)

  while datetime.now(UTC) < deadline:
    for model_name in config.search.model_cycle:
      if datetime.now(UTC) >= deadline:
        break
      candidate_index += 1
      record = propose_candidate(
        model_name=model_name,
        existing_records=existing_records,
        candidate_index=candidate_index,
        rng=rng,
        mutation_power=config.search.mutation_power,
        initial_random_candidates=config.search.initial_random_candidates,
        elite_count=config.search.elite_count,
      )
      _update_search_state(state_path, model_name, candidate_index, deadline, status="running")
      try:
        result = evaluate_candidate(
          config=config,
          rows=rows,
          candidate=record,
        )
        LOGGER.info(
          "Candidate complete model=%s candidate_id=%s fitness=%.6f net_return=%.6f max_drawdown=%.6f trades=%s params=%s",
          model_name,
          record["candidate_id"],
          float(result.get("fitness", 0.0)),
          float(result.get("backtest", {}).get("total_net_return", 0.0)),
          float(result.get("backtest", {}).get("max_drawdown", 0.0)),
          result.get("backtest", {}).get("trades", 0),
          result.get("params", {}),
        )
      except Exception as exc:
        LOGGER.exception("Candidate %s failed: %s", record["candidate_id"], exc)
        result = {
          **record,
          "status": "failed",
          "error": str(exc),
          "completed_at": _format_dt(datetime.now(UTC)),
        }

      _append_jsonl(candidates_path, result)
      existing_records.append(result)
      if result.get("status") == "completed":
        completed += 1
      _write_best_results(config.workspace_root, existing_records)
      _update_search_state(state_path, model_name, candidate_index, deadline, status="running")

  return completed


def propose_candidate(
  *,
  model_name: str,
  existing_records: list[dict[str, Any]],
  candidate_index: int,
  rng: random.Random,
  mutation_power: float,
  initial_random_candidates: int,
  elite_count: int,
) -> dict[str, Any]:
  model_records = [
    record
    for record in existing_records
    if record.get("model_name") == model_name and record.get("status") == "completed"
  ]
  candidate_id = f"{candidate_index:05d}_{utc_run_slug()}"

  if len(model_records) < initial_random_candidates:
    params = random_candidate_params(model_name, rng)
    execution_params = random_candidate_execution_params(model_name, rng)
    origin = "random"
    parent_id = ""
  else:
    sorted_records = sorted(model_records, key=_candidate_sort_key, reverse=True)
    elites = sorted_records[: max(1, elite_count)]
    parent = rng.choice(elites)
    if rng.random() < 0.2:
      params = random_candidate_params(model_name, rng)
      execution_params = random_candidate_execution_params(model_name, rng)
      origin = "explore"
      parent_id = parent["candidate_id"]
    else:
      params = mutate_candidate_params(
        model_name=model_name,
        parent_params=parent["params"],
        rng=rng,
        mutation_power=mutation_power,
      )
      execution_params = mutate_candidate_execution_params(
        model_name=model_name,
        parent_params=dict(parent.get("execution_params", {})),
        rng=rng,
      )
      origin = "mutate"
      parent_id = parent["candidate_id"]

  return {
    "candidate_id": candidate_id,
    "model_name": model_name,
    "params": params,
    "execution_params": execution_params,
    "origin": origin,
    "parent_id": parent_id,
    "started_at": _format_dt(datetime.now(UTC)),
  }


def evaluate_candidate(
  *,
  config: OvernightTrainingConfig,
  rows: list[dict[str, str]],
  candidate: dict[str, Any],
) -> dict[str, Any]:
  model_name = str(candidate["model_name"])
  candidate_sizing = resolve_candidate_execution_sizing(
    base_sizing=config.search.sizing,
    execution_params=dict(candidate.get("execution_params", {})),
  )
  artifact_dir = (
    config.workspace_root
    / "models"
    / "evolution"
    / model_name
    / str(candidate["candidate_id"])
  )
  artifact_dir.mkdir(parents=True, exist_ok=True)

  backtest_input_root = config.workspace_root / "features"
  backtest_dataset_name: str | None = None
  if model_name in HYBRID_META_MODEL_NAMES:
    result = _evaluate_hybrid_meta_candidate(
      config=config,
      rows=rows,
      candidate=candidate,
      artifact_dir=artifact_dir,
      candidate_sizing=candidate_sizing,
    )
    backtest_input_root = artifact_dir / "pipeline" / "features"
    backtest_dataset_name = config.search.hybrid_output_dataset_name
  elif model_name == "transformer_return_forecaster":
    result = _evaluate_transformer_candidate(
      config=config,
      rows=rows,
      candidate=candidate,
      artifact_dir=artifact_dir,
      candidate_sizing=candidate_sizing,
    )
  else:
    result = _evaluate_tabular_candidate(
      config=config,
      rows=rows,
      candidate=candidate,
      artifact_dir=artifact_dir,
      candidate_sizing=candidate_sizing,
    )

  backtest_summary = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=artifact_dir,
      input_root=backtest_input_root,
      output_root=config.workspace_root / "reports" / "backtests",
      dataset_name=backtest_dataset_name,
      fee_bps=config.search.fee_bps,
      slippage_bps=config.search.slippage_bps,
      sizing=candidate_sizing,
    )
  )
  fitness = candidate_fitness(
    model_name=model_name,
    search_config=config.search,
    backtest_summary=backtest_summary,
  )

  result.update(
    {
      "status": "completed",
      "completed_at": _format_dt(datetime.now(UTC)),
      "artifact_dir": str(artifact_dir),
      "backtest_output_dir": str(backtest_summary.output_dir) if backtest_summary.output_dir else "",
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
      },
      "execution_params": dict(candidate.get("execution_params", {})),
      "execution_sizing": asdict(candidate_sizing),
      "fitness": fitness,
    }
  )
  (artifact_dir / "candidate_result.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return result


def _evaluate_tabular_candidate(
  *,
  config: OvernightTrainingConfig,
  rows: list[dict[str, str]],
  candidate: dict[str, Any],
  artifact_dir: Path,
  candidate_sizing: PositionSizingConfig,
) -> dict[str, Any]:
  model_name = str(candidate["model_name"])
  params = dict(candidate["params"])
  model_spec = TABULAR_MODEL_REGISTRY[model_name]
  dataset = build_tabular_dataset(
    rows,
    target_column=model_spec["target_column"],
    classification=model_spec["task_type"] == "classification",
  )
  if len(dataset.rows) < config.search.minimum_rows:
    raise RuntimeError(f"Not enough rows for {model_name}: {len(dataset.rows)}")

  train_indices, validation_indices = chronological_split(len(dataset.rows), config.search.validation_fraction)
  train_dataset = subset_tabular_dataset(dataset, train_indices)
  validation_dataset = subset_tabular_dataset(dataset, validation_indices)
  model = model_spec["builder"](params)
  model.fit(train_dataset.features, train_dataset.targets)
  train_predictions = model.predict(train_dataset.features)
  validation_predictions = model.predict(validation_dataset.features)
  validation_prediction_outputs = build_prediction_outputs(
    task_type=model_spec["task_type"],
    target_column=model_spec["target_column"],
    model=model,
    features=validation_dataset.features,
    predictions=validation_predictions,
  )
  metrics = build_metrics(
    task_type=model_spec["task_type"],
    train_targets=train_dataset.targets,
    train_predictions=train_predictions,
    validation_targets=validation_dataset.targets,
    validation_predictions=validation_predictions,
  )

  model.save(artifact_dir / "model.pkl")
  TrainingArtifactMetadata(
    model_name=model_name,
    task_type=model_spec["task_type"],
    target_column=model_spec["target_column"],
    provider=config.data.provider,
    symbol=config.data.product,
    feature_columns=tuple(dataset.feature_names),
    train_rows=len(train_dataset.rows),
    validation_rows=len(validation_dataset.rows),
    metrics=metrics,
    extra={
      "candidate_id": candidate["candidate_id"],
      "origin": candidate["origin"],
      "parent_id": candidate["parent_id"],
      "params": params,
      "execution_params": dict(candidate.get("execution_params", {})),
      "execution_sizing": asdict(candidate_sizing),
      "config_path": str(config.config_path) if config.config_path else "",
      "dataset_name": "market_supervised",
    },
  ).write_json(artifact_dir / "metadata.json")
  write_validation_predictions(
    artifact_dir / "validation_predictions.csv",
    validation_dataset.rows,
    validation_dataset.targets,
    validation_predictions,
    prediction_outputs=validation_prediction_outputs,
  )
  (artifact_dir / "training_config.json").write_text(
    json.dumps(
      {
        "validation_fraction": config.search.validation_fraction,
        "minimum_rows": config.search.minimum_rows,
        "params": params,
        "execution_params": dict(candidate.get("execution_params", {})),
        "execution_sizing": asdict(candidate_sizing),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return {
    **candidate,
    "task_type": model_spec["task_type"],
    "target_column": model_spec["target_column"],
    "metrics": metrics,
    "train_rows": len(train_dataset.rows),
    "validation_rows": len(validation_dataset.rows),
  }


def _evaluate_transformer_candidate(
  *,
  config: OvernightTrainingConfig,
  rows: list[dict[str, str]],
  candidate: dict[str, Any],
  artifact_dir: Path,
  candidate_sizing: PositionSizingConfig,
) -> dict[str, Any]:
  params = dict(candidate["params"])
  trimmed_rows = rows[-config.search.transformer_recent_rows :] if len(rows) > config.search.transformer_recent_rows else rows
  transformer_config = TransformerForecasterConfig(
    lookback=int(params["lookback"]),
    d_model=int(params["d_model"]),
    nhead=int(params["nhead"]),
    num_layers=int(params["num_layers"]),
    dim_feedforward=int(params["dim_feedforward"]),
    dropout=float(params["dropout"]),
    batch_size=int(params["batch_size"]),
    epochs=int(params["epochs"]),
    learning_rate=float(params["learning_rate"]),
    weight_decay=float(params["weight_decay"]),
    validation_fraction=config.search.validation_fraction,
    device=str(params.get("device", "cpu")),
  )
  dataset = build_sequence_dataset(
    trimmed_rows,
    target_column="forward_return_1_bar",
    lookback=transformer_config.lookback,
  )
  if len(dataset.rows) < config.search.minimum_rows:
    raise RuntimeError(
      f"Not enough rows for transformer_return_forecaster: {len(dataset.rows)}"
    )

  train_indices, validation_indices = chronological_split(len(dataset.rows), config.search.validation_fraction)
  train_dataset = subset_sequence_dataset(dataset, train_indices)
  validation_dataset = subset_sequence_dataset(dataset, validation_indices)
  model = ReturnSequenceTransformer(transformer_config)
  model.fit(
    train_dataset.sequences,
    train_dataset.targets,
    feature_names=train_dataset.feature_names,
    validation_sequences=validation_dataset.sequences,
    validation_targets=validation_dataset.targets,
  )
  train_predictions = model.predict(train_dataset.sequences)
  validation_predictions = model.predict(validation_dataset.sequences)
  metrics = build_metrics(
    task_type="regression",
    train_targets=train_dataset.targets,
    train_predictions=train_predictions,
    validation_targets=validation_dataset.targets,
    validation_predictions=validation_predictions,
  )

  model.save(artifact_dir / "model.pt")
  TrainingArtifactMetadata(
    model_name=model.model_name,
    task_type=model.task_type,
    target_column="forward_return_1_bar",
    provider=config.data.provider,
    symbol=config.data.product,
    feature_columns=tuple(dataset.feature_names),
    train_rows=len(train_dataset.rows),
    validation_rows=len(validation_dataset.rows),
    metrics=metrics,
    extra={
      "candidate_id": candidate["candidate_id"],
      "origin": candidate["origin"],
      "parent_id": candidate["parent_id"],
      "params": params,
      "execution_params": dict(candidate.get("execution_params", {})),
      "execution_sizing": asdict(candidate_sizing),
      "config_path": str(config.config_path) if config.config_path else "",
      "transformer_recent_rows": config.search.transformer_recent_rows,
      "dataset_name": "market_supervised",
    },
  ).write_json(artifact_dir / "metadata.json")
  _write_transformer_predictions(
    artifact_dir / "validation_predictions.csv",
    validation_dataset.rows,
    validation_dataset.targets,
    validation_predictions,
  )
  (artifact_dir / "training_config.json").write_text(
    json.dumps(
      {
        "validation_fraction": config.search.validation_fraction,
        "minimum_rows": config.search.minimum_rows,
        "transformer_recent_rows": config.search.transformer_recent_rows,
        "params": params,
        "execution_params": dict(candidate.get("execution_params", {})),
        "execution_sizing": asdict(candidate_sizing),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return {
    **candidate,
    "task_type": "regression",
    "target_column": "forward_return_1_bar",
    "metrics": metrics,
    "train_rows": len(train_dataset.rows),
    "validation_rows": len(validation_dataset.rows),
    "sequence_rows_used": len(trimmed_rows),
  }


def _evaluate_hybrid_meta_candidate(
  *,
  config: OvernightTrainingConfig,
  rows: list[dict[str, str]],
  candidate: dict[str, Any],
  artifact_dir: Path,
  candidate_sizing: PositionSizingConfig,
) -> dict[str, Any]:
  model_name = str(candidate["model_name"])
  params = dict(candidate["params"])
  transformer_params = dict(params.get("transformer", {}))
  meta_params = dict(params.get("meta", {}))
  trimmed_rows = rows[-config.search.hybrid_recent_rows :] if len(rows) > config.search.hybrid_recent_rows else rows
  if len(trimmed_rows) < config.search.minimum_rows:
    raise RuntimeError(f"Not enough rows for {model_name}: {len(trimmed_rows)}")

  provider = config.data.provider
  symbol = config.data.product
  pipeline_root = artifact_dir / "pipeline"
  feature_root = pipeline_root / "features"
  model_root = pipeline_root / "models"
  _write_candidate_dataset_slice(
    feature_root,
    dataset_name="market_supervised",
    provider=provider,
    symbol=symbol,
    rows=trimmed_rows,
  )

  transformer_config = _hybrid_transformer_config(
    transformer_params,
    validation_fraction=config.search.validation_fraction,
    horizons=config.search.hybrid_horizons,
    quantiles=config.search.hybrid_quantiles,
  )
  target_columns = tuple(build_target_columns(transformer_config.horizons))
  final_transformer = _train_final_hybrid_transformer(
    config=config,
    candidate=candidate,
    rows=trimmed_rows,
    artifact_dir=artifact_dir / "hybrid_transformer",
    transformer_config=transformer_config,
    target_columns=target_columns,
  )

  required_hybrid_rows = max(
    config.search.minimum_rows,
    config.search.hybrid_initial_train_rows
    + config.search.hybrid_embargo_rows
    + config.search.hybrid_validation_rows,
  )
  walk_forward_summary = evaluate_hybrid_transformer_walk_forward(
    HybridTransformerWalkForwardConfig(
      input_root=feature_root,
      output_root=model_root,
      providers=(provider,),
      symbols=(symbol,),
      dataset_name="market_supervised",
      target_columns=target_columns,
      initial_train_rows=config.search.hybrid_initial_train_rows,
      validation_rows=config.search.hybrid_validation_rows,
      step_rows=config.search.hybrid_step_rows,
      embargo_rows=config.search.hybrid_embargo_rows,
      purge_rows=config.search.hybrid_purge_rows,
      minimum_rows=required_hybrid_rows,
      config_path=config.config_path,
      transformer=transformer_config,
    )
  )
  if walk_forward_summary.runs_completed < 1:
    raise RuntimeError("Hybrid walk-forward search produced no completed runs.")

  prediction_columns = tuple(build_prediction_columns(list(target_columns), transformer_config.quantiles))
  stacked_summary = build_stacked_datasets(
    StackedDatasetConfig(
      input_root=feature_root,
      prediction_root=model_root / "walk_forward",
      output_root=feature_root,
      input_dataset_name="market_supervised",
      output_dataset_name=config.search.hybrid_output_dataset_name,
      providers=(provider,),
      symbols=(symbol,),
      prediction_sources=(
        PredictionSource(
          family="hybrid_transformers",
          model_name="quantile_forecaster",
          alias="hybrid_transformer",
          source_columns=prediction_columns,
        ),
      ),
    )
  )
  if stacked_summary.stacked_rows_written < config.search.minimum_rows:
    raise RuntimeError(
      "Hybrid stacked dataset is too small for meta-model training: "
      f"{stacked_summary.stacked_rows_written} rows < {config.search.minimum_rows}"
    )

  stacked_rows = load_series_rows_for_dataset(
    feature_root,
    SeriesKey(provider=provider, symbol=symbol),
    dataset_name=config.search.hybrid_output_dataset_name,
  )
  model_spec = STACKED_META_MODEL_REGISTRY[model_name]
  task_type = model_spec["task_type"]
  target_column = model_spec["target_column"]
  sample_weight_scheme = str(
    meta_params.pop(
      "sample_weight_scheme",
      "hmm_regime_target_balance" if task_type == "classification" else "hmm_regime_balance",
    )
  )
  regime_change_boost = float(meta_params.pop("regime_change_boost", 1.25))
  regime_change_threshold = float(meta_params.pop("regime_change_threshold", 0.35))
  dataset = build_tabular_dataset(
    stacked_rows,
    target_column=target_column,
    classification=task_type == "classification",
  )
  if len(dataset.rows) < config.search.minimum_rows:
    raise RuntimeError(
      f"Not enough stacked rows for {model_name}: {len(dataset.rows)}"
    )

  train_indices, validation_indices = chronological_split(
    len(dataset.rows),
    config.search.hybrid_meta_validation_fraction,
  )
  train_dataset = subset_tabular_dataset(dataset, train_indices)
  validation_dataset = subset_tabular_dataset(dataset, validation_indices)
  if task_type == "classification" and len(set(train_dataset.targets)) < 2:
    raise RuntimeError(f"{model_name} needs at least two classes in the training split.")

  sample_weights = build_meta_sample_weights(
    rows=train_dataset.rows,
    targets=train_dataset.targets,
    scheme=sample_weight_scheme,
    trend_column="trend_return_24_bar",
    trend_threshold_bps=20.0,
    regime_column="regime_hmm_state_index",
    regime_change_column="regime_bocpd_change_prob",
    regime_change_threshold=regime_change_threshold,
    regime_change_boost=regime_change_boost,
    floor=0.5,
    cap=3.0,
  )
  model = model_spec["builder"](meta_params)
  model.fit(
    train_dataset.features,
    train_dataset.targets,
    sample_weights=sample_weights or None,
  )
  train_predictions = model.predict(train_dataset.features)
  validation_predictions = model.predict(validation_dataset.features)
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
  metrics = build_metrics(
    task_type=task_type,
    train_targets=train_dataset.targets,
    train_predictions=train_predictions,
    validation_targets=validation_dataset.targets,
    validation_predictions=validation_predictions,
  )

  model.save(artifact_dir / "model.pkl")
  TrainingArtifactMetadata(
    model_name=model_name,
    task_type=task_type,
    target_column=target_column,
    provider=provider,
    symbol=symbol,
    feature_columns=tuple(dataset.feature_names),
    train_rows=len(train_dataset.rows),
    validation_rows=len(validation_dataset.rows),
    metrics=metrics,
    extra={
      "candidate_id": candidate["candidate_id"],
      "origin": candidate["origin"],
      "parent_id": candidate["parent_id"],
      "params": params,
      "execution_params": dict(candidate.get("execution_params", {})),
      "execution_sizing": asdict(candidate_sizing),
      "config_path": str(config.config_path) if config.config_path else "",
      "dataset_name": config.search.hybrid_output_dataset_name,
      "transformer_params": transformer_params,
      "meta_params": meta_params,
      "sample_weight_scheme": sample_weight_scheme,
      "regime_change_boost": regime_change_boost,
      "regime_change_threshold": regime_change_threshold,
      "hybrid_prediction_columns": prediction_columns,
      "hybrid_target_columns": list(target_columns),
      "hybrid_transformer_artifact_dir": str(final_transformer["artifact_dir"]),
      "hybrid_walk_forward_root": str(model_root / "walk_forward"),
      "stacked_rows_written": stacked_summary.stacked_rows_written,
      "walk_forward_folds_completed": walk_forward_summary.folds_completed,
      **prediction_metadata,
    },
  ).write_json(artifact_dir / "metadata.json")
  write_validation_predictions(
    artifact_dir / "validation_predictions.csv",
    validation_dataset.rows,
    validation_dataset.targets,
    validation_predictions,
    prediction_outputs=validation_prediction_outputs,
  )
  (artifact_dir / "training_config.json").write_text(
    json.dumps(
      {
        "validation_fraction": config.search.hybrid_meta_validation_fraction,
        "minimum_rows": config.search.minimum_rows,
        "hybrid_recent_rows": config.search.hybrid_recent_rows,
        "hybrid_horizons": list(config.search.hybrid_horizons),
        "hybrid_quantiles": list(config.search.hybrid_quantiles),
        "hybrid_initial_train_rows": config.search.hybrid_initial_train_rows,
        "hybrid_validation_rows": config.search.hybrid_validation_rows,
        "hybrid_step_rows": config.search.hybrid_step_rows,
        "hybrid_embargo_rows": config.search.hybrid_embargo_rows,
        "hybrid_purge_rows": config.search.hybrid_purge_rows,
        "output_dataset_name": config.search.hybrid_output_dataset_name,
        "params": params,
        "sample_weight_scheme": sample_weight_scheme,
        "regime_change_boost": regime_change_boost,
        "regime_change_threshold": regime_change_threshold,
        "execution_params": dict(candidate.get("execution_params", {})),
        "execution_sizing": asdict(candidate_sizing),
        "final_transformer": {
          "artifact_dir": str(final_transformer["artifact_dir"]),
          "train_rows": final_transformer["train_rows"],
          "validation_rows": final_transformer["validation_rows"],
          "metrics": final_transformer["metrics"],
        },
        "walk_forward_summary": asdict(walk_forward_summary),
        "stacked_summary": asdict(stacked_summary),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return {
    **candidate,
    "task_type": task_type,
    "target_column": target_column,
    "metrics": metrics,
    "train_rows": len(train_dataset.rows),
    "validation_rows": len(validation_dataset.rows),
    "sequence_rows_used": len(trimmed_rows),
    "hybrid_walk_forward": asdict(walk_forward_summary),
    "hybrid_stacked_dataset": asdict(stacked_summary),
    "final_transformer": {
      "artifact_dir": str(final_transformer["artifact_dir"]),
      "train_rows": final_transformer["train_rows"],
      "validation_rows": final_transformer["validation_rows"],
      "metrics": final_transformer["metrics"],
    },
  }


def _train_final_hybrid_transformer(
  *,
  config: OvernightTrainingConfig,
  candidate: dict[str, Any],
  rows: list[dict[str, str]],
  artifact_dir: Path,
  transformer_config: HybridTransformerConfig,
  target_columns: tuple[str, ...],
) -> dict[str, Any]:
  dataset = build_multi_target_sequence_dataset(
    rows,
    target_columns=target_columns,
    lookback=transformer_config.lookback,
  )
  if len(dataset.rows) < config.search.minimum_rows:
    raise RuntimeError(
      "Not enough rows for hybrid transformer training: "
      f"{len(dataset.rows)} rows < {config.search.minimum_rows}"
    )

  train_indices, validation_indices = chronological_split(
    len(dataset.rows),
    transformer_config.validation_fraction,
  )
  train_dataset = subset_multi_target_sequence_dataset(dataset, train_indices)
  validation_dataset = subset_multi_target_sequence_dataset(dataset, validation_indices)
  model = HybridQuantileTransformer(transformer_config)
  metrics = model.fit(
    train_dataset.sequences,
    train_dataset.targets,
    feature_names=train_dataset.feature_names,
    target_columns=train_dataset.target_columns,
    validation_sequences=validation_dataset.sequences,
    validation_targets=validation_dataset.targets,
  )

  artifact_dir.mkdir(parents=True, exist_ok=True)
  model.save(artifact_dir / "model.pt")
  prediction_columns = build_prediction_columns(train_dataset.target_columns, transformer_config.quantiles)
  TrainingArtifactMetadata(
    model_name=model.model_name,
    task_type=model.task_type,
    target_column=train_dataset.target_columns[0],
    provider=config.data.provider,
    symbol=config.data.product,
    feature_columns=tuple(dataset.feature_names),
    train_rows=len(train_dataset.rows),
    validation_rows=len(validation_dataset.rows),
    metrics=metrics,
    extra={
      "candidate_id": candidate["candidate_id"],
      "origin": candidate["origin"],
      "parent_id": candidate["parent_id"],
      "params": candidate["params"].get("transformer", {}),
      "config_path": str(config.config_path) if config.config_path else "",
      "dataset_name": "market_supervised",
      "target_columns": list(train_dataset.target_columns),
      "prediction_columns": prediction_columns,
      "primary_prediction_column": model.primary_prediction_column,
    },
  ).write_json(artifact_dir / "metadata.json")
  _write_hybrid_transformer_predictions(
    artifact_dir / "validation_predictions.csv",
    rows=validation_dataset.rows,
    targets=validation_dataset.targets,
    predictions=model.predict_output_rows(validation_dataset.sequences),
    target_columns=train_dataset.target_columns,
    prediction_columns=prediction_columns,
  )
  (artifact_dir / "training_config.json").write_text(
    json.dumps(
      {
        "validation_fraction": transformer_config.validation_fraction,
        "minimum_rows": config.search.minimum_rows,
        "params": candidate["params"].get("transformer", {}),
        "target_columns": list(train_dataset.target_columns),
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  return {
    "artifact_dir": artifact_dir,
    "train_rows": len(train_dataset.rows),
    "validation_rows": len(validation_dataset.rows),
    "metrics": metrics,
  }


def random_candidate_params(model_name: str, rng: random.Random) -> dict[str, Any]:
  if model_name == "ridge_return":
    return {"alpha": _log_uniform(rng, 0.01, 100.0)}
  if model_name == "histgb_return":
    return {
      "max_depth": rng.choice((3, 4, 5, 6, 8, None)),
      "learning_rate": _log_uniform(rng, 0.01, 0.2),
      "max_iter": rng.randrange(100, 801, 50),
    }
  if model_name == "transformer_return_forecaster":
    d_model = rng.choice((32, 64, 96))
    nhead = 4 if d_model % 4 == 0 else 2
    return {
      "lookback": rng.choice((16, 24, 32, 48, 64)),
      "d_model": d_model,
      "nhead": nhead,
      "num_layers": rng.choice((1, 2, 3)),
      "dim_feedforward": rng.choice((64, 128, 192, 256)),
      "dropout": round(rng.uniform(0.0, 0.25), 4),
      "batch_size": rng.choice((16, 32, 64)),
      "epochs": rng.choice((6, 8, 10, 12, 16)),
      "learning_rate": _log_uniform(rng, 1e-4, 3e-3),
      "weight_decay": _log_uniform(rng, 1e-6, 2e-3),
      "device": "cpu",
    }
  if model_name in HYBRID_META_MODEL_NAMES:
    return {
      "transformer": _random_hybrid_transformer_params(rng),
      "meta": _random_hybrid_meta_params(rng),
    }
  raise ValueError(f"Unsupported model for overnight search: {model_name}")


def random_candidate_execution_params(model_name: str, rng: random.Random) -> dict[str, Any]:
  return {
    key: rng.choice(choices)
    for key, choices in _execution_search_space(model_name).items()
  }


def mutate_candidate_params(
  *,
  model_name: str,
  parent_params: dict[str, Any],
  rng: random.Random,
  mutation_power: float,
) -> dict[str, Any]:
  params = dict(parent_params)
  if model_name == "ridge_return":
    params["alpha"] = _mutate_log_float(float(params["alpha"]), 0.01, 100.0, mutation_power, rng)
    return params
  if model_name == "histgb_return":
    params["learning_rate"] = _mutate_log_float(float(params["learning_rate"]), 0.01, 0.2, mutation_power, rng)
    params["max_iter"] = _mutate_int(int(params["max_iter"]), 100, 800, 50, mutation_power, rng)
    params["max_depth"] = rng.choice((params.get("max_depth"), 3, 4, 5, 6, 8, None))
    return params
  if model_name == "transformer_return_forecaster":
    params["lookback"] = _mutate_choice(int(params["lookback"]), (16, 24, 32, 48, 64), rng)
    params["d_model"] = _mutate_choice(int(params["d_model"]), (32, 64, 96), rng)
    params["nhead"] = 4 if int(params["d_model"]) % 4 == 0 else 2
    params["num_layers"] = _mutate_choice(int(params["num_layers"]), (1, 2, 3), rng)
    params["dim_feedforward"] = _mutate_choice(int(params["dim_feedforward"]), (64, 128, 192, 256), rng)
    params["dropout"] = round(_mutate_linear_float(float(params["dropout"]), 0.0, 0.3, mutation_power, rng), 4)
    params["batch_size"] = _mutate_choice(int(params["batch_size"]), (16, 32, 64), rng)
    params["epochs"] = _mutate_choice(int(params["epochs"]), (6, 8, 10, 12, 16), rng)
    params["learning_rate"] = _mutate_log_float(float(params["learning_rate"]), 1e-4, 3e-3, mutation_power, rng)
    params["weight_decay"] = _mutate_log_float(float(params["weight_decay"]), 1e-6, 2e-3, mutation_power, rng)
    params["device"] = "cpu"
    return params
  if model_name in HYBRID_META_MODEL_NAMES:
    return {
      "transformer": _mutate_hybrid_transformer_params(
        parent_params=dict(parent_params.get("transformer", {})),
        rng=rng,
        mutation_power=mutation_power,
      ),
      "meta": _mutate_hybrid_meta_params(
        parent_params=dict(parent_params.get("meta", {})),
        rng=rng,
        mutation_power=mutation_power,
      ),
    }
  raise ValueError(f"Unsupported model for mutation: {model_name}")


def mutate_candidate_execution_params(
  *,
  model_name: str,
  parent_params: dict[str, Any],
  rng: random.Random,
) -> dict[str, Any]:
  params = dict(parent_params or random_candidate_execution_params(model_name, rng))
  for key, choices in _execution_search_space(model_name).items():
    current = params.get(key, choices[0])
    params[key] = _mutate_choice(current, choices, rng)
  return params


def resolve_candidate_execution_sizing(
  *,
  base_sizing: PositionSizingConfig,
  execution_params: dict[str, Any],
) -> PositionSizingConfig:
  payload = asdict(base_sizing)
  for key, value in execution_params.items():
    if key in payload:
      payload[key] = value
  return PositionSizingConfig(**payload)


def candidate_fitness(
  *,
  model_name: str,
  search_config: EvolutionSearchConfig,
  backtest_summary: ModelBacktestSummary,
) -> float:
  fitness = float(
    getattr(
      backtest_summary,
      "after_cost_trade_quality",
      float(backtest_summary.total_net_return) - float(backtest_summary.max_drawdown),
    )
  )
  if model_name not in HYBRID_META_MODEL_NAMES:
    return fitness

  minimum_trade_count = max(0, int(search_config.hybrid_min_trade_count))
  if minimum_trade_count <= 0:
    return fitness
  if backtest_summary.trades >= minimum_trade_count:
    return fitness

  inactivity_ratio = (minimum_trade_count - backtest_summary.trades) / minimum_trade_count
  penalty = max(0.0, float(search_config.hybrid_inactivity_penalty)) * inactivity_ratio
  return fitness - penalty


def _execution_search_space(model_name: str) -> dict[str, tuple[float | int, ...]]:
  if model_name == "lightgbm_hybrid_direction_meta":
    return {
      "classification_margin_threshold": (0.05, 0.1, 0.15, 0.2, 0.3),
      "classification_margin_scale": (0.2, 0.35, 0.5, 0.75),
      "classification_hold_margin_threshold": (0.05, 0.1, 0.15, 0.2),
      "classification_flip_margin_threshold": (0.15, 0.25, 0.35, 0.5),
      "class_change_persistence_bars": (0, 1, 2, 3),
      "min_hold_bars": (0, 1, 2, 3),
      "min_rebalance_abs_delta": (0.0, 0.01, 0.025, 0.05, 0.1),
    }

  if model_name in HYBRID_META_MODEL_NAMES:
    return {
      "signal_threshold_bps": (0.0, 0.25, 0.5, 1.0),
      "signal_scale_bps": (2.5, 5.0, 7.5, 10.0),
      "hold_signal_threshold_bps": (0.0, 0.5, 1.0, 2.0, 3.0),
      "flip_signal_threshold_bps": (0.0, 2.5, 5.0, 10.0),
      "min_hold_bars": (0, 1, 2, 3),
      "min_rebalance_abs_delta": (0.0, 0.01, 0.025, 0.05, 0.1),
    }

  return {
    "hold_signal_threshold_bps": (0.0, 1.0, 2.0, 3.0, 5.0),
    "flip_signal_threshold_bps": (0.0, 5.0, 10.0, 15.0),
    "min_hold_bars": (0, 1, 3, 5),
    "min_rebalance_abs_delta": (0.0, 0.05, 0.1, 0.15, 0.2),
  }


def _random_hybrid_transformer_params(rng: random.Random) -> dict[str, Any]:
  d_model = rng.choice((32, 64))
  return {
    "lookback": rng.choice((24, 32, 48, 64)),
    "backbone": "timemixer",
    "d_model": d_model,
    "nhead": 4 if d_model % 4 == 0 else 2,
    "num_layers": rng.choice((1, 2)),
    "dim_feedforward": rng.choice((64, 128, 192)),
    "scales": rng.choice(((1, 2, 4), (1, 2, 4, 8), (1, 3, 6))),
    "mixer_layers": rng.choice((1, 2, 3)),
    "dropout": round(rng.uniform(0.0, 0.2), 4),
    "batch_size": rng.choice((16, 32, 64)),
    "epochs": rng.choice((3, 4, 6)),
    "learning_rate": _log_uniform(rng, 2e-4, 2e-3),
    "weight_decay": _log_uniform(rng, 1e-6, 1e-3),
    "device": "cpu",
  }


def _random_hybrid_meta_params(rng: random.Random) -> dict[str, Any]:
  return {
    "learning_rate": _log_uniform(rng, 0.02, 0.15),
    "n_estimators": rng.randrange(100, 401, 50),
    "num_leaves": rng.choice((15, 31, 63)),
    "min_child_samples": rng.randrange(10, 61, 5),
    "subsample": round(rng.uniform(0.6, 1.0), 4),
    "colsample_bytree": round(rng.uniform(0.6, 1.0), 4),
    "reg_alpha": _log_uniform(rng, 1e-4, 1.0),
    "reg_lambda": _log_uniform(rng, 1e-4, 2.0),
    "sample_weight_scheme": rng.choice(("hmm_regime_balance", "hmm_regime_target_balance")),
    "regime_change_boost": rng.choice((1.0, 1.25, 1.5, 1.75)),
    "regime_change_threshold": rng.choice((0.25, 0.35, 0.45)),
    "random_state": 7,
  }


def _mutate_hybrid_transformer_params(
  *,
  parent_params: dict[str, Any],
  rng: random.Random,
  mutation_power: float,
) -> dict[str, Any]:
  params = dict(parent_params or _random_hybrid_transformer_params(rng))
  params["lookback"] = _mutate_choice(int(params["lookback"]), (24, 32, 48, 64), rng)
  params["backbone"] = "timemixer"
  params["d_model"] = _mutate_choice(int(params["d_model"]), (32, 64), rng)
  params["nhead"] = 4 if int(params["d_model"]) % 4 == 0 else 2
  params["num_layers"] = _mutate_choice(int(params["num_layers"]), (1, 2), rng)
  params["dim_feedforward"] = _mutate_choice(int(params["dim_feedforward"]), (64, 128, 192), rng)
  params["scales"] = _mutate_choice(
    tuple(int(value) for value in params.get("scales", (1, 2, 4))),
    ((1, 2, 4), (1, 2, 4, 8), (1, 3, 6)),
    rng,
  )
  params["mixer_layers"] = _mutate_choice(int(params.get("mixer_layers", 2)), (1, 2, 3), rng)
  params["dropout"] = round(_mutate_linear_float(float(params["dropout"]), 0.0, 0.25, mutation_power, rng), 4)
  params["batch_size"] = _mutate_choice(int(params["batch_size"]), (16, 32, 64), rng)
  params["epochs"] = _mutate_choice(int(params["epochs"]), (3, 4, 6), rng)
  params["learning_rate"] = _mutate_log_float(float(params["learning_rate"]), 2e-4, 2e-3, mutation_power, rng)
  params["weight_decay"] = _mutate_log_float(float(params["weight_decay"]), 1e-6, 1e-3, mutation_power, rng)
  params["device"] = "cpu"
  return params


def _mutate_hybrid_meta_params(
  *,
  parent_params: dict[str, Any],
  rng: random.Random,
  mutation_power: float,
) -> dict[str, Any]:
  params = dict(parent_params or _random_hybrid_meta_params(rng))
  params["learning_rate"] = _mutate_log_float(float(params["learning_rate"]), 0.02, 0.15, mutation_power, rng)
  params["n_estimators"] = _mutate_int(int(params["n_estimators"]), 100, 400, 50, mutation_power, rng)
  params["num_leaves"] = _mutate_choice(int(params["num_leaves"]), (15, 31, 63), rng)
  params["min_child_samples"] = _mutate_int(int(params["min_child_samples"]), 10, 60, 5, mutation_power, rng)
  params["subsample"] = round(_mutate_linear_float(float(params["subsample"]), 0.6, 1.0, mutation_power, rng), 4)
  params["colsample_bytree"] = round(_mutate_linear_float(float(params["colsample_bytree"]), 0.6, 1.0, mutation_power, rng), 4)
  params["reg_alpha"] = _mutate_log_float(float(params["reg_alpha"]), 1e-4, 1.0, mutation_power, rng)
  params["reg_lambda"] = _mutate_log_float(float(params["reg_lambda"]), 1e-4, 2.0, mutation_power, rng)
  params["sample_weight_scheme"] = _mutate_choice(
    str(params.get("sample_weight_scheme", "hmm_regime_target_balance")),
    ("hmm_regime_balance", "hmm_regime_target_balance"),
    rng,
  )
  params["regime_change_boost"] = round(
    _mutate_linear_float(float(params.get("regime_change_boost", 1.25)), 1.0, 2.0, mutation_power, rng),
    4,
  )
  params["regime_change_threshold"] = round(
    _mutate_linear_float(float(params.get("regime_change_threshold", 0.35)), 0.15, 0.6, mutation_power, rng),
    4,
  )
  params["random_state"] = int(params.get("random_state", 7))
  return params


def _write_transformer_predictions(
  path: Path,
  rows: list[dict[str, str]],
  targets: list[float],
  predictions: list[float],
) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=["provider", "symbol", "candle_start", "target", "prediction"],
    )
    writer.writeheader()
    for row, target, prediction in zip(rows, targets, predictions):
      writer.writerow(
        {
          "provider": row.get("provider", ""),
          "symbol": row.get("symbol", ""),
          "candle_start": row.get("candle_start", ""),
          "target": target,
          "prediction": prediction,
        }
      )


def _write_hybrid_transformer_predictions(
  path: Path,
  *,
  rows: list[dict[str, str]],
  targets: list[list[float]],
  predictions: list[dict[str, float]],
  target_columns: list[str],
  prediction_columns: list[str],
) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
      handle,
      fieldnames=[
        "provider",
        "symbol",
        "candle_start",
        *[f"target_{column}" for column in target_columns],
        *prediction_columns,
      ],
    )
    writer.writeheader()
    for row, target_values, prediction_row in zip(rows, targets, predictions):
      payload = {
        "provider": row.get("provider", ""),
        "symbol": row.get("symbol", ""),
        "candle_start": row.get("candle_start", ""),
      }
      for target_column, target_value in zip(target_columns, target_values):
        payload[f"target_{target_column}"] = target_value
      for prediction_column in prediction_columns:
        payload[prediction_column] = prediction_row.get(prediction_column, "")
      writer.writerow(payload)


def _write_candidate_dataset_slice(
  root: Path,
  *,
  dataset_name: str,
  provider: str,
  symbol: str,
  rows: list[dict[str, str]],
) -> None:
  if not rows:
    raise ValueError("Cannot write an empty candidate dataset slice.")

  rows_by_date: dict[str, list[dict[str, str]]] = {}
  for row in rows:
    feature_date = row.get("feature_date") or row.get("candle_start", "")[:10]
    if not feature_date:
      continue
    rows_by_date.setdefault(feature_date, []).append(row)

  if not rows_by_date:
    raise ValueError("Candidate dataset slice does not include any dated rows.")

  fieldnames = list(rows[0].keys())
  for feature_date, date_rows in rows_by_date.items():
    year, month, day = feature_date.split("-")
    dataset_key = DatasetKey(
      provider=provider,
      symbol=symbol,
      year=year,
      month=month,
      day=day,
    )
    path = dataset_file_path(root, f"training_datasets/{dataset_name}", dataset_key)
    write_rows(path, fieldnames, date_rows)


def _hybrid_transformer_config(
  params: dict[str, Any],
  *,
  validation_fraction: float,
  horizons: tuple[int, ...],
  quantiles: tuple[float, ...],
) -> HybridTransformerConfig:
  return HybridTransformerConfig(
    lookback=int(params["lookback"]),
    backbone=str(params.get("backbone", "timemixer")),
    d_model=int(params["d_model"]),
    nhead=int(params["nhead"]),
    num_layers=int(params["num_layers"]),
    dim_feedforward=int(params["dim_feedforward"]),
    dropout=float(params["dropout"]),
    batch_size=int(params["batch_size"]),
    epochs=int(params["epochs"]),
    learning_rate=float(params["learning_rate"]),
    weight_decay=float(params["weight_decay"]),
    validation_fraction=validation_fraction,
    device=str(params.get("device", "cpu")),
    horizons=horizons,
    quantiles=quantiles,
    scales=tuple(int(value) for value in params.get("scales", (1, 2, 4))),
    mixer_layers=int(params.get("mixer_layers", 2)),
  )


def _best_results_by_model(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  best: dict[str, dict[str, Any]] = {}
  for record in records:
    if record.get("status") != "completed":
      continue
    model_name = str(record["model_name"])
    if model_name not in best or _candidate_sort_key(record) > _candidate_sort_key(best[model_name]):
      best[model_name] = record
  return best


def _write_best_results(workspace: Path, records: list[dict[str, Any]]) -> None:
  best = _best_results_by_model(records)
  (workspace / "search" / "best_results.json").write_text(
    json.dumps(best, indent=2, sort_keys=True),
    encoding="utf-8",
  )


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
  backtest = record.get("backtest", {})
  return (
    float(record.get("fitness", float("-inf"))),
    float(backtest.get("total_net_return", float("-inf"))),
    -float(backtest.get("max_drawdown", float("inf"))),
  )


def _materialize_data_times(config: OvernightTrainingConfig) -> OvernightDataConfig:
  end_time = config.data.end_time or datetime.now(UTC)
  candle_start = config.data.candle_start_time or (end_time - timedelta(days=config.data.candle_lookback_days))
  trade_start = config.data.trade_start_time or (end_time - timedelta(hours=config.data.trade_lookback_hours))
  return OvernightDataConfig(
    product=config.data.product,
    provider=config.data.provider,
    candle_interval=config.data.candle_interval,
    candle_start_time=candle_start.astimezone(UTC),
    trade_start_time=trade_start.astimezone(UTC),
    end_time=end_time.astimezone(UTC),
    candle_lookback_days=config.data.candle_lookback_days,
    trade_lookback_hours=config.data.trade_lookback_hours,
    request_timeout_seconds=config.data.request_timeout_seconds,
    request_pause_seconds=config.data.request_pause_seconds,
  )


def _update_search_state(
  state_path: Path,
  model_name: str,
  candidate_index: int,
  deadline: datetime,
  *,
  status: str,
) -> None:
  state = _load_state(state_path)
  state["search_status"] = status
  state["current_model"] = model_name
  state["candidate_index"] = candidate_index
  state["deadline"] = _format_dt(deadline)
  state["updated_at"] = _format_dt(datetime.now(UTC))
  _write_state(state_path, state)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")


def _load_candidate_records(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  records: list[dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      records.append(json.loads(line))
  return records


def _load_state(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clean_workspace(workspace: Path) -> None:
  for name in ("raw", "staging", "features", "models", "reports", "search", "state"):
    path = workspace / name
    if path.exists():
      shutil.rmtree(path)


def _format_dt(value: datetime | None) -> str:
  if value is None:
    return ""
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_dt_optional(value: Any) -> datetime | None:
  if value in (None, ""):
    return None
  return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
  import math

  return math.exp(rng.uniform(math.log(low), math.log(high)))


def _mutate_log_float(
  value: float,
  low: float,
  high: float,
  mutation_power: float,
  rng: random.Random,
) -> float:
  import math

  log_value = math.log(value)
  log_low = math.log(low)
  log_high = math.log(high)
  span = (log_high - log_low) * mutation_power
  mutated = log_value + rng.uniform(-span, span)
  return math.exp(min(log_high, max(log_low, mutated)))


def _mutate_linear_float(
  value: float,
  low: float,
  high: float,
  mutation_power: float,
  rng: random.Random,
) -> float:
  span = (high - low) * mutation_power
  mutated = value + rng.uniform(-span, span)
  return min(high, max(low, mutated))


def _mutate_int(
  value: int,
  low: int,
  high: int,
  step: int,
  mutation_power: float,
  rng: random.Random,
) -> int:
  step_count = max(1, int(round(((high - low) / max(step, 1)) * mutation_power)))
  delta = rng.randint(-step_count, step_count) * step
  mutated = value + delta
  snapped = int(round(mutated / step) * step)
  return min(high, max(low, snapped))


def _mutate_choice(value: int | None, choices: tuple[int | None, ...], rng: random.Random) -> int | None:
  if rng.random() < 0.7:
    return value
  return rng.choice(tuple(choice for choice in choices if choice != value) or choices)

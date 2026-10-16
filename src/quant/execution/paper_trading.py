from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from quant.features.market_features import FeatureBuildConfig, build_market_features
from quant.ingestion.coinbase_historical import (
  CoinbaseHistoricalBackfillConfig,
  interval_to_seconds,
  run_coinbase_historical_backfill,
)
from quant.ingestion.staging import StageConfig, run_stager
from quant.models.baselines import LogisticDirectionClassifier, RidgeReturnRegressor
from quant.models.base import utc_run_slug, verify_artifact_checksum
from quant.models.classification_outputs import (
  PREDICTION_MARGIN_COLUMN,
  build_classification_prediction_outputs,
  is_direction_classification,
  is_supported_ternary_classification,
)
from quant.models.ensembles.stacked_meta import (
  HybridStackedDirectionMetaModel,
  HybridStackedReturnMetaModel,
  HybridStackedTradeabilityMetaModel,
  StackedDirectionMetaModel,
  StackedReturnMetaModel,
)
from quant.models.transformers.hybrid_forecaster import HybridQuantileTransformer
from quant.models.transformers.return_forecaster import ReturnSequenceTransformer
from quant.models.trees.gradient_boosting import (
  HistGradientBoostingDirectionClassifier,
  HistGradientBoostingReturnRegressor,
)
from quant.risk.sizing import (
  PositionSizingConfig,
  RiskState,
  mark_position_state,
  next_active_position_bars,
  next_position_state,
  target_position_from_prediction,
)

LOGGER = logging.getLogger(__name__)

MODEL_LOADERS: dict[str, type[Any]] = {
  "ridge_return": RidgeReturnRegressor,
  "logistic_direction": LogisticDirectionClassifier,
  "histgb_return": HistGradientBoostingReturnRegressor,
  "histgb_direction": HistGradientBoostingDirectionClassifier,
  "histgb_trend_state": HistGradientBoostingDirectionClassifier,
  "histgb_trend_state_fast": HistGradientBoostingDirectionClassifier,
  "histgb_triple_barrier_fast": HistGradientBoostingDirectionClassifier,
  "transformer_return_forecaster": ReturnSequenceTransformer,
  "hybrid_quantile_transformer": HybridQuantileTransformer,
  "lightgbm_return_meta": StackedReturnMetaModel,
  "lightgbm_direction_meta": StackedDirectionMetaModel,
  "lightgbm_hybrid_return_meta": HybridStackedReturnMetaModel,
  "lightgbm_hybrid_direction_meta": HybridStackedDirectionMetaModel,
  "lightgbm_hybrid_tradeability_meta": HybridStackedTradeabilityMetaModel,
}

MODEL_ARTIFACT_FILENAMES: dict[str, str] = {
  "transformer_return_forecaster": "model.pt",
  "hybrid_quantile_transformer": "model.pt",
}

META_MODEL_NAMES = {
  "lightgbm_return_meta",
  "lightgbm_direction_meta",
  "lightgbm_hybrid_return_meta",
  "lightgbm_hybrid_direction_meta",
  "lightgbm_hybrid_tradeability_meta",
}
HYBRID_TRANSFORMER_COLUMN_PREFIX = "oof_hybrid_transformer_"
HYBRID_TRANSFORMER_ARTIFACT_KEY = "__hybrid_transformer__"


@dataclass(frozen=True, slots=True)
class SupportingArtifactSpec:
  prediction_column: str
  family: str
  model_path_name: str
  artifact_filename: str
  loader: type[Any]
  loaded_model_name: str
  source_column: str = "prediction"
  requires_sequence: bool = False


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
  artifact_dir: Path
  metadata: dict[str, Any]
  model: Any


def _classification_supporting_specs(
  *,
  alias: str,
  model_path_name: str,
  loaded_model_name: str,
  margin_source_column: str,
  probability_source_columns: tuple[str, str, str],
) -> dict[str, SupportingArtifactSpec]:
  prediction_column = f"oof_{alias}_prediction"
  output: dict[str, SupportingArtifactSpec] = {
    prediction_column: SupportingArtifactSpec(
      prediction_column=prediction_column,
      family="tabular",
      model_path_name=model_path_name,
      artifact_filename="model.pkl",
      loader=HistGradientBoostingDirectionClassifier,
      loaded_model_name=loaded_model_name,
    ),
  }
  for source_column in (
    margin_source_column,
    *probability_source_columns,
    "prediction_probability_max",
    "prediction_confidence",
  ):
    output_column = f"oof_{alias}_{source_column.removeprefix('prediction_')}"
    output[output_column] = SupportingArtifactSpec(
      prediction_column=output_column,
      family="tabular",
      model_path_name=model_path_name,
      artifact_filename="model.pkl",
      loader=HistGradientBoostingDirectionClassifier,
      loaded_model_name=loaded_model_name,
      source_column=source_column,
    )
  return output


STACKED_SUPPORTING_ARTIFACT_SPECS: dict[str, SupportingArtifactSpec] = {
  "oof_ridge_return_prediction": SupportingArtifactSpec(
    prediction_column="oof_ridge_return_prediction",
    family="tabular",
    model_path_name="ridge_return",
    artifact_filename="model.pkl",
    loader=RidgeReturnRegressor,
    loaded_model_name="ridge_return",
  ),
  "oof_histgb_return_prediction": SupportingArtifactSpec(
    prediction_column="oof_histgb_return_prediction",
    family="tabular",
    model_path_name="histgb_return",
    artifact_filename="model.pkl",
    loader=HistGradientBoostingReturnRegressor,
    loaded_model_name="histgb_return",
  ),
  "oof_logistic_direction_prediction": SupportingArtifactSpec(
    prediction_column="oof_logistic_direction_prediction",
    family="tabular",
    model_path_name="logistic_direction",
    artifact_filename="model.pkl",
    loader=LogisticDirectionClassifier,
    loaded_model_name="logistic_direction",
  ),
  "oof_histgb_direction_prediction": SupportingArtifactSpec(
    prediction_column="oof_histgb_direction_prediction",
    family="tabular",
    model_path_name="histgb_direction",
    artifact_filename="model.pkl",
    loader=HistGradientBoostingDirectionClassifier,
    loaded_model_name="histgb_direction",
  ),
  **_classification_supporting_specs(
    alias="trend_state",
    model_path_name="histgb_trend_state",
    loaded_model_name="histgb_trend_state",
    margin_source_column="prediction_trend_state_margin",
    probability_source_columns=(
      "prediction_probability_downtrend",
      "prediction_probability_chop",
      "prediction_probability_uptrend",
    ),
  ),
  **_classification_supporting_specs(
    alias="fast_trend_state",
    model_path_name="histgb_trend_state_fast",
    loaded_model_name="histgb_trend_state_fast",
    margin_source_column="prediction_trend_state_margin",
    probability_source_columns=(
      "prediction_probability_downtrend",
      "prediction_probability_chop",
      "prediction_probability_uptrend",
    ),
  ),
  **_classification_supporting_specs(
    alias="slow_trend_state",
    model_path_name="histgb_trend_state",
    loaded_model_name="histgb_trend_state",
    margin_source_column="prediction_trend_state_margin",
    probability_source_columns=(
      "prediction_probability_downtrend",
      "prediction_probability_chop",
      "prediction_probability_uptrend",
    ),
  ),
  **_classification_supporting_specs(
    alias="fast_triple_barrier",
    model_path_name="histgb_triple_barrier_fast",
    loaded_model_name="histgb_triple_barrier_fast",
    margin_source_column="prediction_triple_barrier_margin",
    probability_source_columns=(
      "prediction_probability_stop",
      "prediction_probability_no_hit",
      "prediction_probability_profit",
    ),
  ),
  "oof_transformer_return_forecaster_prediction": SupportingArtifactSpec(
    prediction_column="oof_transformer_return_forecaster_prediction",
    family="transformers",
    model_path_name="return_forecaster",
    artifact_filename="model.pt",
    loader=ReturnSequenceTransformer,
    loaded_model_name="transformer_return_forecaster",
    requires_sequence=True,
  ),
}


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
  artifact_dir: Path
  product: str
  artifact_search_root: Path | None = None
  tradeability_artifact_dir: Path | None = None
  raw_root: Path = Path("data/paper/runtime/raw")
  staging_root: Path = Path("data/paper/runtime/staging")
  feature_root: Path = Path("data/paper/runtime/features")
  output_root: Path = Path("data/paper")
  candle_interval: str = "5m"
  lookback_hours: int = 3
  poll_interval_seconds: float = 300.0
  iterations: int = 1
  fee_bps: float = 1.0
  slippage_bps: float = 0.5
  take_profit_multiple: float = 2.0
  request_timeout_seconds: float = 20.0
  request_pause_seconds: float = 0.02
  sizing: PositionSizingConfig = PositionSizingConfig()
  end_time: datetime | None = None
  config_path: Path | None = None


@dataclass(slots=True)
class PaperTradingSummary:
  iterations_completed: int = 0
  signals_written: int = 0
  orders_written: int = 0
  ledger_rows_written: int = 0
  latest_signal_path: Path | None = None
  latest_order_path: Path | None = None
  latest_ledger_path: Path | None = None
  latest_runtime_root: Path | None = None


def run_coinbase_paper_trader(config: PaperTradingConfig) -> PaperTradingSummary:
  artifact_dir = resolve_artifact_dir(config.artifact_dir)
  metadata = _load_json(artifact_dir / "metadata.json")
  model = load_model_from_artifact(artifact_dir)
  supporting_artifacts = load_supporting_artifacts_for_live_scoring(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=config.artifact_search_root,
  )
  ensure_hybrid_transformer_supporting_artifact(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=config.artifact_search_root,
    supporting_artifacts=supporting_artifacts,
  )
  tradeability_artifact, tradeability_supporting_artifacts = load_optional_live_scoring_artifact(
    artifact_dir=config.tradeability_artifact_dir,
    artifact_search_root=config.artifact_search_root,
  )
  state_path = state_file_path(config.output_root, metadata)
  summary = PaperTradingSummary()
  interval_seconds = interval_to_seconds(config.candle_interval)
  required_history_bars = required_history_bars_for_live_scoring(
    metadata=metadata,
    model=model,
    supporting_artifacts=supporting_artifacts,
  )
  if tradeability_artifact is not None:
    required_history_bars = max(
      required_history_bars,
      required_history_bars_for_live_scoring(
        metadata=tradeability_artifact.metadata,
        model=tradeability_artifact.model,
        supporting_artifacts=tradeability_supporting_artifacts,
      ),
    )
  configured_history_bars = max(1, int(math.ceil(config.lookback_hours * 3600 / interval_seconds)))
  effective_history_bars = max(configured_history_bars, required_history_bars + 1)
  effective_lookback = timedelta(seconds=effective_history_bars * interval_seconds)

  for iteration_index in range(config.iterations):
    generated_at = _reference_end_time(config, iteration_index)
    aligned_end = align_to_interval(generated_at, config.candle_interval)
    aligned_start = aligned_end - effective_lookback
    run_slug = utc_run_slug()

    raw_root = config.raw_root / run_slug
    staging_root = config.staging_root / run_slug
    feature_root = config.feature_root / run_slug
    runtime_root = config.output_root / "runtime" / metadata["provider"] / metadata["symbol"] / run_slug

    run_coinbase_historical_backfill(
      CoinbaseHistoricalBackfillConfig(
        products=(config.product,),
        channels=("candles", "trades"),
        start_time=aligned_start,
        end_time=aligned_end,
        output_root=raw_root,
        candle_interval=config.candle_interval,
        request_timeout_seconds=config.request_timeout_seconds,
        request_pause_seconds=config.request_pause_seconds,
      )
    )
    run_stager(
      StageConfig(
        input_root=raw_root,
        output_root=staging_root,
        providers=("coinbase",),
        symbols=(config.product,),
      )
    )
    build_market_features(
      FeatureBuildConfig(
        input_root=staging_root,
        output_root=feature_root,
        providers=("coinbase",),
        symbols=(config.product,),
      )
    )

    feature_rows, latest_row, feature_path = load_latest_feature_row(
      feature_root,
      provider="coinbase",
      symbol=config.product,
    )
    previous_state = _load_state(state_path)
    signal_payload, order_payload = create_paper_signal(
      row=latest_row,
      feature_rows=feature_rows,
      metadata=metadata,
      model=model,
      artifact_dir=artifact_dir,
      config=config,
      generated_at=generated_at,
      previous_position=float(previous_state.get("target_position", 0.0)),
      previous_state=previous_state,
      feature_path=feature_path,
      runtime_root=runtime_root,
      supporting_artifacts=supporting_artifacts,
      tradeability_artifact=tradeability_artifact,
      tradeability_supporting_artifacts=tradeability_supporting_artifacts,
    )
    ledger_payload = create_paper_ledger(
      row=latest_row,
      signal_payload=signal_payload,
      previous_state=previous_state,
    )
    signal_path = append_signal(config.output_root, signal_payload)
    order_path = append_order(config.output_root, order_payload) if order_payload else None
    ledger_path = append_ledger(config.output_root, ledger_payload)
    write_state(state_path, signal_payload, ledger_payload)

    summary.iterations_completed += 1
    summary.signals_written += 1
    summary.orders_written += 1 if order_path is not None else 0
    summary.ledger_rows_written += 1
    summary.latest_signal_path = signal_path
    summary.latest_order_path = order_path
    summary.latest_ledger_path = ledger_path
    summary.latest_runtime_root = runtime_root

    if iteration_index + 1 < config.iterations:
      time.sleep(config.poll_interval_seconds)

  return summary


def resolve_artifact_dir(path: Path) -> Path:
  if (path / "metadata.json").exists():
    return path

  candidates = sorted(
    candidate.parent
    for candidate in path.rglob("metadata.json")
    if candidate.is_file()
  )
  if not candidates:
    raise FileNotFoundError(f"No artifact metadata found under {path}")
  return candidates[-1]


def load_model_from_artifact(artifact_dir: Path) -> Any:
  metadata = _load_json(artifact_dir / "metadata.json")
  model_name = metadata["model_name"]
  if model_name not in MODEL_LOADERS:
    raise ValueError(f"Unsupported paper-trading model: {model_name}")
  artifact_filename = MODEL_ARTIFACT_FILENAMES.get(model_name, "model.pkl")
  verify_artifact_checksum(artifact_dir, metadata, artifact_filename)
  return MODEL_LOADERS[model_name].load(artifact_dir / artifact_filename)


def load_latest_feature_row(
  feature_root: Path,
  *,
  provider: str,
  symbol: str,
) -> tuple[list[dict[str, str]], dict[str, str], Path]:
  dataset_root = feature_root / "market_features" / provider / symbol
  if not dataset_root.exists():
    raise FileNotFoundError(f"No feature rows available under {dataset_root}")

  feature_rows: list[dict[str, str]] = []
  latest_row: dict[str, str] | None = None
  latest_path: Path | None = None
  for path in sorted(dataset_root.rglob("*.csv")):
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        if row.get("is_closed") != "true":
          continue
        feature_rows.append(row)
        if latest_row is None or row.get("candle_start", "") > latest_row.get("candle_start", ""):
          latest_row = row
          latest_path = path

  feature_rows.sort(key=lambda row: row.get("candle_start", ""))

  if latest_row is None or latest_path is None or not feature_rows:
    raise RuntimeError(f"No closed feature rows were found under {dataset_root}")
  return feature_rows, latest_row, latest_path


def create_paper_signal(
  *,
  row: dict[str, str],
  feature_rows: list[dict[str, str]],
  metadata: dict[str, Any],
  model: Any,
  artifact_dir: Path,
  config: PaperTradingConfig,
  generated_at: datetime,
  previous_position: float,
  previous_state: dict[str, Any],
  feature_path: Path,
  runtime_root: Path,
  supporting_artifacts: dict[str, LoadedArtifact] | None = None,
  tradeability_artifact: LoadedArtifact | None = None,
  tradeability_supporting_artifacts: dict[str, LoadedArtifact] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  scoring_row, scoring_context = resolve_live_feature_values(
    row=row,
    feature_rows=feature_rows,
    metadata=metadata,
    artifact_dir=artifact_dir,
    artifact_search_root=config.artifact_search_root,
    supporting_artifacts=supporting_artifacts or {},
  )

  feature_columns = tuple(metadata.get("feature_columns", ()))
  missing_columns = [column for column in feature_columns if column not in scoring_row]
  if missing_columns:
    raise ValueError(
      "Artifact requires feature columns that are unavailable in live market_features: "
      + ", ".join(missing_columns[:8])
    )

  tradeability_outputs: dict[str, float] = {}
  tradeability_context: dict[str, Any] = {}
  if tradeability_artifact is not None:
    tradeability_row, tradeability_context = resolve_live_feature_values(
      row=row,
      feature_rows=feature_rows,
      metadata=tradeability_artifact.metadata,
      artifact_dir=tradeability_artifact.artifact_dir,
      artifact_search_root=config.artifact_search_root,
      supporting_artifacts=tradeability_supporting_artifacts or {},
    )
    tradeability_feature_columns = tuple(tradeability_artifact.metadata.get("feature_columns", ()))
    missing_tradeability_columns = [
      column for column in tradeability_feature_columns if column not in tradeability_row
    ]
    if missing_tradeability_columns:
      raise ValueError(
        "Tradeability artifact requires feature columns that are unavailable in live market_features: "
        + ", ".join(missing_tradeability_columns[:8])
      )
    tradeability_outputs = predict_live_artifact_outputs(
      model=tradeability_artifact.model,
      metadata=tradeability_artifact.metadata,
      row=tradeability_row,
      feature_rows=feature_rows,
    )
    apply_prefixed_prediction_outputs(
      row=scoring_row,
      prefix="tradeability",
      outputs=tradeability_outputs,
    )

  prediction = float(
    predict_live_artifact(
      model=model,
      metadata=metadata,
      row=scoring_row,
      feature_rows=feature_rows,
    )
  )
  execution_cost_rate = (config.fee_bps + config.slippage_bps) / 10_000
  position_state = mark_position_state(
    row=scoring_row,
    previous_position=previous_position,
    entry_price=_coerce_float(previous_state.get("entry_price")),
    max_favorable_return=float(previous_state.get("max_favorable_return", 0.0) or 0.0),
    max_adverse_return=float(previous_state.get("max_adverse_return", 0.0) or 0.0),
  )
  decision = target_position_from_prediction(
    task_type=metadata["task_type"],
    prediction=prediction,
    row=scoring_row,
    sizing=config.sizing,
    execution_cost_rate=execution_cost_rate,
    previous_position=previous_position,
    active_position_bars=int(previous_state.get("active_position_bars", 0)),
    pending_direction=int(previous_state.get("pending_direction", 0)),
    pending_direction_bars=int(previous_state.get("pending_direction_bars", 0)),
    risk_state=build_risk_state(previous_state, signal_candle_start=row.get("candle_start", "")),
    position_state=position_state,
  )
  turnover = abs(decision.target_position - previous_position)
  transaction_cost = turnover * execution_cost_rate
  position_delta = decision.target_position - previous_position
  active_position_bars = next_active_position_bars(
    previous_position=previous_position,
    previous_active_position_bars=int(previous_state.get("active_position_bars", 0)),
    target_position=decision.target_position,
    same_candle=previous_state.get("candle_start") == row.get("candle_start", ""),
  )
  next_state = next_position_state(
    row=scoring_row,
    previous_position=previous_position,
    target_position=decision.target_position,
    previous_state=position_state,
  )
  generated_at_iso = _format_dt(generated_at)
  order_reason = infer_order_reason(
    previous_position=previous_position,
    target_position=decision.target_position,
    position_delta=position_delta,
  )
  trade_plan = resolve_trade_plan(
    row=scoring_row,
    decision=decision,
    previous_position=previous_position,
    previous_state=previous_state,
    entry_price=next_state.entry_price,
    take_profit_multiple=config.take_profit_multiple,
  )
  order_entry_price = trade_plan["entry_price"]
  if order_entry_price is None:
    order_entry_price = _coerce_float(previous_state.get("entry_price"))
  order_stop_loss_price = trade_plan["stop_loss_price"]
  if order_stop_loss_price is None:
    order_stop_loss_price = _coerce_float(previous_state.get("stop_loss_price"))
  order_take_profit_price = trade_plan["take_profit_price"]
  if order_take_profit_price is None:
    order_take_profit_price = _coerce_float(previous_state.get("take_profit_price"))

  signal_payload = {
    "generated_at": generated_at_iso,
    "provider": metadata["provider"],
    "symbol": metadata["symbol"],
    "model_name": metadata["model_name"],
    "task_type": metadata["task_type"],
    "target_column": metadata["target_column"],
    "artifact_dir": str(artifact_dir),
    "feature_path": str(feature_path),
    "runtime_root": str(runtime_root),
    "candle_start": row.get("candle_start", ""),
    "candle_end": row.get("candle_end", ""),
    "prediction": prediction,
    "previous_position": previous_position,
    "target_position": decision.target_position,
    "active_position_bars": active_position_bars,
    "entry_price": next_state.entry_price,
    "open_return": next_state.open_return,
    "max_favorable_return": next_state.max_favorable_return,
    "max_adverse_return": next_state.max_adverse_return,
    "pending_direction": decision.pending_direction,
    "pending_direction_bars": decision.pending_direction_bars,
    "position_delta": position_delta,
    "order_reason": order_reason,
    "direction": decision.direction,
    "signal": decision.signal,
    "confidence": decision.confidence,
    "volatility": decision.volatility,
    "volatility_scale": decision.volatility_scale,
    "raw_target_position": decision.raw_target_position,
    "stop_distance_rate": decision.stop_distance_rate,
    "max_position_by_risk": decision.max_position_by_risk,
    "session_drawdown": decision.session_drawdown,
    "portfolio_drawdown": decision.portfolio_drawdown,
    "constraint_flags": list(decision.constraint_flags),
    "turnover": turnover,
    "transaction_cost": transaction_cost,
    "fee_bps": config.fee_bps,
    "slippage_bps": config.slippage_bps,
    "take_profit_multiple": config.take_profit_multiple,
    "sizing": asdict(config.sizing),
    "feature_values": {column: scoring_row.get(column, "") for column in feature_columns},
  }
  signal_payload.update(trade_plan)
  if scoring_context:
    signal_payload.update(scoring_context)
  if tradeability_artifact is not None:
    signal_payload["tradeability_artifact_dir"] = str(tradeability_artifact.artifact_dir)
    signal_payload["tradeability_prediction_outputs"] = tradeability_outputs
    if tradeability_context.get("base_predictions"):
      signal_payload["tradeability_base_predictions"] = tradeability_context["base_predictions"]
    if tradeability_context.get("supporting_artifacts"):
      signal_payload["tradeability_supporting_artifacts"] = tradeability_context["supporting_artifacts"]

  if turnover <= 0:
    return signal_payload, None

  order_payload = {
    "generated_at": generated_at_iso,
    "provider": metadata["provider"],
    "symbol": metadata["symbol"],
    "model_name": metadata["model_name"],
    "candle_start": row.get("candle_start", ""),
    "order_side": "BUY" if position_delta > 0 else "SELL",
    "raw_target_position": decision.raw_target_position,
    "position_delta": position_delta,
    "previous_position": previous_position,
    "target_position": decision.target_position,
    "active_position_bars": active_position_bars,
    "pending_direction": decision.pending_direction,
    "pending_direction_bars": decision.pending_direction_bars,
    "constraint_flags": list(decision.constraint_flags),
    "turnover": turnover,
    "transaction_cost": transaction_cost,
    "prediction": prediction,
    "entry_price": order_entry_price,
    "stop_loss_price": order_stop_loss_price,
    "take_profit_price": order_take_profit_price,
    "order_reason": order_reason,
  }
  return signal_payload, order_payload


def apply_prefixed_prediction_outputs(
  *,
  row: dict[str, str],
  prefix: str,
  outputs: dict[str, float],
) -> None:
  for key, value in outputs.items():
    row[f"{prefix}_{key}"] = str(value)


def build_trade_plan(
  *,
  row: dict[str, str],
  decision: Any,
  take_profit_multiple: float,
  entry_price: float | None = None,
) -> dict[str, Any]:
  mark_price = _row_price(row, "close")
  entry_price = mark_price if entry_price is None else entry_price
  if entry_price is None or decision.direction == 0 or decision.target_position == 0:
    return {
      "mark_price": mark_price,
      "entry_price": None,
      "stop_loss_price": None,
      "take_profit_price": None,
      "take_profit_distance_rate": 0.0,
      "planned_reward_risk_ratio": 0.0,
      "trade_plan_status": "flat",
    }

  stop_distance_rate = max(float(decision.stop_distance_rate), 0.0)
  take_profit_distance_rate = stop_distance_rate * max(take_profit_multiple, 0.0)
  if decision.direction > 0:
    stop_loss_price = entry_price * (1.0 - stop_distance_rate)
    take_profit_price = entry_price * (1.0 + take_profit_distance_rate)
  else:
    stop_loss_price = entry_price * (1.0 + stop_distance_rate)
    take_profit_price = entry_price * (1.0 - take_profit_distance_rate)

  reward_risk_ratio = (
    take_profit_distance_rate / stop_distance_rate
    if stop_distance_rate > 0
    else 0.0
  )
  return {
    "mark_price": mark_price,
    "entry_price": entry_price,
    "stop_loss_price": stop_loss_price,
    "take_profit_price": take_profit_price,
    "take_profit_distance_rate": take_profit_distance_rate,
    "planned_reward_risk_ratio": reward_risk_ratio,
    "trade_plan_status": "active",
  }


def resolve_trade_plan(
  *,
  row: dict[str, str],
  decision: Any,
  previous_position: float,
  previous_state: dict[str, Any],
  entry_price: float | None,
  take_profit_multiple: float,
) -> dict[str, Any]:
  previous_direction = _sign(previous_position)
  target_direction = _sign(float(decision.target_position))
  preserve_existing_plan = (
    previous_direction != 0
    and previous_direction == target_direction
    and abs(float(decision.target_position)) <= abs(previous_position)
    and entry_price is not None
  )
  if preserve_existing_plan:
    stop_loss_price = _coerce_float(previous_state.get("stop_loss_price"))
    take_profit_price = _coerce_float(previous_state.get("take_profit_price"))
    if stop_loss_price is not None or take_profit_price is not None:
      return {
        "mark_price": _row_price(row, "close"),
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "take_profit_distance_rate": float(previous_state.get("take_profit_distance_rate", 0.0) or 0.0),
        "planned_reward_risk_ratio": float(previous_state.get("planned_reward_risk_ratio", 0.0) or 0.0),
        "trade_plan_status": "active",
      }
  return build_trade_plan(
    row=row,
    decision=decision,
    take_profit_multiple=take_profit_multiple,
    entry_price=entry_price,
  )


def infer_order_reason(
  *,
  previous_position: float,
  target_position: float,
  position_delta: float,
) -> str:
  if target_position == previous_position:
    return "hold"
  if previous_position == 0 and target_position != 0:
    return "enter"
  if previous_position != 0 and target_position == 0:
    return "exit"
  if previous_position != 0 and target_position != 0 and _sign(previous_position) != _sign(target_position):
    return "flip"
  if abs(target_position) > abs(previous_position):
    return "scale_in"
  if abs(target_position) < abs(previous_position):
    return "scale_out"
  return "rebalance"


def build_feature_vector(row: dict[str, str], feature_columns: tuple[str, ...]) -> list[float]:
  return [float(row.get(column, "") or 0.0) for column in feature_columns]


def predict_live_artifact(
  *,
  model: Any,
  metadata: dict[str, Any],
  row: dict[str, str],
  feature_rows: list[dict[str, str]],
) -> float:
  model_name = str(metadata.get("model_name", ""))
  feature_columns = tuple(metadata.get("feature_columns", ()))
  if model_name in {"transformer_return_forecaster", "hybrid_quantile_transformer"}:
    lookback = int(getattr(getattr(model, "config", None), "lookback", 0))
    if lookback < 1:
      raise ValueError("Transformer artifact is missing a valid lookback configuration.")
    sequence = build_sequence_feature_vector(
      feature_rows=feature_rows,
      feature_columns=feature_columns,
      lookback=lookback,
    )
    return float(model.predict([sequence])[0])
  feature_vector = build_feature_vector(row, feature_columns)
  prediction = float(model.predict([feature_vector])[0])
  if (
    is_direction_classification(
      task_type=str(metadata.get("task_type", "")),
      target_column=str(metadata.get("target_column", "")),
    )
    and str(metadata.get("extra", {}).get("primary_prediction_column", "")) == PREDICTION_MARGIN_COLUMN
  ):
    output = build_classification_prediction_outputs(
      model=model,
      features=[feature_vector],
      target_column=str(metadata.get("target_column", "")),
      predictions=[prediction],
    )[0]
    return float(output.get(PREDICTION_MARGIN_COLUMN, prediction))
  return prediction


def predict_live_artifact_outputs(
  *,
  model: Any,
  metadata: dict[str, Any],
  row: dict[str, str],
  feature_rows: list[dict[str, str]],
) -> dict[str, float]:
  prediction = predict_live_artifact(
    model=model,
    metadata=metadata,
    row=row,
    feature_rows=feature_rows,
  )
  if not is_supported_ternary_classification(
    task_type=str(metadata.get("task_type", "")),
    target_column=str(metadata.get("target_column", "")),
  ):
    return {"prediction": prediction}

  feature_vector = build_feature_vector(row, tuple(metadata.get("feature_columns", ())))
  return build_classification_prediction_outputs(
    model=model,
    features=[feature_vector],
    target_column=str(metadata.get("target_column", "")),
    predictions=[prediction],
  )[0]


def resolve_live_feature_values(
  *,
  row: dict[str, str],
  feature_rows: list[dict[str, str]],
  metadata: dict[str, Any],
  artifact_dir: Path,
  artifact_search_root: Path | None,
  supporting_artifacts: dict[str, LoadedArtifact],
) -> tuple[dict[str, str], dict[str, Any]]:
  live_row = dict(row)
  model_name = str(metadata.get("model_name", ""))
  if model_name not in META_MODEL_NAMES:
    return live_row, {}

  if not supporting_artifacts:
    supporting_artifacts = load_supporting_artifacts_for_live_scoring(
      artifact_dir=artifact_dir,
      metadata=metadata,
      artifact_search_root=artifact_search_root,
    )

  base_predictions: dict[str, float] = {}
  supporting_artifact_paths: dict[str, str] = {}
  required_columns = tuple(metadata.get("feature_columns", ()))
  hybrid_transformer_columns = [
    column
    for column in required_columns
    if column.startswith(HYBRID_TRANSFORMER_COLUMN_PREFIX)
  ]
  if hybrid_transformer_columns:
    artifact = supporting_artifacts.get(HYBRID_TRANSFORMER_ARTIFACT_KEY)
    if artifact is None:
      artifact = load_hybrid_transformer_artifact(
        artifact_dir=artifact_dir,
        metadata=metadata,
        artifact_search_root=artifact_search_root,
      )
      supporting_artifacts[HYBRID_TRANSFORMER_ARTIFACT_KEY] = artifact
    hybrid_outputs = predict_live_hybrid_transformer_outputs(
      model=artifact.model,
      feature_rows=feature_rows,
      metadata=artifact.metadata,
    )
    for column in hybrid_transformer_columns:
      source_column = "prediction_" + column.removeprefix(HYBRID_TRANSFORMER_COLUMN_PREFIX)
      if source_column not in hybrid_outputs:
        raise KeyError(f"Hybrid transformer output {source_column} is unavailable for live scoring.")
      live_row[column] = str(hybrid_outputs[source_column])
      base_predictions[column] = float(hybrid_outputs[source_column])
      supporting_artifact_paths[column] = str(artifact.artifact_dir)

  for column in required_columns:
    spec = STACKED_SUPPORTING_ARTIFACT_SPECS.get(column)
    if spec is None:
      continue
    artifact = supporting_artifacts.get(column)
    if artifact is None:
      raise FileNotFoundError(f"Missing supporting artifact for {column}")
    artifact_outputs = predict_live_artifact_outputs(
      model=artifact.model,
      metadata=artifact.metadata,
      row=row,
      feature_rows=feature_rows,
    )
    if spec.source_column not in artifact_outputs:
      raise KeyError(
        f"Supporting artifact output {spec.source_column} is unavailable for {column}."
      )
    base_prediction = float(artifact_outputs[spec.source_column])
    live_row[column] = str(base_prediction)
    base_predictions[column] = base_prediction
    supporting_artifact_paths[column] = str(artifact.artifact_dir)

  return live_row, {
    "base_predictions": base_predictions,
    "supporting_artifacts": supporting_artifact_paths,
  }


def load_supporting_artifacts_for_live_scoring(
  *,
  artifact_dir: Path,
  metadata: dict[str, Any],
  artifact_search_root: Path | None,
) -> dict[str, LoadedArtifact]:
  if str(metadata.get("model_name", "")) not in META_MODEL_NAMES:
    return {}

  feature_columns = tuple(metadata.get("feature_columns", ()))
  required_specs = [
    STACKED_SUPPORTING_ARTIFACT_SPECS[column]
    for column in feature_columns
    if column in STACKED_SUPPORTING_ARTIFACT_SPECS
  ]
  if not required_specs:
    return {}

  model_root = resolve_artifact_search_root(artifact_dir, artifact_search_root)
  provider = str(metadata.get("provider", ""))
  symbol = str(metadata.get("symbol", ""))
  artifacts: dict[str, LoadedArtifact] = {}
  for spec in required_specs:
    run_root = resolve_artifact_dir(model_root / spec.family / spec.model_path_name / provider / symbol)
    support_metadata = _load_json(run_root / "metadata.json")
    if str(support_metadata.get("model_name", "")) != spec.loaded_model_name:
      raise ValueError(
        f"Supporting artifact at {run_root} does not match expected model {spec.loaded_model_name}"
      )
    artifacts[spec.prediction_column] = LoadedArtifact(
      artifact_dir=run_root,
      metadata=support_metadata,
      model=spec.loader.load(run_root / spec.artifact_filename),
    )
  return artifacts


def load_optional_live_scoring_artifact(
  *,
  artifact_dir: Path | None,
  artifact_search_root: Path | None,
) -> tuple[LoadedArtifact | None, dict[str, LoadedArtifact]]:
  if artifact_dir is None:
    return None, {}

  resolved_artifact_dir = resolve_artifact_dir(artifact_dir)
  metadata = _load_json(resolved_artifact_dir / "metadata.json")
  model = load_model_from_artifact(resolved_artifact_dir)
  supporting_artifacts = load_supporting_artifacts_for_live_scoring(
    artifact_dir=resolved_artifact_dir,
    metadata=metadata,
    artifact_search_root=artifact_search_root,
  )
  ensure_hybrid_transformer_supporting_artifact(
    artifact_dir=resolved_artifact_dir,
    metadata=metadata,
    artifact_search_root=artifact_search_root,
    supporting_artifacts=supporting_artifacts,
  )
  return LoadedArtifact(
    artifact_dir=resolved_artifact_dir,
    metadata=metadata,
    model=model,
  ), supporting_artifacts


def ensure_hybrid_transformer_supporting_artifact(
  *,
  artifact_dir: Path,
  metadata: dict[str, Any],
  artifact_search_root: Path | None,
  supporting_artifacts: dict[str, LoadedArtifact],
) -> None:
  feature_columns = tuple(metadata.get("feature_columns", ()))
  if not any(column.startswith(HYBRID_TRANSFORMER_COLUMN_PREFIX) for column in feature_columns):
    return
  if HYBRID_TRANSFORMER_ARTIFACT_KEY in supporting_artifacts:
    return
  supporting_artifacts[HYBRID_TRANSFORMER_ARTIFACT_KEY] = load_hybrid_transformer_artifact(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=artifact_search_root,
  )


def load_hybrid_transformer_artifact(
  *,
  artifact_dir: Path,
  metadata: dict[str, Any],
  artifact_search_root: Path | None,
) -> LoadedArtifact:
  configured_path = str(metadata.get("extra", {}).get("hybrid_transformer_artifact_dir", "")).strip()
  if configured_path:
    run_root = resolve_saved_artifact_path(artifact_dir, configured_path)
    if not _is_deployable_hybrid_transformer_artifact_dir(run_root):
      fallback_run_root = resolve_deployable_hybrid_transformer_hint(
        run_root,
        provider=str(metadata.get("provider", "")),
        symbol=str(metadata.get("symbol", "")),
      )
      if fallback_run_root is None:
        raise FileNotFoundError(
          f"Configured hybrid transformer artifact path is not deployable: {run_root}"
        )
      run_root = fallback_run_root
  else:
    model_root = resolve_artifact_search_root(artifact_dir, artifact_search_root)
    provider = str(metadata.get("provider", ""))
    symbol = str(metadata.get("symbol", ""))
    run_root = resolve_artifact_dir(model_root / "hybrid_transformers" / "quantile_forecaster" / provider / symbol)
  support_metadata = _load_json(run_root / "metadata.json")
  if str(support_metadata.get("model_name", "")) != "hybrid_quantile_transformer":
    raise ValueError(f"Supporting artifact at {run_root} is not a hybrid quantile transformer.")
  return LoadedArtifact(
    artifact_dir=run_root,
    metadata=support_metadata,
    model=HybridQuantileTransformer.load(run_root / "model.pt"),
  )


def _is_deployable_hybrid_transformer_artifact_dir(path: Path) -> bool:
  return (path / "metadata.json").exists() and (path / "model.pt").exists()


def resolve_deployable_hybrid_transformer_hint(
  run_root: Path,
  *,
  provider: str,
  symbol: str,
) -> Path | None:
  walk_forward_root: Path | None = None
  for candidate in (run_root, *run_root.parents):
    if candidate.name == "walk_forward":
      walk_forward_root = candidate
      break
  if walk_forward_root is None:
    return None

  deployable_root = (
    walk_forward_root.parent
    / "hybrid_transformers"
    / "quantile_forecaster"
    / provider
    / symbol
  )
  if not deployable_root.exists():
    return None
  try:
    return resolve_artifact_dir(deployable_root)
  except FileNotFoundError:
    return None


def predict_live_hybrid_transformer_outputs(
  *,
  model: HybridQuantileTransformer,
  feature_rows: list[dict[str, str]],
  metadata: dict[str, Any],
) -> dict[str, float]:
  feature_columns = tuple(metadata.get("feature_columns", ()))
  lookback = int(getattr(getattr(model, "config", None), "lookback", 0))
  if lookback < 1:
    raise ValueError("Hybrid transformer artifact is missing a valid lookback configuration.")
  sequence = build_sequence_feature_vector(
    feature_rows=feature_rows,
    feature_columns=feature_columns,
    lookback=lookback,
  )
  output_rows = model.predict_output_rows([sequence])
  if not output_rows:
    raise RuntimeError("Hybrid transformer did not return any live outputs.")
  return output_rows[0]


def required_history_bars_for_live_scoring(
  *,
  metadata: dict[str, Any],
  model: Any,
  supporting_artifacts: dict[str, LoadedArtifact],
) -> int:
  required = 1
  model_name = str(metadata.get("model_name", ""))
  if model_name in {"transformer_return_forecaster", "hybrid_quantile_transformer"}:
    required = max(required, _artifact_lookback(model))

  for artifact in supporting_artifacts.values():
    required = max(required, _artifact_lookback(artifact.model))
  return required


def resolve_artifact_search_root(artifact_dir: Path, artifact_search_root: Path | None) -> Path:
  if artifact_search_root is not None:
    return artifact_search_root
  for parent in (artifact_dir, *artifact_dir.parents):
    if parent.name == "models":
      return parent
  raise FileNotFoundError(
    f"Unable to infer a models root from {artifact_dir}. Provide artifact_search_root explicitly."
  )


def resolve_saved_artifact_path(artifact_dir: Path, configured_path: str) -> Path:
  candidate = Path(configured_path)
  if candidate.is_absolute():
    if candidate.exists():
      return candidate
    raise FileNotFoundError(f"Configured artifact path does not exist: {candidate}")

  search_roots = [artifact_dir, *artifact_dir.parents]
  for root in search_roots:
    resolved = (root / candidate).resolve()
    if resolved.exists():
      return resolved

  raise FileNotFoundError(
    f"Configured artifact path {configured_path} could not be resolved from {artifact_dir}"
  )


def build_sequence_feature_vector(
  *,
  feature_rows: list[dict[str, str]],
  feature_columns: tuple[str, ...],
  lookback: int,
) -> list[list[float]]:
  if len(feature_rows) < lookback:
    raise ValueError(
      f"Need at least {lookback} closed feature rows for transformer scoring, got {len(feature_rows)}"
    )
  window = feature_rows[-lookback:]
  return [build_feature_vector(window_row, feature_columns) for window_row in window]


def _artifact_lookback(model: Any) -> int:
  return max(0, int(getattr(getattr(model, "config", None), "lookback", 0)))


def append_signal(output_root: Path, payload: dict[str, Any]) -> Path:
  return _append_partitioned_jsonl(output_root / "signals", payload)


def append_order(output_root: Path, payload: dict[str, Any]) -> Path:
  return _append_partitioned_jsonl(output_root / "orders", payload)


def append_ledger(output_root: Path, payload: dict[str, Any]) -> Path:
  return _append_partitioned_jsonl(output_root / "ledger", payload)


def state_file_path(output_root: Path, metadata: dict[str, Any]) -> Path:
  return output_root / "state" / metadata["provider"] / metadata["symbol"] / f"{metadata['model_name']}.json"


def write_state(path: Path, signal_payload: dict[str, Any], ledger_payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(
      {
        "updated_at": signal_payload["generated_at"],
        "provider": signal_payload["provider"],
        "symbol": signal_payload["symbol"],
        "model_name": signal_payload["model_name"],
        "candle_start": signal_payload["candle_start"],
        "target_position": signal_payload["target_position"],
        "active_position_bars": signal_payload.get("active_position_bars", 0),
        "pending_direction": signal_payload.get("pending_direction", 0),
        "pending_direction_bars": signal_payload.get("pending_direction_bars", 0),
        "prediction": signal_payload["prediction"],
        "mark_price": signal_payload.get("mark_price"),
        "entry_price": signal_payload.get("entry_price"),
        "open_return": signal_payload.get("open_return", 0.0),
        "max_favorable_return": signal_payload.get("max_favorable_return", 0.0),
        "max_adverse_return": signal_payload.get("max_adverse_return", 0.0),
        "stop_loss_price": signal_payload.get("stop_loss_price"),
        "take_profit_price": signal_payload.get("take_profit_price"),
        "take_profit_distance_rate": signal_payload.get("take_profit_distance_rate"),
        "planned_reward_risk_ratio": signal_payload.get("planned_reward_risk_ratio"),
        "trade_plan_status": signal_payload.get("trade_plan_status", ""),
        "equity": ledger_payload["equity"],
        "peak_equity": ledger_payload["peak_equity"],
        "session_start_equity": ledger_payload["session_start_equity"],
        "session_date": ledger_payload["session_date"],
        "session_drawdown": ledger_payload["session_drawdown"],
        "portfolio_drawdown": ledger_payload["portfolio_drawdown"],
        "gross_return": ledger_payload["gross_return"],
        "net_return": ledger_payload["net_return"],
        "realized_bar_return": ledger_payload["realized_bar_return"],
        "turnover": ledger_payload["turnover"],
        "transaction_cost": ledger_payload["transaction_cost"],
        "exit_price": ledger_payload.get("exit_price"),
        "exit_reason": ledger_payload.get("exit_reason", ""),
        "take_profit_hit": ledger_payload.get("take_profit_hit", False),
        "stop_loss_hit": ledger_payload.get("stop_loss_hit", False),
        "constraint_flags": signal_payload["constraint_flags"],
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )


def create_paper_ledger(
  *,
  row: dict[str, str],
  signal_payload: dict[str, Any],
  previous_state: dict[str, Any],
) -> dict[str, Any]:
  previous_position = float(previous_state.get("target_position", 0.0))
  previous_equity = float(previous_state.get("equity", 1.0))
  previous_peak_equity = float(previous_state.get("peak_equity", previous_equity))
  same_candle = previous_state.get("candle_start") == signal_payload["candle_start"]
  session_date = signal_payload["candle_start"][:10]
  previous_session_date = str(previous_state.get("session_date", ""))
  session_start_equity = (
    previous_equity
    if session_date and session_date != previous_session_date
    else float(previous_state.get("session_start_equity", previous_equity))
  )
  exit_context = build_exit_context(row=row, previous_state=previous_state, same_candle=same_candle)
  realized_bar_return = float(exit_context["realized_bar_return"])
  gross_return = previous_position * realized_bar_return
  net_return = gross_return - float(signal_payload["transaction_cost"])
  equity = previous_equity * (1.0 + net_return)
  peak_equity = max(previous_peak_equity, equity)
  session_drawdown = _drawdown_fraction(session_start_equity, equity)
  portfolio_drawdown = _drawdown_fraction(peak_equity, equity)

  return {
    "generated_at": signal_payload["generated_at"],
    "provider": signal_payload["provider"],
    "symbol": signal_payload["symbol"],
    "model_name": signal_payload["model_name"],
    "task_type": signal_payload["task_type"],
    "candle_start": signal_payload["candle_start"],
    "candle_end": signal_payload["candle_end"],
    "prediction": signal_payload["prediction"],
    "previous_position": previous_position,
    "target_position": signal_payload["target_position"],
    "active_position_bars": signal_payload.get("active_position_bars", 0),
    "position_delta": signal_payload["position_delta"],
    "turnover": signal_payload["turnover"],
    "transaction_cost": signal_payload["transaction_cost"],
    "realized_bar_return": realized_bar_return,
    "gross_return": gross_return,
    "net_return": net_return,
    "equity": equity,
    "peak_equity": peak_equity,
    "session_start_equity": session_start_equity,
    "session_date": session_date,
    "session_drawdown": session_drawdown,
    "portfolio_drawdown": portfolio_drawdown,
    "entry_price": previous_state.get("entry_price"),
    "stop_loss_price": previous_state.get("stop_loss_price"),
    "take_profit_price": previous_state.get("take_profit_price"),
    "exit_price": exit_context["exit_price"],
    "exit_reason": exit_context["exit_reason"],
    "take_profit_hit": exit_context["take_profit_hit"],
    "stop_loss_hit": exit_context["stop_loss_hit"],
    "bar_open": exit_context["bar_open"],
    "bar_high": exit_context["bar_high"],
    "bar_low": exit_context["bar_low"],
    "bar_close": exit_context["bar_close"],
  }


def build_exit_context(
  *,
  row: dict[str, str],
  previous_state: dict[str, Any],
  same_candle: bool,
) -> dict[str, Any]:
  bar_open = _row_price(row, "open")
  bar_high = _row_price(row, "high")
  bar_low = _row_price(row, "low")
  bar_close = _row_price(row, "close")
  default_bar_return = (
    float(row.get("close_return_1_bar", "") or 0.0)
    if not same_candle
    else 0.0
  )
  previous_position = float(previous_state.get("target_position", 0.0))
  stop_loss_price = _coerce_float(previous_state.get("stop_loss_price"))
  take_profit_price = _coerce_float(previous_state.get("take_profit_price"))

  if same_candle or previous_position == 0 or bar_open is None:
    return {
      "realized_bar_return": default_bar_return,
      "exit_price": None,
      "exit_reason": "",
      "take_profit_hit": False,
      "stop_loss_hit": False,
      "bar_open": bar_open,
      "bar_high": bar_high,
      "bar_low": bar_low,
      "bar_close": bar_close,
    }

  direction = _sign(previous_position)
  take_profit_hit = _take_profit_hit(direction, take_profit_price, bar_high, bar_low)
  stop_loss_hit = _stop_loss_hit(direction, stop_loss_price, bar_high, bar_low)

  exit_price: float | None = None
  exit_reason = ""
  if take_profit_hit and stop_loss_hit:
    exit_price = stop_loss_price
    exit_reason = "stop_loss_ambiguous"
  elif stop_loss_hit:
    exit_price = stop_loss_price
    exit_reason = "stop_loss"
  elif take_profit_hit:
    exit_price = take_profit_price
    exit_reason = "take_profit"

  if exit_price is None:
    return {
      "realized_bar_return": default_bar_return,
      "exit_price": None,
      "exit_reason": "",
      "take_profit_hit": False,
      "stop_loss_hit": False,
      "bar_open": bar_open,
      "bar_high": bar_high,
      "bar_low": bar_low,
      "bar_close": bar_close,
    }

  return {
    "realized_bar_return": (exit_price / bar_open) - 1.0,
    "exit_price": exit_price,
    "exit_reason": exit_reason,
    "take_profit_hit": take_profit_hit and not stop_loss_hit,
    "stop_loss_hit": stop_loss_hit,
    "bar_open": bar_open,
    "bar_high": bar_high,
    "bar_low": bar_low,
    "bar_close": bar_close,
  }


def align_to_interval(value: datetime, interval: str) -> datetime:
  seconds = interval_to_seconds(interval)
  timestamp = int(value.astimezone(UTC).timestamp())
  floored = timestamp - (timestamp % seconds)
  return datetime.fromtimestamp(floored, tz=UTC)


def _append_partitioned_jsonl(root: Path, payload: dict[str, Any]) -> Path:
  event_dt = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00")).astimezone(UTC)
  path = (
    root
    / str(payload["provider"])
    / str(payload["symbol"])
    / event_dt.strftime("%Y")
    / event_dt.strftime("%m")
    / event_dt.strftime("%d")
    / f"{event_dt.strftime('%H')}.jsonl"
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    json.dump(payload, handle, separators=(",", ":"))
    handle.write("\n")
  return path


def _reference_end_time(config: PaperTradingConfig, iteration_index: int) -> datetime:
  if config.end_time is not None:
    return config.end_time.astimezone(UTC) + timedelta(seconds=iteration_index * config.poll_interval_seconds)
  return datetime.now(UTC)


def _take_profit_hit(
  direction: int,
  take_profit_price: float | None,
  bar_high: float | None,
  bar_low: float | None,
) -> bool:
  if take_profit_price is None:
    return False
  if direction > 0 and bar_high is not None:
    return bar_high >= take_profit_price
  if direction < 0 and bar_low is not None:
    return bar_low <= take_profit_price
  return False


def _stop_loss_hit(
  direction: int,
  stop_loss_price: float | None,
  bar_high: float | None,
  bar_low: float | None,
) -> bool:
  if stop_loss_price is None:
    return False
  if direction > 0 and bar_low is not None:
    return bar_low <= stop_loss_price
  if direction < 0 and bar_high is not None:
    return bar_high >= stop_loss_price
  return False


def _row_price(row: dict[str, str], column: str) -> float | None:
  value = row.get(column, "")
  if value in ("", None):
    return None
  return float(value)


def _coerce_float(value: Any) -> float | None:
  if value in ("", None):
    return None
  return float(value)


def _sign(value: float) -> int:
  if value > 0:
    return 1
  if value < 0:
    return -1
  return 0


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _load_state(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  return _load_json(path)


def _format_dt(value: datetime) -> str:
  return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_risk_state(previous_state: dict[str, Any], *, signal_candle_start: str) -> RiskState:
  equity = float(previous_state.get("equity", 1.0))
  peak_equity = float(previous_state.get("peak_equity", equity))
  session_date = signal_candle_start[:10]
  previous_session_date = str(previous_state.get("session_date", ""))
  session_start_equity = (
    equity
    if session_date and session_date != previous_session_date
    else float(previous_state.get("session_start_equity", equity))
  )
  return RiskState(
    equity=equity,
    peak_equity=peak_equity,
    session_start_equity=session_start_equity,
  )


def _drawdown_fraction(peak_equity: float, equity: float) -> float:
  if peak_equity <= 0:
    return 0.0
  return max(0.0, (peak_equity - equity) / peak_equity)

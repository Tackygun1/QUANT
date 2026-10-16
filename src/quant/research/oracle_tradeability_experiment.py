from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, ModelBacktestSummary, run_model_backtest
from quant.execution.paper_trading import (
  ensure_hybrid_transformer_supporting_artifact,
  load_model_from_artifact,
  load_supporting_artifacts_for_live_scoring,
  predict_live_artifact_outputs,
  resolve_live_feature_values,
)
from quant.research.execution_tuning import load_sizing_from_backtest_config
from quant.research.oracle_profit_bound import OracleConstraintProfile, compute_constrained_cost_aware_oracle_positions
from quant.research.rolling_holdout import aggregate_model_window_results
from quant.risk.sizing import PositionSizingConfig
from quant.training.datasets import SeriesKey, discover_series, load_series_rows_for_dataset
from quant.training.supervised_dataset import DatasetKey, dataset_file_path, write_rows
from quant.training.train_meta import MetaTrainingConfig, train_meta_models


@dataclass(frozen=True, slots=True)
class TradeabilityModelParamSet:
  slug: str
  params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OracleTradeabilityExperimentConfig:
  feature_root: Path
  direction_artifact_dir: Path
  base_execution_config_path: Path
  holdout_root: Path
  output_root: Path
  source_dataset_name: str = "market_hybrid_stacked_meta_only_zones"
  providers: tuple[str, ...] = ("coinbase",)
  symbols: tuple[str, ...] = ("BTC-USD",)
  oracle_profiles: tuple[OracleConstraintProfile, ...] = (
    OracleConstraintProfile(
      label="hold2_max40",
      min_hold_bars=2,
      max_position_change=1.0,
      max_trades_per_day=40,
    ),
    OracleConstraintProfile(
      label="hold6_max40",
      min_hold_bars=6,
      max_position_change=1.0,
      max_trades_per_day=40,
    ),
    OracleConstraintProfile(
      label="hold6_max20",
      min_hold_bars=6,
      max_position_change=1.0,
      max_trades_per_day=20,
    ),
  )
  model_param_sets: tuple[TradeabilityModelParamSet, ...] = (
    TradeabilityModelParamSet(
      slug="std_regularized",
      params={
        "learning_rate": 0.0690648415165857,
        "n_estimators": 150,
        "num_leaves": 15,
        "max_depth": 3,
        "min_child_samples": 72,
        "min_split_gain": 0.1,
        "subsample": 0.6,
        "colsample_bytree": 0.7371,
        "feature_fraction_bynode": 0.5,
        "reg_alpha": 0.02,
        "reg_lambda": 4.0,
        "path_smooth": 20.0,
        "random_state": 7,
      },
    ),
    TradeabilityModelParamSet(
      slug="extra_regularized",
      params={
        "learning_rate": 0.045,
        "n_estimators": 220,
        "num_leaves": 11,
        "max_depth": 3,
        "min_child_samples": 96,
        "min_split_gain": 0.15,
        "subsample": 0.65,
        "colsample_bytree": 0.7,
        "feature_fraction_bynode": 0.4,
        "reg_alpha": 0.05,
        "reg_lambda": 6.0,
        "path_smooth": 30.0,
        "random_state": 7,
      },
    ),
    TradeabilityModelParamSet(
      slug="deeper_balanced",
      params={
        "learning_rate": 0.05,
        "n_estimators": 180,
        "num_leaves": 31,
        "max_depth": 4,
        "min_child_samples": 64,
        "min_split_gain": 0.05,
        "subsample": 0.75,
        "colsample_bytree": 0.8,
        "feature_fraction_bynode": 0.6,
        "reg_alpha": 0.02,
        "reg_lambda": 3.0,
        "path_smooth": 15.0,
        "random_state": 7,
      },
    ),
  )
  tradeability_gate_thresholds: tuple[float, ...] = (0.5, 0.55, 0.6, 0.65, 0.7)
  tradeability_gate_min_confidences: tuple[float, ...] = (0.0, 0.05, 0.1)
  validation_fraction: float = 0.2
  minimum_rows: int = 512
  sample_weight_scheme: str = "hmm_regime_target_balance"
  regime_change_boost: float = 1.25
  regime_change_threshold: float = 0.35
  probability_calibration_method: str = ""
  top_k_holdout_candidates: int = 5
  oracle_label_mode: str = "entry_action"
  baseline_tradeability_artifact_dirs: tuple[Path, ...] = ()
  artifact_search_root: Path | None = None
  supporting_hybrid_transformer_artifact_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class OracleDatasetSummary:
  dataset_name: str
  label_mode: str
  rows_written: int
  positive_labels: int
  active_periods: int
  trades: int
  series_processed: int


@dataclass(frozen=True, slots=True)
class CandidateTrainingResult:
  candidate_slug: str
  dataset_name: str
  artifact_dir: Path
  profile_label: str
  param_slug: str
  source: str
  validation_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class GateVariantResult:
  variant_slug: str
  sizing: PositionSizingConfig
  backtest: ModelBacktestSummary


VALID_ORACLE_LABEL_MODES = {
  "active_position",
  "entry_action",
}


def _normalize_oracle_label_mode(label_mode: str) -> str:
  normalized = str(label_mode or "entry_action").strip().lower()
  if normalized not in VALID_ORACLE_LABEL_MODES:
    raise ValueError(
      "oracle_label_mode must be one of: "
      + ", ".join(sorted(VALID_ORACLE_LABEL_MODES))
    )
  return normalized


def _oracle_label_mode_slug(label_mode: str) -> str:
  normalized = _normalize_oracle_label_mode(label_mode)
  return "entry" if normalized == "entry_action" else "active"


def run_oracle_tradeability_experiment(config: OracleTradeabilityExperimentConfig) -> dict[str, Any]:
  label_mode = _normalize_oracle_label_mode(config.oracle_label_mode)
  output_root = config.output_root
  output_root.mkdir(parents=True, exist_ok=True)
  base_execution_config = _load_json(config.base_execution_config_path)
  base_sizing = load_sizing_from_backtest_config(config.base_execution_config_path)
  direction_metadata = _load_json(config.direction_artifact_dir / "metadata.json")
  direction_dataset_name = str(direction_metadata.get("extra", {}).get("dataset_name", config.source_dataset_name))
  artifact_search_root = config.artifact_search_root or _default_artifact_search_root(config.direction_artifact_dir)
  fee_bps = float(base_execution_config.get("fee_bps", 1.0))
  slippage_bps = float(base_execution_config.get("slippage_bps", 0.5))
  execution_cost_rate = (fee_bps + slippage_bps) / 10_000.0

  dataset_summaries: list[dict[str, Any]] = []
  trained_candidates: list[CandidateTrainingResult] = []

  for profile in config.oracle_profiles:
    dataset_name = f"{config.source_dataset_name}_oracle_{profile.label}_{_oracle_label_mode_slug(label_mode)}"
    summary = build_oracle_tradeability_dataset(
      input_root=config.feature_root,
      output_root=config.feature_root,
      input_dataset_name=config.source_dataset_name,
      output_dataset_name=dataset_name,
      providers=config.providers,
      symbols=config.symbols,
      oracle_profile=profile,
      execution_cost_rate=execution_cost_rate,
      label_mode=label_mode,
    )
    dataset_summaries.append(asdict(summary))
    for param_set in config.model_param_sets:
      trained_candidates.append(
        train_tradeability_candidate(
          feature_root=config.feature_root,
          output_root=output_root / "models" / _oracle_label_mode_slug(label_mode) / profile.label / param_set.slug,
          dataset_name=dataset_name,
          provider=config.providers[0],
          symbol=config.symbols[0],
          param_set=param_set,
          sample_weight_scheme=config.sample_weight_scheme,
          validation_fraction=config.validation_fraction,
          minimum_rows=config.minimum_rows,
          regime_change_boost=config.regime_change_boost,
          regime_change_threshold=config.regime_change_threshold,
          probability_calibration_method=config.probability_calibration_method,
          supporting_hybrid_transformer_artifact_dir=config.supporting_hybrid_transformer_artifact_dir,
        )
      )

  for artifact_dir in config.baseline_tradeability_artifact_dirs:
    metadata = _load_json(artifact_dir / "metadata.json")
    trained_candidates.append(
      CandidateTrainingResult(
        candidate_slug=f"baseline_{artifact_dir.name}",
        dataset_name=str(metadata.get("extra", {}).get("dataset_name", direction_dataset_name)),
        artifact_dir=artifact_dir,
        profile_label="baseline",
        param_slug="baseline",
        source="baseline_artifact",
        validation_metrics={str(key): float(value) for key, value in metadata.get("metrics", {}).items()},
      )
    )

  baseline_tuning = evaluate_direction_baseline(
    feature_root=config.feature_root,
    dataset_name=direction_dataset_name,
    direction_artifact_dir=config.direction_artifact_dir,
    output_root=output_root / "tuning_baseline",
    fee_bps=fee_bps,
    slippage_bps=slippage_bps,
    sizing=base_sizing,
  )

  tuning_candidates: list[dict[str, Any]] = []
  for candidate in trained_candidates:
    tuning_result = evaluate_tradeability_candidate_on_tuning(
      feature_root=config.feature_root,
      direction_artifact_dir=config.direction_artifact_dir,
      direction_dataset_name=direction_dataset_name,
      tradeability_candidate=candidate,
      output_root=output_root / "tuning",
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      base_sizing=base_sizing,
      gate_thresholds=config.tradeability_gate_thresholds,
      gate_min_confidences=config.tradeability_gate_min_confidences,
    )
    tuning_candidates.append(tuning_result)

  tuning_candidates.sort(
    key=lambda item: (
      float(item["best_tuning"]["backtest"]["total_net_return"]),
      float(item["best_tuning"]["backtest"]["profit_factor_after_cost"]),
      -float(item["best_tuning"]["backtest"]["max_drawdown"]),
    ),
    reverse=True,
  )
  top_holdout_candidates = tuning_candidates[: max(1, config.top_k_holdout_candidates)]
  holdout_results = [
    evaluate_tradeability_candidate_on_exact_holdouts(
      holdout_root=config.holdout_root,
      direction_artifact_dir=config.direction_artifact_dir,
      tradeability_candidate=item,
      output_root=output_root / "holdouts",
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      artifact_search_root=artifact_search_root,
      supporting_hybrid_transformer_artifact_dir=config.supporting_hybrid_transformer_artifact_dir,
    )
    for item in top_holdout_candidates
  ]
  holdout_results.sort(
    key=lambda item: (
      float(item["long_holdout"]["backtest"]["total_net_return"]),
      float(item["rolling_aggregate"]["compounded_net_return"]),
      -float(item["long_holdout"]["backtest"]["max_drawdown"]),
    ),
    reverse=True,
  )

  payload = {
    "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "output_root": str(output_root),
    "feature_root": str(config.feature_root),
    "direction_artifact_dir": str(config.direction_artifact_dir),
    "holdout_root": str(config.holdout_root),
    "base_execution_config_path": str(config.base_execution_config_path),
    "base_direction_dataset_name": direction_dataset_name,
    "oracle_label_mode": label_mode,
    "baseline_direction_backtest": _backtest_payload(baseline_tuning.backtest),
    "oracle_dataset_summaries": dataset_summaries,
    "trained_candidates": [
      {
        "candidate_slug": candidate.candidate_slug,
        "dataset_name": candidate.dataset_name,
        "artifact_dir": str(candidate.artifact_dir),
        "profile_label": candidate.profile_label,
        "param_slug": candidate.param_slug,
        "source": candidate.source,
        "validation_metrics": candidate.validation_metrics,
      }
      for candidate in trained_candidates
    ],
    "tuning_candidates": tuning_candidates,
    "top_holdout_candidates": holdout_results,
  }
  summary_path = output_root / "summary.json"
  summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
  return payload


def build_oracle_tradeability_dataset(
  *,
  input_root: Path,
  output_root: Path,
  input_dataset_name: str,
  output_dataset_name: str,
  providers: tuple[str, ...],
  symbols: tuple[str, ...],
  oracle_profile: OracleConstraintProfile,
  execution_cost_rate: float,
  label_mode: str,
) -> OracleDatasetSummary:
  normalized_label_mode = _normalize_oracle_label_mode(label_mode)
  rows_written = 0
  positive_labels = 0
  active_periods = 0
  total_trades = 0
  series_processed = 0
  for series_key in discover_series(
    input_root,
    dataset_name=input_dataset_name,
    providers=providers,
    symbols=symbols,
  ):
    rows = load_series_rows_for_dataset(
      input_root,
      series_key,
      dataset_name=input_dataset_name,
    )
    if not rows:
      continue
    series_rows, series_stats = relabel_tradeability_rows(
      rows=rows,
      oracle_profile=oracle_profile,
      execution_cost_rate=execution_cost_rate,
      label_mode=normalized_label_mode,
    )
    if not series_rows:
      continue
    write_partitioned_dataset_rows(
      output_root=output_root,
      dataset_name=output_dataset_name,
      series_key=series_key,
      rows=series_rows,
    )
    rows_written += len(series_rows)
    positive_labels += int(series_stats["positive_labels"])
    active_periods += int(series_stats["active_periods"])
    total_trades += int(series_stats["trades"])
    series_processed += 1
  return OracleDatasetSummary(
    dataset_name=output_dataset_name,
    label_mode=normalized_label_mode,
    rows_written=rows_written,
    positive_labels=positive_labels,
    active_periods=active_periods,
    trades=total_trades,
    series_processed=series_processed,
  )


def relabel_tradeability_rows(
  *,
  rows: list[dict[str, str]],
  oracle_profile: OracleConstraintProfile,
  execution_cost_rate: float,
  label_mode: str = "entry_action",
) -> tuple[list[dict[str, str]], dict[str, int]]:
  normalized_label_mode = _normalize_oracle_label_mode(label_mode)
  valid_indices: list[int] = []
  candle_starts: list[str] = []
  realized_returns: list[float] = []
  for index, row in enumerate(rows):
    value = row.get("forward_return_1_bar", "")
    if value == "":
      continue
    valid_indices.append(index)
    candle_starts.append(str(row.get("candle_start", "")))
    realized_returns.append(float(value))
  oracle_positions: tuple[int, ...] = ()
  if valid_indices:
    oracle_positions = compute_constrained_cost_aware_oracle_positions(
      candle_starts=tuple(candle_starts),
      realized_returns=tuple(realized_returns),
      execution_cost_rate=execution_cost_rate,
      profile=oracle_profile,
    )
  position_by_index = {
    row_index: position
    for row_index, position in zip(valid_indices, oracle_positions, strict=True)
  }
  output_rows: list[dict[str, str]] = []
  positive_labels = 0
  active_periods = 0
  trades = 0
  previous_position = 0
  for index, row in enumerate(rows):
    output_row = dict(row)
    output_row.setdefault("legacy_tradeable_label_1_bar", row.get("tradeable_label_1_bar", ""))
    output_row.setdefault("oracle_position_label_1_bar", "")
    output_row.setdefault("oracle_tradeability_profile_1_bar", oracle_profile.label)
    output_row.setdefault("oracle_tradeability_label_mode_1_bar", normalized_label_mode)
    output_row.setdefault("oracle_tradeability_active_label_1_bar", "")
    output_row.setdefault("oracle_tradeability_entry_label_1_bar", "")
    position = position_by_index.get(index)
    if position is None:
      output_row["tradeable_label_1_bar"] = ""
      output_row["oracle_position_label_1_bar"] = ""
    else:
      active_label = "1" if position != 0 else "0"
      entry_label = "1" if position != 0 and position != previous_position else "0"
      if normalized_label_mode == "active_position":
        label = active_label
      else:
        label = entry_label
      output_row["tradeable_label_1_bar"] = label
      output_row["oracle_position_label_1_bar"] = str(position)
      output_row["oracle_tradeability_active_label_1_bar"] = active_label
      output_row["oracle_tradeability_entry_label_1_bar"] = entry_label
      positive_labels += 1 if label == "1" else 0
      active_periods += 1 if position != 0 else 0
      trades += 1 if position != previous_position else 0
      previous_position = position
    output_rows.append(output_row)
  return output_rows, {
    "positive_labels": positive_labels,
    "active_periods": active_periods,
    "trades": trades,
  }


def write_partitioned_dataset_rows(
  *,
  output_root: Path,
  dataset_name: str,
  series_key: SeriesKey,
  rows: list[dict[str, str]],
) -> None:
  if not rows:
    return
  fieldnames = list(rows[0].keys())
  rows_by_date: dict[str, list[dict[str, str]]] = {}
  for row in rows:
    feature_date = str(row.get("feature_date", ""))
    rows_by_date.setdefault(feature_date, []).append(row)
  for feature_date, feature_rows in rows_by_date.items():
    year, month, day = feature_date.split("-")
    output_path = dataset_file_path(
      output_root,
      f"training_datasets/{dataset_name}",
      DatasetKey(provider=series_key.provider, symbol=series_key.symbol, year=year, month=month, day=day),
    )
    write_rows(output_path, fieldnames, feature_rows)


def train_tradeability_candidate(
  *,
  feature_root: Path,
  output_root: Path,
  dataset_name: str,
  provider: str,
  symbol: str,
  param_set: TradeabilityModelParamSet,
  sample_weight_scheme: str,
  validation_fraction: float,
  minimum_rows: int,
  regime_change_boost: float,
  regime_change_threshold: float,
  probability_calibration_method: str,
  supporting_hybrid_transformer_artifact_dir: Path | None,
) -> CandidateTrainingResult:
  config = MetaTrainingConfig(
    input_root=feature_root,
    output_root=output_root,
    providers=(provider,),
    symbols=(symbol,),
    model_names=("lightgbm_hybrid_tradeability_meta",),
    validation_fraction=validation_fraction,
    minimum_rows=minimum_rows,
    dataset_name=dataset_name,
    model_params={"lightgbm_hybrid_tradeability_meta": param_set.params},
    sample_weight_scheme=sample_weight_scheme,
    regime_change_boost=regime_change_boost,
    regime_change_threshold=regime_change_threshold,
    probability_calibration_method=probability_calibration_method,
    calibration_model_names=(),
  )
  train_meta_models(config)
  artifact_dir = resolve_latest_artifact_dir(
    output_root / "ensembles" / "lightgbm_hybrid_tradeability_meta" / provider / symbol
  )
  if supporting_hybrid_transformer_artifact_dir is not None:
    metadata_path = artifact_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    metadata.setdefault("extra", {})
    metadata["extra"]["hybrid_transformer_artifact_dir"] = str(supporting_hybrid_transformer_artifact_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
  metadata = _load_json(artifact_dir / "metadata.json")
  candidate_slug = f"{dataset_name.removeprefix('market_hybrid_stacked_')}_{param_set.slug}"
  return CandidateTrainingResult(
    candidate_slug=candidate_slug,
    dataset_name=dataset_name,
    artifact_dir=artifact_dir,
    profile_label=dataset_name.removeprefix("market_hybrid_stacked_meta_only_zones_oracle_"),
    param_slug=param_set.slug,
    source="trained_oracle",
    validation_metrics={str(key): float(value) for key, value in metadata.get("metrics", {}).items()},
  )


def evaluate_direction_baseline(
  *,
  feature_root: Path,
  dataset_name: str,
  direction_artifact_dir: Path,
  output_root: Path,
  fee_bps: float,
  slippage_bps: float,
  sizing: PositionSizingConfig,
) -> GateVariantResult:
  backtest = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=direction_artifact_dir,
      input_root=feature_root,
      output_root=output_root,
      dataset_name=dataset_name,
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      sizing=sizing,
    )
  )
  return GateVariantResult(
    variant_slug="baseline_no_gate",
    sizing=sizing,
    backtest=backtest,
  )


def evaluate_tradeability_candidate_on_tuning(
  *,
  feature_root: Path,
  direction_artifact_dir: Path,
  direction_dataset_name: str,
  tradeability_candidate: CandidateTrainingResult,
  output_root: Path,
  fee_bps: float,
  slippage_bps: float,
  base_sizing: PositionSizingConfig,
  gate_thresholds: tuple[float, ...],
  gate_min_confidences: tuple[float, ...],
) -> dict[str, Any]:
  augmented_dataset_name = f"{tradeability_candidate.dataset_name}_gate_{tradeability_candidate.param_slug}"
  augment_dataset_with_tradeability_predictions(
    input_root=feature_root,
    output_root=feature_root,
    input_dataset_name=tradeability_candidate.dataset_name,
    output_dataset_name=augmented_dataset_name,
    providers=(tradeability_candidate.artifact_dir.parts[-3],),
    symbols=(tradeability_candidate.artifact_dir.parts[-2],),
    prediction_lookup=load_tradeability_prediction_lookup_from_csv(
      tradeability_candidate.artifact_dir / "validation_predictions.csv",
    ),
  )

  baseline_backtest = evaluate_direction_baseline(
    feature_root=feature_root,
    dataset_name=direction_dataset_name,
    direction_artifact_dir=direction_artifact_dir,
    output_root=output_root / "baseline",
    fee_bps=fee_bps,
    slippage_bps=slippage_bps,
    sizing=base_sizing,
  )
  variants: list[GateVariantResult] = []
  for threshold in gate_thresholds:
    for min_confidence in gate_min_confidences:
      sizing = replace(
        base_sizing,
        tradeability_gate_enabled=True,
        tradeability_gate_threshold=threshold,
        tradeability_gate_min_confidence=min_confidence,
      )
      variant_slug = f"gate{int(round(threshold * 100)):02d}_conf{int(round(min_confidence * 100)):02d}"
      backtest = run_model_backtest(
        ModelBacktestConfig(
          artifact_dir=direction_artifact_dir,
          input_root=feature_root,
          output_root=output_root / tradeability_candidate.candidate_slug / variant_slug,
          dataset_name=augmented_dataset_name,
          fee_bps=fee_bps,
          slippage_bps=slippage_bps,
          sizing=sizing,
        )
      )
      variants.append(GateVariantResult(variant_slug=variant_slug, sizing=sizing, backtest=backtest))
  variants.sort(
    key=lambda item: (
      float(item.backtest.total_net_return),
      float(item.backtest.profit_factor_after_cost),
      -float(item.backtest.max_drawdown),
    ),
    reverse=True,
  )
  best_variant = variants[0] if variants else baseline_backtest
  return {
    "candidate_slug": tradeability_candidate.candidate_slug,
    "dataset_name": tradeability_candidate.dataset_name,
    "source": tradeability_candidate.source,
    "artifact_dir": str(tradeability_candidate.artifact_dir),
    "validation_metrics": tradeability_candidate.validation_metrics,
    "baseline_tuning": {
      "variant_slug": baseline_backtest.variant_slug,
      "backtest": _backtest_payload(baseline_backtest.backtest),
      "sizing": asdict(baseline_backtest.sizing),
    },
    "best_tuning": {
      "variant_slug": best_variant.variant_slug,
      "backtest": _backtest_payload(best_variant.backtest),
      "sizing": asdict(best_variant.sizing),
    },
    "all_variants": [
      {
        "variant_slug": variant.variant_slug,
        "backtest": _backtest_payload(variant.backtest),
        "sizing": asdict(variant.sizing),
      }
      for variant in variants
    ],
  }


def augment_dataset_with_tradeability_predictions(
  *,
  input_root: Path,
  output_root: Path,
  input_dataset_name: str,
  output_dataset_name: str,
  providers: tuple[str, ...],
  symbols: tuple[str, ...],
  prediction_lookup: dict[str, dict[str, str]],
) -> None:
  for series_key in discover_series(
    input_root,
    dataset_name=input_dataset_name,
    providers=providers,
    symbols=symbols,
  ):
    rows = load_series_rows_for_dataset(
      input_root,
      series_key,
      dataset_name=input_dataset_name,
    )
    if not rows:
      continue
    output_rows = []
    for row in rows:
      output_row = dict(row)
      prediction_row = prediction_lookup.get(str(row.get("candle_start", "")), {})
      output_row["tradeability_prediction"] = prediction_row.get("prediction", "")
      output_row["tradeability_prediction_probability_not_tradeable"] = prediction_row.get("prediction_probability_not_tradeable", "")
      output_row["tradeability_prediction_probability_tradeable"] = prediction_row.get("prediction_probability_tradeable", "")
      output_row["tradeability_prediction_confidence"] = prediction_row.get("prediction_confidence", "")
      output_row["tradeability_prediction_probability_max"] = prediction_row.get("prediction_probability_max", "")
      output_row["tradeability_prediction_tradeability_margin"] = prediction_row.get("prediction_tradeability_margin", "")
      output_rows.append(output_row)
    write_partitioned_dataset_rows(
      output_root=output_root,
      dataset_name=output_dataset_name,
      series_key=series_key,
      rows=output_rows,
    )


def load_tradeability_prediction_lookup_from_csv(path: Path) -> dict[str, dict[str, str]]:
  lookup: dict[str, dict[str, str]] = {}
  with path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
      candle_start = str(row.get("candle_start", ""))
      if candle_start == "":
        continue
      lookup[candle_start] = dict(row)
  return lookup


def evaluate_tradeability_candidate_on_exact_holdouts(
  *,
  holdout_root: Path,
  direction_artifact_dir: Path,
  tradeability_candidate: dict[str, Any],
  output_root: Path,
  fee_bps: float,
  slippage_bps: float,
  artifact_search_root: Path,
  supporting_hybrid_transformer_artifact_dir: Path | None,
) -> dict[str, Any]:
  tradeability_artifact_dir = Path(tradeability_candidate["artifact_dir"])
  sizing = PositionSizingConfig(**tradeability_candidate["best_tuning"]["sizing"])
  long_holdout_root = holdout_root / "long_holdout"
  long_result = evaluate_tradeability_candidate_on_existing_holdout(
    holdout_window_root=long_holdout_root,
    direction_artifact_dir=direction_artifact_dir,
    tradeability_artifact_dir=tradeability_artifact_dir,
    output_root=output_root / tradeability_candidate["candidate_slug"] / "long_holdout",
    sizing=sizing,
    fee_bps=fee_bps,
    slippage_bps=slippage_bps,
    artifact_search_root=artifact_search_root,
    supporting_hybrid_transformer_artifact_dir=supporting_hybrid_transformer_artifact_dir,
  )
  rolling_window_results: list[dict[str, Any]] = []
  for window_root in sorted((holdout_root / "rolling_windows").iterdir()):
    if not window_root.is_dir():
      continue
    rolling_window_results.append(
      evaluate_tradeability_candidate_on_existing_holdout(
        holdout_window_root=window_root,
        direction_artifact_dir=direction_artifact_dir,
        tradeability_artifact_dir=tradeability_artifact_dir,
        output_root=output_root / tradeability_candidate["candidate_slug"] / "rolling" / window_root.name,
        sizing=sizing,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        artifact_search_root=artifact_search_root,
        supporting_hybrid_transformer_artifact_dir=supporting_hybrid_transformer_artifact_dir,
      )
    )
  rolling_aggregate = aggregate_model_window_results(
    tradeability_candidate["candidate_slug"],
    [
      {
        "model_name": tradeability_candidate["candidate_slug"],
        "evaluation_rows": result["evaluation_rows"],
        "metrics": {},
        "backtest": result["backtest"],
        "aggregate_metrics": result["aggregate_metrics"],
        "weakness": {
          "primary_weakness": "n/a",
          "secondary_weakness": "n/a",
          "cost_share_of_gross": 0.0,
          "average_resize_rate": 0.0,
        },
      }
      for result in rolling_window_results
    ],
  )
  return {
    "candidate_slug": tradeability_candidate["candidate_slug"],
    "artifact_dir": str(tradeability_artifact_dir),
    "best_tuning": tradeability_candidate["best_tuning"],
    "long_holdout": long_result,
    "rolling_windows": rolling_window_results,
    "rolling_aggregate": rolling_aggregate,
  }


def evaluate_tradeability_candidate_on_existing_holdout(
  *,
  holdout_window_root: Path,
  direction_artifact_dir: Path,
  tradeability_artifact_dir: Path,
  output_root: Path,
  sizing: PositionSizingConfig,
  fee_bps: float,
  slippage_bps: float,
  artifact_search_root: Path,
  supporting_hybrid_transformer_artifact_dir: Path | None,
) -> dict[str, Any]:
  evaluation_artifact_dir = holdout_window_root / "evaluation_artifact"
  evaluation_metadata = _load_json(evaluation_artifact_dir / "metadata.json")
  evaluation_start = _parse_dt(str(evaluation_metadata.get("extra", {}).get("evaluation_start_time", "")))
  evaluation_end = _parse_dt(str(evaluation_metadata.get("extra", {}).get("evaluation_end_time", "")))
  rows = load_series_rows_for_dataset(
    holdout_window_root / "features",
    SeriesKey(
      provider=str(evaluation_metadata["provider"]),
      symbol=str(evaluation_metadata["symbol"]),
    ),
    dataset_name="market_supervised",
  )
  prediction_lookup = score_tradeability_artifact_for_window(
    rows=rows,
    artifact_dir=tradeability_artifact_dir,
    evaluation_start=evaluation_start,
    evaluation_end=evaluation_end,
    artifact_search_root=artifact_search_root,
    supporting_hybrid_transformer_artifact_dir=supporting_hybrid_transformer_artifact_dir,
  )
  dataset_name = f"market_supervised_tradeability_gate_{tradeability_artifact_dir.name}"
  augment_dataset_with_tradeability_predictions(
    input_root=holdout_window_root / "features",
    output_root=holdout_window_root / "features",
    input_dataset_name="market_supervised",
    output_dataset_name=dataset_name,
    providers=(str(evaluation_metadata["provider"]),),
    symbols=(str(evaluation_metadata["symbol"]),),
    prediction_lookup=prediction_lookup,
  )
  backtest = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=evaluation_artifact_dir,
      input_root=holdout_window_root / "features",
      output_root=output_root,
      dataset_name=dataset_name,
      fee_bps=fee_bps,
      slippage_bps=slippage_bps,
      sizing=sizing,
    )
  )
  aggregate_metrics = {}
  if backtest.output_dir is not None:
    aggregate_metrics = _load_json(backtest.output_dir / "summary.json").get("aggregate_metrics", {})
  return {
    "holdout_root": str(holdout_window_root),
    "evaluation_rows": backtest.joined_rows,
    "dataset_name": dataset_name,
    "backtest": _backtest_payload(backtest),
    "aggregate_metrics": aggregate_metrics,
  }


def score_tradeability_artifact_for_window(
  *,
  rows: list[dict[str, str]],
  artifact_dir: Path,
  evaluation_start: datetime,
  evaluation_end: datetime,
  artifact_search_root: Path,
  supporting_hybrid_transformer_artifact_dir: Path | None,
) -> dict[str, dict[str, str]]:
  metadata = _load_json(artifact_dir / "metadata.json")
  if supporting_hybrid_transformer_artifact_dir is not None:
    metadata.setdefault("extra", {})
    metadata["extra"].setdefault("hybrid_transformer_artifact_dir", str(supporting_hybrid_transformer_artifact_dir))
  model = load_model_from_artifact(artifact_dir)
  supporting_artifacts = load_supporting_artifacts_for_live_scoring(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=artifact_search_root,
  )
  ensure_hybrid_transformer_supporting_artifact(
    artifact_dir=artifact_dir,
    metadata=metadata,
    artifact_search_root=artifact_search_root,
    supporting_artifacts=supporting_artifacts,
  )
  closed_rows = [row for row in rows if row.get("is_closed") == "true"]
  lookup: dict[str, dict[str, str]] = {}
  for index, row in enumerate(closed_rows):
    candle_dt = _parse_dt(str(row.get("candle_start", "")))
    if not (evaluation_start <= candle_dt < evaluation_end):
      continue
    history = closed_rows[: index + 1]
    scoring_row, _context = resolve_live_feature_values(
      row=row,
      feature_rows=history,
      metadata=metadata,
      artifact_dir=artifact_dir,
      artifact_search_root=artifact_search_root,
      supporting_artifacts=supporting_artifacts,
    )
    outputs = predict_live_artifact_outputs(
      model=model,
      metadata=metadata,
      row=scoring_row,
      feature_rows=history,
    )
    lookup[str(row.get("candle_start", ""))] = {
      key: str(value)
      for key, value in outputs.items()
    }
  return lookup


def resolve_latest_artifact_dir(root: Path) -> Path:
  if not root.exists():
    raise FileNotFoundError(f"No artifacts found under {root}")
  candidates = sorted(path for path in root.iterdir() if path.is_dir())
  if not candidates:
    raise FileNotFoundError(f"No artifact runs found under {root}")
  return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def _backtest_payload(summary: ModelBacktestSummary) -> dict[str, Any]:
  return {
    "joined_rows": summary.joined_rows,
    "active_periods": summary.active_periods,
    "trades": summary.trades,
    "total_net_return": summary.total_net_return,
    "max_drawdown": summary.max_drawdown,
    "sharpe_like": summary.sharpe_like,
    "total_transaction_cost": summary.total_transaction_cost,
    "total_turnover": summary.total_turnover,
    "active_hit_rate": summary.active_hit_rate,
    "average_net_return_per_trade": summary.average_net_return_per_trade,
    "profit_factor_after_cost": summary.profit_factor_after_cost,
    "after_cost_trade_quality": summary.after_cost_trade_quality,
    "output_dir": str(summary.output_dir) if summary.output_dir else "",
  }


def _default_artifact_search_root(direction_artifact_dir: Path) -> Path:
  current = direction_artifact_dir
  for parent in (current, *current.parents):
    if parent.name == "models":
      return parent
  raise FileNotFoundError(f"Unable to infer artifact search root from {direction_artifact_dir}")


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)

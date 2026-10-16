from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.research.model_weakness_diagnostics import (
  ModelWeaknessDiagnosticsConfig,
  run_model_weakness_diagnostics,
)
from quant.risk.sizing import PositionSizingConfig


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
  window_index: int
  holdout_root: Path
  evaluation_artifact_dir: Path
  model_name: str


@dataclass(frozen=True, slots=True)
class ExecutionVariant:
  slug: str
  sizing: PositionSizingConfig


@dataclass(frozen=True, slots=True)
class ExecutionTuningConfig:
  rolling_holdout_root: Path
  output_root: Path
  model_names: tuple[str, ...]
  top_k: int = 5
  minimum_trade_count: int = 24
  minimum_active_coverage: float = 0.01
  inactivity_trade_penalty: float = 0.01
  inactivity_coverage_penalty: float = 0.01


@dataclass(frozen=True, slots=True)
class ArtifactEvaluation:
  name: str
  artifact_dir: Path
  feature_root: Path
  dataset_name: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactExecutionTuningConfig:
  tuning_evaluation: ArtifactEvaluation
  output_root: Path
  base_sizing: PositionSizingConfig = PositionSizingConfig()
  top_k: int = 5
  holdout_evaluations: tuple[ArtifactEvaluation, ...] = ()
  minimum_trade_count: int = 24
  minimum_active_coverage: float = 0.01
  inactivity_trade_penalty: float = 0.01
  inactivity_coverage_penalty: float = 0.01


def run_execution_tuning(config: ExecutionTuningConfig) -> dict[str, Any]:
  output_root = config.output_root
  output_root.mkdir(parents=True, exist_ok=True)
  summary_path = config.rolling_holdout_root / "summary.json"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))

  model_results: dict[str, Any] = {}
  for model_name in config.model_names:
    windows = load_window_evaluations(summary, model_name)
    if not windows:
      continue
    base_sizing = load_base_sizing(windows[0].evaluation_artifact_dir)
    variants = build_execution_variants(model_name=model_name, base_sizing=base_sizing)
    candidate_summaries = []
    for variant in variants:
      candidate = evaluate_execution_variant(
        windows=windows,
        model_name=model_name,
        variant=variant,
        output_root=output_root / model_name,
      )
      annotate_candidate_selection_score(
        candidate,
        minimum_trade_count=config.minimum_trade_count,
        minimum_active_coverage=config.minimum_active_coverage,
        inactivity_trade_penalty=config.inactivity_trade_penalty,
        inactivity_coverage_penalty=config.inactivity_coverage_penalty,
      )
      candidate_summaries.append(candidate)
    candidate_summaries.sort(
      key=lambda item: (
        float(item["selection_score"]),
        float(item["total_after_cost_trade_quality"]),
        float(item["compounded_net_return"]),
        -float(item["mean_max_drawdown"]),
        -float(item["total_transaction_cost"]),
      ),
      reverse=True,
    )
    top_candidates = candidate_summaries[: max(1, config.top_k)]
    best_candidate = enrich_best_candidate_with_weakness(top_candidates[0])
    top_candidates[0] = best_candidate
    model_results[model_name] = {
      "base_sizing": asdict(base_sizing),
      "variant_count": len(variants),
      "best_candidate": best_candidate,
      "top_candidates": top_candidates,
    }

  payload = {
    "rolling_holdout_root": str(config.rolling_holdout_root),
    "output_root": str(output_root),
    "model_results": model_results,
  }
  (output_root / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return payload


def run_artifact_execution_tuning(config: ArtifactExecutionTuningConfig) -> dict[str, Any]:
  output_root = config.output_root
  output_root.mkdir(parents=True, exist_ok=True)
  metadata = _load_artifact_metadata(config.tuning_evaluation.artifact_dir)
  model_name = str(metadata["model_name"])
  variants = build_artifact_execution_variants(
    model_name=model_name,
    base_sizing=config.base_sizing,
    feature_columns=tuple(metadata.get("feature_columns", ())),
  )

  candidate_summaries = []
  for variant in variants:
    candidate = evaluate_artifact_execution_variant(
      evaluation=config.tuning_evaluation,
      model_name=model_name,
      variant=variant,
      output_root=output_root / "tuning",
    )
    annotate_candidate_selection_score(
      candidate,
      minimum_trade_count=config.minimum_trade_count,
      minimum_active_coverage=config.minimum_active_coverage,
      inactivity_trade_penalty=config.inactivity_trade_penalty,
      inactivity_coverage_penalty=config.inactivity_coverage_penalty,
    )
    candidate_summaries.append(candidate)
  candidate_summaries.sort(
    key=lambda item: (
      float(item["selection_score"]),
      float(item["after_cost_trade_quality"]),
      float(item["net_return"]),
      -float(item["max_drawdown"]),
      -float(item["total_transaction_cost"]),
    ),
    reverse=True,
  )
  top_candidates = candidate_summaries[: max(1, config.top_k)]
  if top_candidates:
    best_candidate = enrich_best_candidate_with_weakness_for_evaluation(
      candidate=top_candidates[0],
      evaluation=config.tuning_evaluation,
    )
    top_candidates[0] = best_candidate
  for candidate in top_candidates:
    candidate["holdout_results"] = {
      evaluation.name: evaluate_artifact_execution_variant(
        evaluation=evaluation,
        model_name=model_name,
        variant=ExecutionVariant(slug=candidate["variant_slug"], sizing=PositionSizingConfig(**candidate["sizing"])),
        output_root=output_root / "holdouts" / evaluation.name,
      )
      for evaluation in config.holdout_evaluations
    }

  payload = {
    "tuning_evaluation": {
      "name": config.tuning_evaluation.name,
      "artifact_dir": str(config.tuning_evaluation.artifact_dir),
      "feature_root": str(config.tuning_evaluation.feature_root),
      "dataset_name": config.tuning_evaluation.dataset_name or metadata.get("extra", {}).get("dataset_name", ""),
    },
    "output_root": str(output_root),
    "model_name": model_name,
    "base_sizing": asdict(config.base_sizing),
    "variant_count": len(variants),
    "selection": {
      "minimum_trade_count": config.minimum_trade_count,
      "minimum_active_coverage": config.minimum_active_coverage,
      "inactivity_trade_penalty": config.inactivity_trade_penalty,
      "inactivity_coverage_penalty": config.inactivity_coverage_penalty,
    },
    "top_candidates": top_candidates,
  }
  (output_root / "summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  return payload


def load_window_evaluations(summary: dict[str, Any], model_name: str) -> list[WindowEvaluation]:
  windows: list[WindowEvaluation] = []
  for window in summary.get("windows", []):
    result = window.get("results", {}).get(model_name)
    if result is None:
      continue
    windows.append(
      WindowEvaluation(
        window_index=int(window["window_index"]),
        holdout_root=Path(window["holdout_root"]),
        evaluation_artifact_dir=Path(result["evaluation_artifact_dir"]),
        model_name=model_name,
      )
    )
  return windows


def load_sizing_from_backtest_config(path: Path) -> PositionSizingConfig:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return PositionSizingConfig(**payload["sizing"])


def load_base_sizing(evaluation_artifact_dir: Path) -> PositionSizingConfig:
  metadata = _load_artifact_metadata(evaluation_artifact_dir)
  payload = metadata.get("extra", {}).get("execution_sizing", {})
  return PositionSizingConfig(**payload)


def build_execution_variants(*, model_name: str, base_sizing: PositionSizingConfig) -> list[ExecutionVariant]:
  variants: list[ExecutionVariant] = []
  seen: set[str] = set()
  if model_name == "histgb_return":
    for threshold in (0.0, 0.1, 0.25, 0.5):
      for scale in (1.0, 2.5, 5.0):
        for min_hold in (0, 1, 3):
          for min_rebalance in (0.0, 0.01, 0.025, 0.05):
            for hold_threshold in (0.0, 0.25, 0.5):
              sizing = replace(
                base_sizing,
                signal_threshold_bps=threshold,
                signal_scale_bps=scale,
                min_hold_bars=min_hold,
                min_rebalance_abs_delta=min_rebalance,
                hold_signal_threshold_bps=hold_threshold,
              )
              _append_variant(variants, seen, slug_for_histgb(sizing), sizing)
  elif model_name == "lightgbm_hybrid_direction_meta":
    trend_profiles = (
      {"trend_filter_enabled": False, "trend_filter_countertrend_scale": 0.0, "trend_filter_regime_bias_bps": 0.0, "trend_filter_threshold_bps": 20.0},
      {"trend_filter_enabled": True, "trend_filter_countertrend_scale": 0.0, "trend_filter_regime_bias_bps": 0.0, "trend_filter_threshold_bps": 20.0},
      {"trend_filter_enabled": True, "trend_filter_countertrend_scale": 0.25, "trend_filter_regime_bias_bps": 0.0, "trend_filter_threshold_bps": 20.0},
      {"trend_filter_enabled": True, "trend_filter_countertrend_scale": 0.0, "trend_filter_regime_bias_bps": 2.5, "trend_filter_threshold_bps": 20.0},
      {"trend_filter_enabled": True, "trend_filter_countertrend_scale": 0.25, "trend_filter_regime_bias_bps": 2.5, "trend_filter_threshold_bps": 20.0},
      {"trend_filter_enabled": True, "trend_filter_countertrend_scale": 0.0, "trend_filter_regime_bias_bps": 5.0, "trend_filter_threshold_bps": 20.0},
    )
    for min_hold in (0, 2, 3, 5):
      for min_rebalance in (0.0, 0.025, 0.05, 0.1):
        for max_position_change in (0.25, 0.5):
          for margin_threshold in (0.1, 0.15, 0.2):
            for flip_margin_threshold in (0.2, 0.35):
              for persistence_bars in (1, 2):
                for profile in trend_profiles:
                  sizing = replace(
                    base_sizing,
                    min_hold_bars=min_hold,
                    min_rebalance_abs_delta=min_rebalance,
                    max_position_change=max_position_change,
                    classification_margin_threshold=margin_threshold,
                    classification_flip_margin_threshold=flip_margin_threshold,
                    class_change_persistence_bars=persistence_bars,
                    **profile,
                  )
                  _append_variant(variants, seen, slug_for_hybrid_direction(sizing), sizing)
  else:
    _append_variant(variants, seen, model_name, base_sizing)
  return variants


def build_artifact_execution_variants(
  *,
  model_name: str,
  base_sizing: PositionSizingConfig,
  feature_columns: tuple[str, ...] = (),
) -> list[ExecutionVariant]:
  feature_columns_set = set(feature_columns)
  if (
    model_name == "lightgbm_hybrid_direction_meta"
    and (
      "oof_fast_trend_state_probability_uptrend" in feature_columns_set
      or "oof_slow_trend_state_probability_uptrend" in feature_columns_set
      or "oof_fast_triple_barrier_probability_profit" in feature_columns_set
    )
  ):
    return build_executable_direction_variants(
      base_sizing=base_sizing,
      feature_columns=feature_columns,
    )
  if model_name == "lightgbm_hybrid_direction_meta":
    return build_focused_hybrid_direction_artifact_variants(
      base_sizing=base_sizing,
      feature_columns=feature_columns,
    )
  return build_execution_variants(model_name=model_name, base_sizing=base_sizing)


def build_focused_hybrid_direction_artifact_variants(
  *,
  base_sizing: PositionSizingConfig,
  feature_columns: tuple[str, ...] = (),
) -> list[ExecutionVariant]:
  variants: list[ExecutionVariant] = []
  seen: set[str] = set()
  feature_columns_set = set(feature_columns)
  has_raw_trend = {
    base_sizing.trend_filter_column,
    base_sizing.trend_filter_higher_timeframe_column,
  }.issubset(feature_columns_set)

  directional_profiles = [
    {
      "name": "plain",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_filter_exit_enabled": False,
      },
    },
  ]
  if has_raw_trend:
    directional_profiles.append(
      {
        "name": "rawtrend",
        "kwargs": {
          "trend_filter_enabled": True,
          "trend_filter_countertrend_scale": 0.0,
          "trend_filter_regime_bias_bps": 0.0,
          "trend_filter_threshold_bps": base_sizing.trend_filter_threshold_bps,
        },
      }
    )

  for profile in directional_profiles:
    for min_hold in (1, 2):
      for min_rebalance in (0.0, 0.025):
        for max_position_change in (0.15, 0.25):
          for margin_threshold in (0.1, 0.12, 0.15):
            for flip_margin_threshold in (0.2, 0.25):
              for persistence_bars in (0, 1):
                sizing = replace(
                  base_sizing,
                  hold_until_stop_or_take_profit=False,
                  profit_take_exit_multiple=0.0,
                  max_hold_bars=0,
                  stale_exit_bars=0,
                  classification_stale_exit_margin_threshold=0.0,
                  trailing_profit_activation_multiple=0.0,
                  trailing_profit_retrace_multiple=0.0,
                  min_hold_bars=min_hold,
                  min_rebalance_abs_delta=min_rebalance,
                  max_position_change=max_position_change,
                  classification_margin_threshold=margin_threshold,
                  classification_flip_margin_threshold=flip_margin_threshold,
                  class_change_persistence_bars=persistence_bars,
                  **profile["kwargs"],
                )
                _append_variant(
                  variants,
                  seen,
                  slug_for_executable_direction(profile["name"], "nolock", sizing),
                  sizing,
                )

  lock_exit_profiles = [
    {
      "name": "lock_xhold4_stale2_pt0p5",
      "kwargs": {
        "max_hold_bars": 4,
        "stale_exit_bars": 2,
        "classification_stale_exit_margin_threshold": 0.0,
        "profit_take_exit_multiple": 0.5,
      },
    },
    {
      "name": "lock_xhold4_stale3_pt0p5",
      "kwargs": {
        "max_hold_bars": 4,
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
        "profit_take_exit_multiple": 0.5,
      },
    },
    {
      "name": "lock_xhold6_stale3_pt0p5",
      "kwargs": {
        "max_hold_bars": 6,
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
        "profit_take_exit_multiple": 0.5,
      },
    },
    {
      "name": "lock_xhold6_stale4_pt1",
      "kwargs": {
        "max_hold_bars": 6,
        "stale_exit_bars": 4,
        "classification_stale_exit_margin_threshold": 0.05,
        "profit_take_exit_multiple": 1.0,
      },
    },
    {
      "name": "lock_xhold8_stale4_pt1",
      "kwargs": {
        "max_hold_bars": 8,
        "stale_exit_bars": 4,
        "classification_stale_exit_margin_threshold": 0.1,
        "profit_take_exit_multiple": 1.0,
      },
    },
  ]

  for profile in directional_profiles:
    profile_kwargs = dict(profile["kwargs"])
    if has_raw_trend and profile["name"] == "rawtrend":
      profile_kwargs["trend_filter_exit_enabled"] = True
    else:
      profile_kwargs["trend_filter_exit_enabled"] = False

    for min_hold in (1, 2):
      for min_rebalance in (0.0, 0.025):
        for margin_threshold in (0.1, 0.12):
          for flip_margin_threshold in (0.2, 0.25):
            for hold_threshold in (0.5, 1.0):
              for exit_profile in lock_exit_profiles:
                sizing = replace(
                  base_sizing,
                  hold_until_stop_or_take_profit=True,
                  trailing_profit_activation_multiple=0.0,
                  trailing_profit_retrace_multiple=0.0,
                  min_hold_bars=min_hold,
                  min_rebalance_abs_delta=min_rebalance,
                  max_position_change=0.15,
                  classification_margin_threshold=margin_threshold,
                  classification_flip_margin_threshold=flip_margin_threshold,
                  class_change_persistence_bars=0,
                  hold_signal_threshold_bps=hold_threshold,
                  **profile_kwargs,
                  **exit_profile["kwargs"],
                )
                _append_variant(
                  variants,
                  seen,
                  slug_for_executable_direction(profile["name"], exit_profile["name"], sizing),
                  sizing,
                )
  return variants


def build_executable_direction_variants(
  *,
  base_sizing: PositionSizingConfig,
  feature_columns: tuple[str, ...] = (),
) -> list[ExecutionVariant]:
  variants: list[ExecutionVariant] = []
  seen: set[str] = set()
  feature_columns_set = set(feature_columns)

  overlay_profiles = [
    {
      "name": "plain",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_state_filter_enabled": False,
      },
    },
    {
      "name": "fast_gate_soft",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_state_filter_enabled": True,
        "trend_state_uptrend_probability_column": "oof_fast_trend_state_probability_uptrend",
        "trend_state_downtrend_probability_column": "oof_fast_trend_state_probability_downtrend",
        "trend_state_chop_probability_column": "oof_fast_trend_state_probability_chop",
        "trend_state_confidence_column": "oof_fast_trend_state_confidence",
        "trend_state_gate_threshold": 0.6,
        "trend_state_countertrend_scale": 0.25,
        "trend_state_chop_threshold": 0.65,
        "trend_state_chop_scale": 0.25,
        "trend_state_min_confidence": 0.05,
      },
      "required_columns": {
        "oof_fast_trend_state_probability_uptrend",
        "oof_fast_trend_state_probability_downtrend",
        "oof_fast_trend_state_probability_chop",
        "oof_fast_trend_state_confidence",
      },
    },
    {
      "name": "fast_gate_strict",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_state_filter_enabled": True,
        "trend_state_uptrend_probability_column": "oof_fast_trend_state_probability_uptrend",
        "trend_state_downtrend_probability_column": "oof_fast_trend_state_probability_downtrend",
        "trend_state_chop_probability_column": "oof_fast_trend_state_probability_chop",
        "trend_state_confidence_column": "oof_fast_trend_state_confidence",
        "trend_state_gate_threshold": 0.65,
        "trend_state_countertrend_scale": 0.0,
        "trend_state_chop_threshold": 0.6,
        "trend_state_chop_scale": 0.0,
        "trend_state_min_confidence": 0.1,
      },
      "required_columns": {
        "oof_fast_trend_state_probability_uptrend",
        "oof_fast_trend_state_probability_downtrend",
        "oof_fast_trend_state_probability_chop",
        "oof_fast_trend_state_confidence",
      },
    },
    {
      "name": "slow_gate_soft",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_state_filter_enabled": True,
        "trend_state_uptrend_probability_column": "oof_slow_trend_state_probability_uptrend",
        "trend_state_downtrend_probability_column": "oof_slow_trend_state_probability_downtrend",
        "trend_state_chop_probability_column": "oof_slow_trend_state_probability_chop",
        "trend_state_confidence_column": "oof_slow_trend_state_confidence",
        "trend_state_gate_threshold": 0.6,
        "trend_state_countertrend_scale": 0.25,
        "trend_state_chop_threshold": 0.7,
        "trend_state_chop_scale": 0.25,
        "trend_state_min_confidence": 0.05,
      },
      "required_columns": {
        "oof_slow_trend_state_probability_uptrend",
        "oof_slow_trend_state_probability_downtrend",
        "oof_slow_trend_state_probability_chop",
        "oof_slow_trend_state_confidence",
      },
    },
    {
      "name": "slow_gate_strict",
      "kwargs": {
        "trend_filter_enabled": False,
        "trend_state_filter_enabled": True,
        "trend_state_uptrend_probability_column": "oof_slow_trend_state_probability_uptrend",
        "trend_state_downtrend_probability_column": "oof_slow_trend_state_probability_downtrend",
        "trend_state_chop_probability_column": "oof_slow_trend_state_probability_chop",
        "trend_state_confidence_column": "oof_slow_trend_state_confidence",
        "trend_state_gate_threshold": 0.65,
        "trend_state_countertrend_scale": 0.0,
        "trend_state_chop_threshold": 0.65,
        "trend_state_chop_scale": 0.0,
        "trend_state_min_confidence": 0.1,
      },
      "required_columns": {
        "oof_slow_trend_state_probability_uptrend",
        "oof_slow_trend_state_probability_downtrend",
        "oof_slow_trend_state_probability_chop",
        "oof_slow_trend_state_confidence",
      },
    },
    {
      "name": "slow_gate_with_rawtrend",
      "kwargs": {
        "trend_filter_enabled": True,
        "trend_filter_countertrend_scale": 0.25,
        "trend_filter_regime_bias_bps": 0.0,
        "trend_filter_threshold_bps": 20.0,
        "trend_state_filter_enabled": True,
        "trend_state_uptrend_probability_column": "oof_slow_trend_state_probability_uptrend",
        "trend_state_downtrend_probability_column": "oof_slow_trend_state_probability_downtrend",
        "trend_state_chop_probability_column": "oof_slow_trend_state_probability_chop",
        "trend_state_confidence_column": "oof_slow_trend_state_confidence",
        "trend_state_gate_threshold": 0.6,
        "trend_state_countertrend_scale": 0.0,
        "trend_state_chop_threshold": 0.65,
        "trend_state_chop_scale": 0.25,
        "trend_state_min_confidence": 0.05,
      },
      "required_columns": {
        "oof_slow_trend_state_probability_uptrend",
        "oof_slow_trend_state_probability_downtrend",
        "oof_slow_trend_state_probability_chop",
        "oof_slow_trend_state_confidence",
      },
    },
  ]

  available_profiles = []
  for profile in overlay_profiles:
    required_columns = set(profile.get("required_columns", set()))
    if required_columns and not required_columns.issubset(feature_columns_set):
      continue
    available_profiles.append(profile)

  preferred_profile_names = (
    "plain",
    "slow_gate_soft",
    "slow_gate_strict",
    "slow_gate_with_rawtrend",
    "fast_gate_soft",
    "fast_gate_strict",
  )
  preferred_profiles = [
    profile
    for profile in available_profiles
    if profile["name"] in preferred_profile_names
  ]
  if preferred_profiles:
    available_profiles = preferred_profiles

  exit_profiles = [
    {"name": "exit_none", "kwargs": {}},
    {
      "name": "exit_stale3_soft",
      "kwargs": {
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
      },
    },
    {
      "name": "exit_stale4_soft",
      "kwargs": {
        "stale_exit_bars": 4,
        "classification_stale_exit_margin_threshold": 0.05,
      },
    },
    {
      "name": "exit_stale3_strict",
      "kwargs": {
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.1,
      },
    },
    {
      "name": "exit_hard4",
      "kwargs": {
        "max_hold_bars": 4,
      },
    },
    {
      "name": "exit_hard5",
      "kwargs": {
        "max_hold_bars": 5,
      },
    },
    {
      "name": "exit_hard4_stale3",
      "kwargs": {
        "max_hold_bars": 4,
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
      },
    },
    {
      "name": "exit_hard5_stale4",
      "kwargs": {
        "max_hold_bars": 5,
        "stale_exit_bars": 4,
        "classification_stale_exit_margin_threshold": 0.05,
      },
    },
    {
      "name": "exit_trend_stale3",
      "kwargs": {
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
        "trend_state_exit_enabled": True,
        "trend_state_exit_gate_threshold": 0.75,
        "trend_state_chop_exit_threshold": 0.8,
      },
    },
    {
      "name": "exit_trend_hard5",
      "kwargs": {
        "max_hold_bars": 5,
        "stale_exit_bars": 3,
        "classification_stale_exit_margin_threshold": 0.05,
        "trend_state_exit_enabled": True,
        "trend_state_exit_gate_threshold": 0.75,
        "trend_state_chop_exit_threshold": 0.8,
      },
    },
    {
      "name": "exit_profit1p5",
      "kwargs": {
        "profit_take_exit_multiple": 1.5,
      },
    },
    {
      "name": "exit_profit2",
      "kwargs": {
        "profit_take_exit_multiple": 2.0,
      },
    },
    {
      "name": "exit_trail1p5_r1",
      "kwargs": {
        "trailing_profit_activation_multiple": 1.5,
        "trailing_profit_retrace_multiple": 1.0,
      },
    },
    {
      "name": "exit_trail2_r1",
      "kwargs": {
        "trailing_profit_activation_multiple": 2.0,
        "trailing_profit_retrace_multiple": 1.0,
      },
    },
    {
      "name": "exit_hard5_trail1p5_r1",
      "kwargs": {
        "max_hold_bars": 5,
        "trailing_profit_activation_multiple": 1.5,
        "trailing_profit_retrace_multiple": 1.0,
      },
    },
  ]

  for min_hold in (2, 4, 6):
    for min_rebalance in (0.025, 0.05):
      for max_position_change in (0.15,):
        for margin_threshold in (0.15, 0.2):
          for flip_margin_threshold in (0.25,):
            for persistence_bars in (1,):
              for profile in available_profiles:
                for exit_profile in exit_profiles:
                  sizing = replace(
                    base_sizing,
                    min_hold_bars=min_hold,
                    min_rebalance_abs_delta=min_rebalance,
                    max_position_change=max_position_change,
                    classification_margin_threshold=margin_threshold,
                    classification_flip_margin_threshold=flip_margin_threshold,
                    class_change_persistence_bars=persistence_bars,
                    **profile["kwargs"],
                    **exit_profile["kwargs"],
                  )
                  _append_variant(
                    variants,
                    seen,
                    slug_for_executable_direction(profile["name"], exit_profile["name"], sizing),
                    sizing,
                  )
  return variants


def evaluate_execution_variant(
  *,
  windows: list[WindowEvaluation],
  model_name: str,
  variant: ExecutionVariant,
  output_root: Path,
) -> dict[str, Any]:
  per_window: list[dict[str, Any]] = []
  compounded = 1.0
  total_gross = 0.0
  total_cost = 0.0
  total_turnover = 0.0
  total_after_cost_trade_quality = 0.0
  total_trades = 0
  total_active_periods = 0
  total_rows = 0

  for window in windows:
    metadata = _load_artifact_metadata(window.evaluation_artifact_dir)
    dataset_name = str(metadata.get("extra", {}).get("dataset_name", "market_supervised"))
    summary = run_model_backtest(
      ModelBacktestConfig(
        artifact_dir=window.evaluation_artifact_dir,
        input_root=window.holdout_root / "features",
        output_root=output_root / variant.slug / f"window_{window.window_index:02d}",
        fee_bps=1.0,
        slippage_bps=0.5,
        sizing=variant.sizing,
      )
    )
    backtest_summary = json.loads((summary.output_dir / "summary.json").read_text(encoding="utf-8"))
    aggregate_metrics = backtest_summary.get("aggregate_metrics", {})
    net_return = float(summary.total_net_return)
    compounded *= 1.0 + net_return
    total_gross += float(aggregate_metrics.get("total_gross_return", 0.0))
    total_cost += float(aggregate_metrics.get("total_transaction_cost", 0.0))
    total_turnover += float(aggregate_metrics.get("total_turnover", 0.0))
    total_after_cost_trade_quality += float(aggregate_metrics.get("after_cost_trade_quality", 0.0))
    total_trades += int(summary.trades)
    total_active_periods += int(summary.active_periods)
    total_rows += int(summary.joined_rows)
    per_window.append(
      {
        "window_index": window.window_index,
        "output_dir": str(summary.output_dir) if summary.output_dir else "",
        "feature_root": str(window.holdout_root / "features"),
        "evaluation_artifact_dir": str(window.evaluation_artifact_dir),
        "dataset_name": dataset_name,
        "provider": str(metadata.get("provider", "")),
        "symbol": str(metadata.get("symbol", "")),
        "joined_rows": summary.joined_rows,
        "active_periods": summary.active_periods,
        "trades": summary.trades,
        "net_return": summary.total_net_return,
        "max_drawdown": summary.max_drawdown,
        "gross_return": float(aggregate_metrics.get("total_gross_return", 0.0)),
        "transaction_cost": float(aggregate_metrics.get("total_transaction_cost", 0.0)),
        "turnover": float(aggregate_metrics.get("total_turnover", 0.0)),
        "after_cost_trade_quality": float(aggregate_metrics.get("after_cost_trade_quality", 0.0)),
      }
    )

  net_returns = [float(item["net_return"]) for item in per_window]
  max_drawdowns = [float(item["max_drawdown"]) for item in per_window]
  trade_quality_scores = [float(item["after_cost_trade_quality"]) for item in per_window]
  cost_share = (total_cost / total_gross) if total_gross > 0 else 0.0
  return {
    "model_name": model_name,
    "variant_slug": variant.slug,
    "sizing": asdict(variant.sizing),
    "window_count": len(per_window),
    "flat_windows": sum(1 for item in per_window if abs(float(item["net_return"])) < 1e-12),
    "positive_windows": sum(1 for item in per_window if float(item["net_return"]) > 0),
    "negative_windows": sum(1 for item in per_window if float(item["net_return"]) < 0),
    "compounded_net_return": compounded - 1.0,
    "mean_net_return": _mean(net_returns),
    "median_net_return": _median(net_returns),
    "mean_max_drawdown": _mean(max_drawdowns),
    "max_window_drawdown": max(max_drawdowns) if max_drawdowns else 0.0,
    "total_trades": total_trades,
    "mean_trades": total_trades / len(per_window) if per_window else 0.0,
    "total_gross_return": total_gross,
    "total_transaction_cost": total_cost,
    "total_after_cost_trade_quality": total_after_cost_trade_quality,
    "mean_after_cost_trade_quality": _mean(trade_quality_scores),
    "cost_share_of_gross_total": cost_share,
    "total_turnover": total_turnover,
    "mean_turnover_per_period": total_turnover / total_active_periods if total_active_periods else 0.0,
    "total_active_periods": total_active_periods,
    "total_evaluation_rows": total_rows,
    "per_window": per_window,
  }


def evaluate_artifact_execution_variant(
  *,
  evaluation: ArtifactEvaluation,
  model_name: str,
  variant: ExecutionVariant,
  output_root: Path,
) -> dict[str, Any]:
  summary = run_model_backtest(
    ModelBacktestConfig(
      artifact_dir=evaluation.artifact_dir,
      input_root=evaluation.feature_root,
      output_root=output_root / variant.slug,
      dataset_name=evaluation.dataset_name,
      fee_bps=1.0,
      slippage_bps=0.5,
      sizing=variant.sizing,
    )
  )
  backtest_summary = json.loads((summary.output_dir / "summary.json").read_text(encoding="utf-8"))
  aggregate_metrics = backtest_summary.get("aggregate_metrics", {})
  return {
    "evaluation_name": evaluation.name,
    "model_name": model_name,
    "variant_slug": variant.slug,
    "sizing": asdict(variant.sizing),
    "artifact_dir": str(evaluation.artifact_dir),
    "feature_root": str(evaluation.feature_root),
    "dataset_name": evaluation.dataset_name or backtest_summary.get("dataset_name", ""),
    "output_dir": str(summary.output_dir) if summary.output_dir else "",
    "joined_rows": summary.joined_rows,
    "active_periods": summary.active_periods,
    "trades": summary.trades,
    "net_return": summary.total_net_return,
    "max_drawdown": summary.max_drawdown,
    "total_transaction_cost": summary.total_transaction_cost,
    "total_turnover": summary.total_turnover,
    "average_net_return_per_trade": summary.average_net_return_per_trade,
    "profit_factor_after_cost": summary.profit_factor_after_cost,
    "after_cost_trade_quality": summary.after_cost_trade_quality,
    "aggregate_metrics": aggregate_metrics,
  }


def annotate_candidate_selection_score(
  candidate: dict[str, Any],
  *,
  minimum_trade_count: int,
  minimum_active_coverage: float,
  inactivity_trade_penalty: float,
  inactivity_coverage_penalty: float,
) -> dict[str, Any]:
  trades = int(candidate.get("trades", candidate.get("total_trades", 0)))
  active_periods = int(candidate.get("active_periods", candidate.get("total_active_periods", 0)))
  joined_rows = int(candidate.get("joined_rows", candidate.get("total_evaluation_rows", 0)))
  after_cost_trade_quality = float(
    candidate.get("after_cost_trade_quality", candidate.get("total_after_cost_trade_quality", 0.0))
  )
  active_coverage = (active_periods / joined_rows) if joined_rows else 0.0
  missing_trade_ratio = (
    max(0, minimum_trade_count - trades) / minimum_trade_count
    if minimum_trade_count > 0
    else 0.0
  )
  missing_coverage_ratio = (
    max(0.0, minimum_active_coverage - active_coverage) / minimum_active_coverage
    if minimum_active_coverage > 0
    else 0.0
  )
  inactivity_penalty = (
    inactivity_trade_penalty * missing_trade_ratio
    + inactivity_coverage_penalty * missing_coverage_ratio
  )
  candidate["active_coverage"] = active_coverage
  candidate["missing_trade_ratio"] = missing_trade_ratio
  candidate["missing_active_coverage_ratio"] = missing_coverage_ratio
  candidate["inactivity_penalty"] = inactivity_penalty
  candidate["selection_score"] = after_cost_trade_quality - inactivity_penalty
  return candidate


def enrich_best_candidate_with_weakness(candidate: dict[str, Any]) -> dict[str, Any]:
  enriched = dict(candidate)
  weakness_counts: dict[str, int] = {}
  secondary_counts: dict[str, int] = {}
  for window in enriched.get("per_window", []):
    backtest_dir = Path(window["output_dir"])
    weakness = run_model_weakness_diagnostics(
        ModelWeaknessDiagnosticsConfig(
          backtest_dir=backtest_dir,
          feature_root=Path(window["feature_root"]),
          provider=str(window.get("provider", "")),
          symbol=str(window.get("symbol", "")),
          dataset_name=str(window.get("dataset_name", "market_supervised") or "market_supervised"),
          trend_column="trend_return_24_bar",
          trend_threshold_bps=20.0,
        )
      )
    primary = str(weakness.primary_weakness)
    secondary = str(weakness.secondary_weakness)
    if primary:
      weakness_counts[primary] = weakness_counts.get(primary, 0) + 1
    if secondary:
      secondary_counts[secondary] = secondary_counts.get(secondary, 0) + 1
    window["weakness"] = {
      "primary_weakness": primary,
      "secondary_weakness": secondary,
      "cost_share_of_gross": weakness.cost_share_of_gross,
      "average_resize_rate": weakness.average_resize_rate,
    }
  enriched["primary_weakness_counts"] = weakness_counts
  enriched["secondary_weakness_counts"] = secondary_counts
  return enriched


def enrich_best_candidate_with_weakness_for_evaluation(
  *,
  candidate: dict[str, Any],
  evaluation: ArtifactEvaluation,
) -> dict[str, Any]:
  enriched = dict(candidate)
  metadata = _load_artifact_metadata(evaluation.artifact_dir)
  weakness = run_model_weakness_diagnostics(
    ModelWeaknessDiagnosticsConfig(
      backtest_dir=Path(candidate["output_dir"]),
      feature_root=evaluation.feature_root,
      provider=str(metadata.get("provider", "")),
      symbol=str(metadata.get("symbol", "")),
      dataset_name=evaluation.dataset_name or str(metadata.get("extra", {}).get("dataset_name", "market_supervised")),
      trend_column="trend_return_24_bar",
      trend_threshold_bps=20.0,
    )
  )
  enriched["weakness"] = {
    "primary_weakness": weakness.primary_weakness,
    "secondary_weakness": weakness.secondary_weakness,
    "cost_share_of_gross": weakness.cost_share_of_gross,
    "average_resize_rate": weakness.average_resize_rate,
  }
  return enriched


def slug_for_histgb(sizing: PositionSizingConfig) -> str:
  return (
    f"thr{sizing.signal_threshold_bps:g}_"
    f"scale{sizing.signal_scale_bps:g}_"
    f"hold{sizing.min_hold_bars}_"
    f"keep{sizing.hold_signal_threshold_bps:g}_"
    f"rebal{sizing.min_rebalance_abs_delta:g}"
  )


def slug_for_hybrid_direction(sizing: PositionSizingConfig) -> str:
  trend = "trend" if sizing.trend_filter_enabled else "plain"
  return (
    f"{trend}_"
    f"hold{sizing.min_hold_bars}_"
    f"rebal{sizing.min_rebalance_abs_delta:g}_"
    f"mpc{sizing.max_position_change:g}_"
    f"mthr{sizing.classification_margin_threshold:g}_"
    f"flipm{sizing.classification_flip_margin_threshold:g}_"
    f"persist{sizing.class_change_persistence_bars}_"
    f"cts{sizing.trend_filter_countertrend_scale:g}_"
    f"bias{sizing.trend_filter_regime_bias_bps:g}"
  )


def slug_for_executable_direction(profile_name: str, exit_name: str, sizing: PositionSizingConfig) -> str:
  return (
    f"{profile_name}_"
    f"{exit_name}_"
    f"hold{sizing.min_hold_bars}_"
    f"rebal{sizing.min_rebalance_abs_delta:g}_"
    f"mpc{sizing.max_position_change:g}_"
    f"mthr{sizing.classification_margin_threshold:g}_"
    f"flipm{sizing.classification_flip_margin_threshold:g}_"
    f"persist{sizing.class_change_persistence_bars}_"
    f"xhold{sizing.max_hold_bars}_"
    f"stale{sizing.stale_exit_bars}x{sizing.classification_stale_exit_margin_threshold:g}_"
    f"pt{sizing.profit_take_exit_multiple:g}_"
    f"trail{sizing.trailing_profit_activation_multiple:g}x{sizing.trailing_profit_retrace_multiple:g}_"
    f"texit{int(sizing.trend_state_exit_enabled)}"
  )


def _append_variant(
  variants: list[ExecutionVariant],
  seen: set[str],
  slug: str,
  sizing: PositionSizingConfig,
) -> None:
  if slug in seen:
    return
  seen.add(slug)
  variants.append(ExecutionVariant(slug=slug, sizing=sizing))


def _mean(values: list[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
  if not values:
    return 0.0
  ordered = sorted(values)
  middle = len(ordered) // 2
  if len(ordered) % 2:
    return ordered[middle]
  return (ordered[middle - 1] + ordered[middle]) / 2.0


def _load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
  return json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from quant.research.oracle_profit_bound import OracleConstraintProfile
from quant.research.oracle_tradeability_experiment import (
  OracleTradeabilityExperimentConfig,
  TradeabilityModelParamSet,
  run_oracle_tradeability_experiment,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  config = OracleTradeabilityExperimentConfig(
    feature_root=Path(config_value(args.feature_root, payload, "feature_root")),
    direction_artifact_dir=Path(config_value(args.direction_artifact_dir, payload, "direction_artifact_dir")),
    base_execution_config_path=Path(config_value(args.base_execution_config_path, payload, "base_execution_config_path")),
    holdout_root=Path(config_value(args.holdout_root, payload, "holdout_root")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/oracle_tradeability_experiment")),
    source_dataset_name=str(config_value(args.source_dataset_name, payload, "source_dataset_name", "market_hybrid_stacked_meta_only_zones")),
    providers=tuple(config_value(args.providers, payload, "providers", ("coinbase",))),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ("BTC-USD",))),
    oracle_profiles=_oracle_profiles(payload),
    model_param_sets=_model_param_sets(payload),
    tradeability_gate_thresholds=tuple(config_value(args.tradeability_gate_thresholds, payload, "tradeability_gate_thresholds", (0.5, 0.55, 0.6, 0.65, 0.7))),
    tradeability_gate_min_confidences=tuple(config_value(args.tradeability_gate_min_confidences, payload, "tradeability_gate_min_confidences", (0.0, 0.05, 0.1))),
    validation_fraction=float(config_value(args.validation_fraction, payload, "validation_fraction", 0.2)),
    minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 512)),
    sample_weight_scheme=str(config_value(args.sample_weight_scheme, payload, "sample_weight_scheme", "hmm_regime_target_balance")),
    regime_change_boost=float(config_value(args.regime_change_boost, payload, "regime_change_boost", 1.25)),
    regime_change_threshold=float(config_value(args.regime_change_threshold, payload, "regime_change_threshold", 0.35)),
    probability_calibration_method=str(config_value(args.probability_calibration_method, payload, "probability_calibration_method", "")),
    top_k_holdout_candidates=int(config_value(args.top_k_holdout_candidates, payload, "top_k_holdout_candidates", 5)),
    oracle_label_mode=str(config_value(args.oracle_label_mode, payload, "oracle_label_mode", "entry_action")),
    baseline_tradeability_artifact_dirs=tuple(Path(path) for path in config_value(args.baseline_tradeability_artifact_dirs, payload, "baseline_tradeability_artifact_dirs", ())),
    artifact_search_root=Path(config_value(args.artifact_search_root, payload, "artifact_search_root")) if config_value(args.artifact_search_root, payload, "artifact_search_root", "") else None,
    supporting_hybrid_transformer_artifact_dir=(
      Path(config_value(args.supporting_hybrid_transformer_artifact_dir, payload, "supporting_hybrid_transformer_artifact_dir"))
      if config_value(args.supporting_hybrid_transformer_artifact_dir, payload, "supporting_hybrid_transformer_artifact_dir", "")
      else None
    ),
  )
  summary = run_oracle_tradeability_experiment(config)
  logging.getLogger(__name__).info(
    "Oracle tradeability experiment complete. candidates=%s output_root=%s",
    len(summary.get("trained_candidates", ())),
    summary.get("output_root", ""),
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Train and evaluate oracle-derived tradeability gate models.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--feature-root")
  parser.add_argument("--direction-artifact-dir")
  parser.add_argument("--base-execution-config-path")
  parser.add_argument("--holdout-root")
  parser.add_argument("--output-root")
  parser.add_argument("--source-dataset-name")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--baseline-tradeability-artifact-dir", dest="baseline_tradeability_artifact_dirs", action="append")
  parser.add_argument("--tradeability-gate-threshold", dest="tradeability_gate_thresholds", action="append", type=float)
  parser.add_argument("--tradeability-gate-min-confidence", dest="tradeability_gate_min_confidences", action="append", type=float)
  parser.add_argument("--validation-fraction", type=float)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--sample-weight-scheme")
  parser.add_argument("--regime-change-boost", type=float)
  parser.add_argument("--regime-change-threshold", type=float)
  parser.add_argument("--probability-calibration-method")
  parser.add_argument("--top-k-holdout-candidates", type=int)
  parser.add_argument("--oracle-label-mode")
  parser.add_argument("--artifact-search-root")
  parser.add_argument("--supporting-hybrid-transformer-artifact-dir")
  parser.add_argument("--log-level", default="INFO")
  return parser


def _oracle_profiles(payload: dict[str, Any]) -> tuple[OracleConstraintProfile, ...]:
  profiles = payload.get("oracle_profiles", ())
  if not profiles:
    return OracleTradeabilityExperimentConfig.__dataclass_fields__["oracle_profiles"].default  # type: ignore[attr-defined]
  return tuple(
    OracleConstraintProfile(
      label=str(item["label"]),
      min_hold_bars=int(item.get("min_hold_bars", 0)),
      max_position_change=float(item.get("max_position_change", 2.0)),
      max_trades_per_day=int(item.get("max_trades_per_day", 0)),
    )
    for item in profiles
  )


def _model_param_sets(payload: dict[str, Any]) -> tuple[TradeabilityModelParamSet, ...]:
  param_sets = payload.get("model_param_sets", ())
  if not param_sets:
    return OracleTradeabilityExperimentConfig.__dataclass_fields__["model_param_sets"].default  # type: ignore[attr-defined]
  return tuple(
    TradeabilityModelParamSet(
      slug=str(item["slug"]),
      params=dict(item.get("params", {})),
    )
    for item in param_sets
  )


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


if __name__ == "__main__":
  raise SystemExit(main())

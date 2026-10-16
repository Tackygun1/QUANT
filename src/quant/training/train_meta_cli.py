from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.training.train_meta import MetaTrainingConfig, train_meta_models
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  model_names = tuple(config_value(args.models, payload, "model_names", ()))
  if not model_names:
    model_names = ("lightgbm_return_meta", "lightgbm_direction_meta")

  config = MetaTrainingConfig(
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "data/models")),
    providers=tuple(config_value(args.providers, payload, "providers", ())),
    symbols=tuple(config_value(args.symbols, payload, "symbols", ())),
    model_names=model_names,
    validation_fraction=float(config_value(args.validation_fraction, payload, "validation_fraction", 0.25)),
    minimum_rows=int(config_value(args.minimum_rows, payload, "minimum_rows", 64)),
    dataset_name=payload.get("dataset_name", "market_stacked"),
    target_overrides=payload.get("target_overrides"),
    model_params=payload.get("model_params"),
    feature_groups=tuple(config_value(args.feature_groups, payload, "feature_groups", ())),
    exclude_feature_groups=tuple(config_value(args.exclude_feature_groups, payload, "exclude_feature_groups", ())),
    feature_prefixes=tuple(config_value(args.feature_prefixes, payload, "feature_prefixes", ())),
    exclude_feature_prefixes=tuple(config_value(args.exclude_feature_prefixes, payload, "exclude_feature_prefixes", ())),
    include_feature_columns=tuple(config_value(args.include_feature_columns, payload, "include_feature_columns", ())),
    exclude_feature_columns=tuple(config_value(args.exclude_feature_columns, payload, "exclude_feature_columns", ())),
    sample_weight_scheme=str(config_value(args.sample_weight_scheme, payload, "sample_weight_scheme", "")),
    trend_column=str(config_value(args.trend_column, payload, "trend_column", "trend_return_24_bar")),
    trend_threshold_bps=float(args.trend_threshold_bps if args.trend_threshold_bps is not None else payload.get("trend_threshold_bps", 20.0)),
    regime_column=str(config_value(args.regime_column, payload, "regime_column", "regime_hmm_state_index")),
    regime_change_column=str(config_value(args.regime_change_column, payload, "regime_change_column", "regime_bocpd_change_prob")),
    regime_change_threshold=float(args.regime_change_threshold if args.regime_change_threshold is not None else payload.get("regime_change_threshold", 0.35)),
    regime_change_boost=float(args.regime_change_boost if args.regime_change_boost is not None else payload.get("regime_change_boost", 1.0)),
    sample_weight_floor=float(args.sample_weight_floor if args.sample_weight_floor is not None else payload.get("sample_weight_floor", 0.5)),
    sample_weight_cap=float(args.sample_weight_cap if args.sample_weight_cap is not None else payload.get("sample_weight_cap", 3.0)),
    probability_calibration_method=str(config_value(args.probability_calibration_method, payload, "probability_calibration_method", "temperature")),
    probability_calibration_fraction=float(
      args.probability_calibration_fraction
      if args.probability_calibration_fraction is not None
      else payload.get("probability_calibration_fraction", 0.1)
    ),
    minimum_calibration_rows=int(
      args.minimum_calibration_rows
      if args.minimum_calibration_rows is not None
      else payload.get("minimum_calibration_rows", 128)
    ),
    calibration_model_names=tuple(config_value(args.calibration_model_names, payload, "calibration_model_names", ("lightgbm_hybrid_direction_meta",))),
    config_path=args.config,
  )

  summary = train_meta_models(config)
  logging.getLogger(__name__).info(
    "Meta-model training complete. runs_attempted=%s runs_completed=%s artifacts_written=%s",
    summary.runs_attempted,
    summary.runs_completed,
    summary.artifacts_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Train stacked LightGBM meta-models on the stacked training dataset.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--provider", dest="providers", action="append")
  parser.add_argument("--symbol", dest="symbols", action="append")
  parser.add_argument("--model", dest="models", action="append")
  parser.add_argument("--feature-group", dest="feature_groups", action="append")
  parser.add_argument("--exclude-feature-group", dest="exclude_feature_groups", action="append")
  parser.add_argument("--feature-prefix", dest="feature_prefixes", action="append")
  parser.add_argument("--exclude-feature-prefix", dest="exclude_feature_prefixes", action="append")
  parser.add_argument("--feature-column", dest="include_feature_columns", action="append")
  parser.add_argument("--exclude-feature-column", dest="exclude_feature_columns", action="append")
  parser.add_argument("--sample-weight-scheme")
  parser.add_argument("--trend-column")
  parser.add_argument("--trend-threshold-bps", type=float)
  parser.add_argument("--regime-column")
  parser.add_argument("--regime-change-column")
  parser.add_argument("--regime-change-threshold", type=float)
  parser.add_argument("--regime-change-boost", type=float)
  parser.add_argument("--sample-weight-floor", type=float)
  parser.add_argument("--sample-weight-cap", type=float)
  parser.add_argument("--probability-calibration-method")
  parser.add_argument("--probability-calibration-fraction", type=float)
  parser.add_argument("--minimum-calibration-rows", type=int)
  parser.add_argument("--calibration-model", dest="calibration_model_names", action="append")
  parser.add_argument("--validation-fraction", type=float)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


if __name__ == "__main__":
  raise SystemExit(main())

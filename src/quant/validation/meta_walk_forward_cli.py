from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.utils.config import load_json_config
from quant.validation.meta_walk_forward import MetaWalkForwardConfig, evaluate_meta_walk_forward


def main() -> int:
  args = _build_parser().parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  payload = load_json_config(args.config)
  config = MetaWalkForwardConfig(
    input_root=Path(payload["input_root"]),
    output_root=Path(payload["output_root"]),
    dataset_name=str(payload["dataset_name"]),
    providers=tuple(payload.get("providers", ())),
    symbols=tuple(payload.get("symbols", ())),
    model_names=tuple(payload.get("model_names", ("lightgbm_hybrid_direction_meta",))),
    initial_train_rows=int(payload.get("initial_train_rows", 8192)),
    validation_rows=int(payload.get("validation_rows", 1024)),
    step_rows=int(payload["step_rows"]) if payload.get("step_rows") is not None else None,
    embargo_rows=int(payload.get("embargo_rows", 1)),
    purge_rows=int(payload["purge_rows"]) if payload.get("purge_rows") is not None else None,
    minimum_rows=int(payload.get("minimum_rows", 9217)),
    target_overrides=payload.get("target_overrides"),
    model_params=payload.get("model_params"),
    feature_groups=tuple(payload.get("feature_groups", ())),
    exclude_feature_groups=tuple(payload.get("exclude_feature_groups", ())),
    feature_prefixes=tuple(payload.get("feature_prefixes", ())),
    exclude_feature_prefixes=tuple(payload.get("exclude_feature_prefixes", ())),
    include_feature_columns=tuple(payload.get("include_feature_columns", ())),
    exclude_feature_columns=tuple(payload.get("exclude_feature_columns", ())),
    sample_weight_scheme=str(payload.get("sample_weight_scheme", "")),
    trend_column=str(payload.get("trend_column", "trend_return_24_bar")),
    trend_threshold_bps=float(payload.get("trend_threshold_bps", 20.0)),
    regime_column=str(payload.get("regime_column", "regime_hmm_state_index")),
    regime_change_column=str(payload.get("regime_change_column", "regime_bocpd_change_prob")),
    regime_change_threshold=float(payload.get("regime_change_threshold", 0.35)),
    regime_change_boost=float(payload.get("regime_change_boost", 1.0)),
    sample_weight_floor=float(payload.get("sample_weight_floor", 0.5)),
    sample_weight_cap=float(payload.get("sample_weight_cap", 3.0)),
    probability_calibration_method=str(payload.get("probability_calibration_method", "temperature")),
    probability_calibration_fraction=float(payload.get("probability_calibration_fraction", 0.1)),
    minimum_calibration_rows=int(payload.get("minimum_calibration_rows", 128)),
    calibration_model_names=tuple(payload.get("calibration_model_names", ("lightgbm_hybrid_direction_meta",))),
    config_path=args.config,
  )
  summary = evaluate_meta_walk_forward(config)
  logging.getLogger(__name__).info(
    "Meta OOF evaluation complete. runs=%s folds=%s predictions=%s outputs=%s",
    summary.runs_completed,
    summary.folds_completed,
    summary.prediction_rows,
    [str(path) for path in summary.output_dirs],
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run purged expanding-window OOF evaluation for stacked meta-models.")
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--log-level", default="INFO")
  return parser


if __name__ == "__main__":
  raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.trend_recognition_comparison import (
  TrendRecognitionComparisonConfig,
  run_trend_recognition_comparison,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  config = TrendRecognitionComparisonConfig(
    feature_root=Path(config_value(args.feature_root, payload, "feature_root")),
    baseline_prediction_path=Path(config_value(args.baseline_prediction_path, payload, "baseline_prediction_path")),
    trend_prediction_path=Path(config_value(args.trend_prediction_path, payload, "trend_prediction_path")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/trend_recognition")),
    provider=str(config_value(args.provider, payload, "provider", "coinbase")),
    symbol=str(config_value(args.symbol, payload, "symbol", "BTC-USD")),
    dataset_name=str(config_value(args.dataset_name, payload, "dataset_name", "market_supervised")),
    target_column=str(config_value(args.target_column, payload, "target_column", "future_trend_state_label_24_bar")),
    regime_change_column=str(config_value(args.regime_change_column, payload, "regime_change_column", "regime_bocpd_change_prob")),
    regime_change_threshold=float(config_value(args.regime_change_threshold, payload, "regime_change_threshold", 0.35)),
    baseline_return_prediction_column=str(config_value(args.baseline_return_prediction_column, payload, "baseline_return_prediction_column", "prediction_return_q50_24_bar")),
    trend_return_prediction_column=str(config_value(args.trend_return_prediction_column, payload, "trend_return_prediction_column", "prediction_return_q50_24_bar")),
    trend_state_prediction_column=str(config_value(args.trend_state_prediction_column, payload, "trend_state_prediction_column", "prediction_trend_state_24_bar")),
    uptrend_probability_column=str(config_value(args.uptrend_probability_column, payload, "uptrend_probability_column", "prediction_probability_uptrend_24_bar")),
    downtrend_probability_column=str(config_value(args.downtrend_probability_column, payload, "downtrend_probability_column", "prediction_probability_downtrend_24_bar")),
    chop_probability_column=str(config_value(args.chop_probability_column, payload, "chop_probability_column", "prediction_probability_chop_24_bar")),
    gate_thresholds=tuple(config_value(args.gate_thresholds, payload, "gate_thresholds", (0.55, 0.6, 0.65))),
    chop_thresholds=tuple(config_value(args.chop_thresholds, payload, "chop_thresholds", (0.55, 0.6, 0.7))),
    countertrend_scales=tuple(config_value(args.countertrend_scales, payload, "countertrend_scales", (0.0, 0.15, 0.25, 0.5))),
    calibration_fraction=float(config_value(args.calibration_fraction, payload, "calibration_fraction", 0.5)),
    minimum_calibration_rows=int(config_value(args.minimum_calibration_rows, payload, "minimum_calibration_rows", 128)),
    probability_calibration_method=str(config_value(args.probability_calibration_method, payload, "probability_calibration_method", "none")),
    optimization_weights=_optimization_weights(args.optimization_weights, payload),
    max_accuracy_drop=float(config_value(args.max_accuracy_drop, payload, "max_accuracy_drop", 0.03)),
    max_trend_direction_accuracy_drop=float(config_value(args.max_trend_direction_accuracy_drop, payload, "max_trend_direction_accuracy_drop", 0.075)),
    constraint_penalty_weight=float(config_value(args.constraint_penalty_weight, payload, "constraint_penalty_weight", 2.0)),
  )

  summary = run_trend_recognition_comparison(config)
  logging.getLogger(__name__).info(
    "Trend recognition comparison complete. common_rows=%s best_profile=%s calibrated_holdout_profile=%s output_path=%s",
    summary.common_rows,
    summary.best_profile,
    summary.calibrated_holdout_profile,
    summary.output_path,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Compare baseline and trend-aware hybrid trend recognition.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--feature-root")
  parser.add_argument("--baseline-prediction-path")
  parser.add_argument("--trend-prediction-path")
  parser.add_argument("--output-root")
  parser.add_argument("--provider")
  parser.add_argument("--symbol")
  parser.add_argument("--dataset-name")
  parser.add_argument("--target-column")
  parser.add_argument("--regime-change-column")
  parser.add_argument("--regime-change-threshold", type=float)
  parser.add_argument("--baseline-return-prediction-column")
  parser.add_argument("--trend-return-prediction-column")
  parser.add_argument("--trend-state-prediction-column")
  parser.add_argument("--uptrend-probability-column")
  parser.add_argument("--downtrend-probability-column")
  parser.add_argument("--chop-probability-column")
  parser.add_argument("--gate-threshold", dest="gate_thresholds", type=float, action="append")
  parser.add_argument("--chop-threshold", dest="chop_thresholds", type=float, action="append")
  parser.add_argument("--countertrend-scale", dest="countertrend_scales", type=float, action="append")
  parser.add_argument("--calibration-fraction", type=float)
  parser.add_argument("--minimum-calibration-rows", type=int)
  parser.add_argument("--probability-calibration-method")
  parser.add_argument("--optimization-weight", dest="optimization_weights", action="append")
  parser.add_argument("--max-accuracy-drop", type=float)
  parser.add_argument("--max-trend-direction-accuracy-drop", type=float)
  parser.add_argument("--constraint-penalty-weight", type=float)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _optimization_weights(overrides: list[str] | None, payload: dict[str, object]) -> dict[str, float]:
  values = dict(payload.get("optimization_weights", {}) or {})
  for override in overrides or ():
    if "=" not in override:
      raise ValueError("--optimization-weight must use metric.path=weight")
    metric_path, weight = override.split("=", 1)
    values[metric_path] = float(weight)
  return {str(metric_path): float(weight) for metric_path, weight in values.items()}

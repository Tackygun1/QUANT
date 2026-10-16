from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.research.model_weakness_diagnostics import (
  ModelWeaknessDiagnosticsConfig,
  run_model_weakness_diagnostics,
)
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)

  config = ModelWeaknessDiagnosticsConfig(
    backtest_dir=Path(config_value(args.backtest_dir, payload, "backtest_dir")),
    feature_root=Path(config_value(args.feature_root, payload, "feature_root")),
    provider=config_value(args.provider, payload, "provider", "coinbase"),
    symbol=config_value(args.symbol, payload, "symbol", "BTC-USD"),
    dataset_name=config_value(args.dataset_name, payload, "dataset_name", "market_supervised"),
    trend_column=config_value(args.trend_column, payload, "trend_column", "trend_return_24_bar"),
    trend_threshold_bps=float(args.trend_threshold_bps if args.trend_threshold_bps is not None else payload.get("trend_threshold_bps", 20.0)),
    trend_state_target_column=config_value(args.trend_state_target_column, payload, "trend_state_target_column", "future_trend_state_label_24_bar"),
    trend_state_prediction_column=config_value(args.trend_state_prediction_column, payload, "trend_state_prediction_column", "oof_slow_trend_state_prediction"),
    regime_change_column=config_value(args.regime_change_column, payload, "regime_change_column", "regime_bocpd_change_prob"),
    regime_change_threshold=float(args.regime_change_threshold if args.regime_change_threshold is not None else payload.get("regime_change_threshold", 0.35)),
  )

  summary = run_model_weakness_diagnostics(config)
  logging.getLogger(__name__).info(
    "Model weakness diagnostics complete. primary_weakness=%s secondary_weakness=%s backtest_dir=%s",
    summary.primary_weakness,
    summary.secondary_weakness,
    config.backtest_dir,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Diagnose whether a model is weakest in trend recognition, timing, or hold behavior.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--backtest-dir")
  parser.add_argument("--feature-root")
  parser.add_argument("--provider")
  parser.add_argument("--symbol")
  parser.add_argument("--dataset-name")
  parser.add_argument("--trend-column")
  parser.add_argument("--trend-threshold-bps", type=float)
  parser.add_argument("--trend-state-target-column")
  parser.add_argument("--trend-state-prediction-column")
  parser.add_argument("--regime-change-column")
  parser.add_argument("--regime-change-threshold", type=float)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

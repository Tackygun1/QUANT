from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from quant.research.overnight_training import (
  EvolutionSearchConfig,
  OvernightDataConfig,
  OvernightTrainingConfig,
  run_overnight_training,
)
from quant.risk.sizing import PositionSizingConfig
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  data_payload = payload.get("data", {})
  search_payload = payload.get("search", {})
  sizing_payload = search_payload.get("sizing", {})
  candle_start_time = config_value(args.candle_start_time, data_payload, "candle_start_time")
  trade_start_time = config_value(args.trade_start_time, data_payload, "trade_start_time")
  end_time = config_value(args.end_time, data_payload, "end_time")

  config = OvernightTrainingConfig(
    workspace_root=Path(config_value(args.workspace_root, payload, "workspace_root", "experiments/overnight/default")),
    data=OvernightDataConfig(
      product=config_value(args.product, data_payload, "product", "BTC-USD"),
      provider=config_value(args.provider, data_payload, "provider", "coinbase"),
      candle_interval=config_value(args.candle_interval, data_payload, "candle_interval", "5m"),
      candle_start_time=_parse_dt(candle_start_time) if candle_start_time else None,
      trade_start_time=_parse_dt(trade_start_time) if trade_start_time else None,
      end_time=_parse_dt(end_time) if end_time else None,
      candle_lookback_days=int(config_value(args.candle_lookback_days, data_payload, "candle_lookback_days", 90)),
      trade_lookback_hours=int(config_value(args.trade_lookback_hours, data_payload, "trade_lookback_hours", 72)),
      request_timeout_seconds=float(config_value(args.request_timeout_seconds, data_payload, "request_timeout_seconds", 20.0)),
      request_pause_seconds=float(config_value(args.request_pause_seconds, data_payload, "request_pause_seconds", 0.02)),
    ),
    search=EvolutionSearchConfig(
      duration_hours=float(config_value(args.duration_hours, search_payload, "duration_hours", 8.0)),
      validation_fraction=float(config_value(args.validation_fraction, search_payload, "validation_fraction", 0.2)),
      minimum_rows=int(config_value(args.minimum_rows, search_payload, "minimum_rows", 512)),
      transformer_recent_rows=int(config_value(args.transformer_recent_rows, search_payload, "transformer_recent_rows", 2048)),
      hybrid_recent_rows=int(config_value(args.hybrid_recent_rows, search_payload, "hybrid_recent_rows", 16384)),
      hybrid_horizons=tuple(int(value) for value in config_value(args.hybrid_horizon, search_payload, "hybrid_horizons", (1, 6, 24, 96))),
      hybrid_quantiles=tuple(float(value) for value in config_value(args.hybrid_quantile, search_payload, "hybrid_quantiles", (0.1, 0.5, 0.9))),
      hybrid_meta_validation_fraction=float(config_value(args.hybrid_meta_validation_fraction, search_payload, "hybrid_meta_validation_fraction", 0.2)),
      hybrid_output_dataset_name=str(config_value(args.hybrid_output_dataset_name, search_payload, "hybrid_output_dataset_name", "market_hybrid_stacked")),
      hybrid_initial_train_rows=int(config_value(args.hybrid_initial_train_rows, search_payload, "hybrid_initial_train_rows", 2048)),
      hybrid_validation_rows=int(config_value(args.hybrid_validation_rows, search_payload, "hybrid_validation_rows", 512)),
      hybrid_step_rows=int(config_value(args.hybrid_step_rows, search_payload, "hybrid_step_rows", 512)),
      hybrid_embargo_rows=int(config_value(args.hybrid_embargo_rows, search_payload, "hybrid_embargo_rows", 1)),
      hybrid_purge_rows=int(config_value(args.hybrid_purge_rows, search_payload, "hybrid_purge_rows", 96)),
      hybrid_min_trade_count=int(config_value(args.hybrid_min_trade_count, search_payload, "hybrid_min_trade_count", 25)),
      hybrid_inactivity_penalty=float(args.hybrid_inactivity_penalty if args.hybrid_inactivity_penalty is not None else search_payload.get("hybrid_inactivity_penalty", 0.01)),
      initial_random_candidates=int(config_value(args.initial_random_candidates, search_payload, "initial_random_candidates", 3)),
      elite_count=int(config_value(args.elite_count, search_payload, "elite_count", 2)),
      mutation_power=float(config_value(args.mutation_power, search_payload, "mutation_power", 0.35)),
      retry_attempts=int(config_value(args.retry_attempts, search_payload, "retry_attempts", 3)),
      retry_pause_seconds=float(config_value(args.retry_pause_seconds, search_payload, "retry_pause_seconds", 20.0)),
      seed=int(config_value(args.seed, search_payload, "seed", 7)),
      model_cycle=tuple(config_value(args.model_cycle, search_payload, "model_cycle", ("histgb_return", "ridge_return", "lightgbm_hybrid_return_meta", "lightgbm_hybrid_direction_meta"))),
      fee_bps=float(args.fee_bps if args.fee_bps is not None else search_payload.get("fee_bps", 1.0)),
      slippage_bps=float(args.slippage_bps if args.slippage_bps is not None else search_payload.get("slippage_bps", 0.5)),
      sizing=PositionSizingConfig(
        volatility_column=config_value(args.volatility_column, sizing_payload, "volatility_column", "rolling_return_std_5_bar"),
        volatility_target=float(args.volatility_target if args.volatility_target is not None else sizing_payload.get("volatility_target", 0.0015)),
        volatility_floor=float(args.volatility_floor if args.volatility_floor is not None else sizing_payload.get("volatility_floor", 0.0005)),
        max_abs_position=float(args.max_abs_position if args.max_abs_position is not None else sizing_payload.get("max_abs_position", 1.0)),
        signal_threshold_bps=float(args.signal_threshold_bps if args.signal_threshold_bps is not None else sizing_payload.get("signal_threshold_bps", 0.5)),
        signal_scale_bps=float(args.signal_scale_bps if args.signal_scale_bps is not None else sizing_payload.get("signal_scale_bps", 5.0)),
        max_trade_risk_fraction=float(args.max_trade_risk_fraction if args.max_trade_risk_fraction is not None else sizing_payload.get("max_trade_risk_fraction", 0.01)),
        stop_loss_sigma=float(args.stop_loss_sigma if args.stop_loss_sigma is not None else sizing_payload.get("stop_loss_sigma", 2.0)),
        stop_loss_floor_bps=float(args.stop_loss_floor_bps if args.stop_loss_floor_bps is not None else sizing_payload.get("stop_loss_floor_bps", 100.0)),
        max_position_change=float(args.max_position_change if args.max_position_change is not None else sizing_payload.get("max_position_change", 0.5)),
        max_daily_loss_fraction=float(args.max_daily_loss_fraction if args.max_daily_loss_fraction is not None else sizing_payload.get("max_daily_loss_fraction", 0.02)),
        max_portfolio_drawdown_fraction=float(args.max_portfolio_drawdown_fraction if args.max_portfolio_drawdown_fraction is not None else sizing_payload.get("max_portfolio_drawdown_fraction", 0.1)),
        hold_signal_threshold_bps=float(args.hold_signal_threshold_bps if args.hold_signal_threshold_bps is not None else sizing_payload.get("hold_signal_threshold_bps", 0.0)),
        flip_signal_threshold_bps=float(args.flip_signal_threshold_bps if args.flip_signal_threshold_bps is not None else sizing_payload.get("flip_signal_threshold_bps", 0.0)),
        classification_stale_exit_margin_threshold=float(args.classification_stale_exit_margin_threshold if args.classification_stale_exit_margin_threshold is not None else sizing_payload.get("classification_stale_exit_margin_threshold", 0.0)),
        hold_until_stop_or_take_profit=bool(config_value(args.hold_until_stop_or_take_profit, sizing_payload, "hold_until_stop_or_take_profit", False)),
        profit_take_exit_multiple=float(args.profit_take_exit_multiple if args.profit_take_exit_multiple is not None else sizing_payload.get("profit_take_exit_multiple", 0.0)),
        max_hold_bars=int(args.max_hold_bars if args.max_hold_bars is not None else sizing_payload.get("max_hold_bars", 0)),
        stale_exit_bars=int(args.stale_exit_bars if args.stale_exit_bars is not None else sizing_payload.get("stale_exit_bars", 0)),
        classification_margin_threshold=float(sizing_payload.get("classification_margin_threshold", 0.15)),
        classification_margin_scale=float(sizing_payload.get("classification_margin_scale", 0.35)),
        classification_fixed_target_position=float(sizing_payload.get("classification_fixed_target_position", 0.0)),
        classification_hold_margin_threshold=float(sizing_payload.get("classification_hold_margin_threshold", 0.1)),
        classification_flip_margin_threshold=float(sizing_payload.get("classification_flip_margin_threshold", 0.25)),
        tradeability_gate_enabled=bool(sizing_payload.get("tradeability_gate_enabled", False)),
        tradeability_gate_probability_column=config_value(None, sizing_payload, "tradeability_gate_probability_column", "tradeability_prediction_probability_tradeable"),
        tradeability_gate_confidence_column=config_value(None, sizing_payload, "tradeability_gate_confidence_column", "tradeability_prediction_confidence"),
        tradeability_gate_threshold=float(sizing_payload.get("tradeability_gate_threshold", 0.55)),
        tradeability_gate_min_confidence=float(sizing_payload.get("tradeability_gate_min_confidence", 0.0)),
        forecast_gate_enabled=bool(sizing_payload.get("forecast_gate_enabled", False)),
        forecast_gate_fast_column=config_value(None, sizing_payload, "forecast_gate_fast_column", "oof_hybrid_transformer_return_q50_6_bar"),
        forecast_gate_slow_column=config_value(None, sizing_payload, "forecast_gate_slow_column", "oof_hybrid_transformer_return_q50_24_bar"),
        forecast_gate_fast_threshold_bps=float(sizing_payload.get("forecast_gate_fast_threshold_bps", 3.0)),
        forecast_gate_slow_threshold_bps=float(sizing_payload.get("forecast_gate_slow_threshold_bps", 5.0)),
        forecast_gate_require_alignment=bool(sizing_payload.get("forecast_gate_require_alignment", True)),
        regime_chop_gate_enabled=bool(sizing_payload.get("regime_chop_gate_enabled", False)),
        regime_chop_probability_column=config_value(None, sizing_payload, "regime_chop_probability_column", "regime_hmm_p_chop"),
        regime_chop_threshold=float(sizing_payload.get("regime_chop_threshold", 0.55)),
        regime_chop_entry_threshold_scale=float(sizing_payload.get("regime_chop_entry_threshold_scale", 1.5)),
        regime_chop_flip_threshold_scale=float(sizing_payload.get("regime_chop_flip_threshold_scale", 1.5)),
        regime_chop_max_position_scale=float(sizing_payload.get("regime_chop_max_position_scale", 0.5)),
        target_position_smoothing=float(sizing_payload.get("target_position_smoothing", 0.0)),
        hold_flat_until_time_stop=bool(sizing_payload.get("hold_flat_until_time_stop", False)),
        class_change_persistence_bars=int(sizing_payload.get("class_change_persistence_bars", 0)),
        min_hold_bars=int(args.min_hold_bars if args.min_hold_bars is not None else sizing_payload.get("min_hold_bars", 0)),
        min_rebalance_abs_delta=float(args.min_rebalance_abs_delta if args.min_rebalance_abs_delta is not None else sizing_payload.get("min_rebalance_abs_delta", 0.0)),
        trend_filter_exit_enabled=bool(config_value(args.trend_filter_exit_enabled, sizing_payload, "trend_filter_exit_enabled", False)),
      ),
    ),
    clean_start=bool(config_value(args.clean_start, payload, "clean_start", False)),
    config_path=args.config,
  )

  summary = run_overnight_training(config)
  logging.getLogger(__name__).info(
    "Overnight training complete. workspace_root=%s dataset_rows=%s candidates_completed=%s state_path=%s",
    summary.workspace_root,
    summary.dataset_rows,
    summary.candidates_completed,
    summary.state_path,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run a resumable overnight data build and evolutionary training job.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--workspace-root")
  parser.add_argument("--provider")
  parser.add_argument("--product")
  parser.add_argument("--candle-interval")
  parser.add_argument("--candle-start-time")
  parser.add_argument("--trade-start-time")
  parser.add_argument("--end-time")
  parser.add_argument("--candle-lookback-days", type=int)
  parser.add_argument("--trade-lookback-hours", type=int)
  parser.add_argument("--request-timeout-seconds", type=float)
  parser.add_argument("--request-pause-seconds", type=float)
  parser.add_argument("--duration-hours", type=float)
  parser.add_argument("--validation-fraction", type=float)
  parser.add_argument("--minimum-rows", type=int)
  parser.add_argument("--transformer-recent-rows", type=int)
  parser.add_argument("--hybrid-recent-rows", type=int)
  parser.add_argument("--hybrid-horizon", action="append", type=int)
  parser.add_argument("--hybrid-quantile", action="append", type=float)
  parser.add_argument("--hybrid-meta-validation-fraction", type=float)
  parser.add_argument("--hybrid-output-dataset-name")
  parser.add_argument("--hybrid-initial-train-rows", type=int)
  parser.add_argument("--hybrid-validation-rows", type=int)
  parser.add_argument("--hybrid-step-rows", type=int)
  parser.add_argument("--hybrid-embargo-rows", type=int)
  parser.add_argument("--hybrid-purge-rows", type=int)
  parser.add_argument("--hybrid-min-trade-count", type=int)
  parser.add_argument("--hybrid-inactivity-penalty", type=float)
  parser.add_argument("--initial-random-candidates", type=int)
  parser.add_argument("--elite-count", type=int)
  parser.add_argument("--mutation-power", type=float)
  parser.add_argument("--retry-attempts", type=int)
  parser.add_argument("--retry-pause-seconds", type=float)
  parser.add_argument("--seed", type=int)
  parser.add_argument("--model-cycle", action="append")
  parser.add_argument("--fee-bps", type=float)
  parser.add_argument("--slippage-bps", type=float)
  parser.add_argument("--volatility-column")
  parser.add_argument("--volatility-target", type=float)
  parser.add_argument("--volatility-floor", type=float)
  parser.add_argument("--max-abs-position", type=float)
  parser.add_argument("--signal-threshold-bps", type=float)
  parser.add_argument("--signal-scale-bps", type=float)
  parser.add_argument("--max-trade-risk-fraction", type=float)
  parser.add_argument("--stop-loss-sigma", type=float)
  parser.add_argument("--stop-loss-floor-bps", type=float)
  parser.add_argument("--max-position-change", type=float)
  parser.add_argument("--max-daily-loss-fraction", type=float)
  parser.add_argument("--max-portfolio-drawdown-fraction", type=float)
  parser.add_argument("--hold-signal-threshold-bps", type=float)
  parser.add_argument("--flip-signal-threshold-bps", type=float)
  parser.add_argument("--classification-stale-exit-margin-threshold", type=float)
  parser.add_argument("--hold-until-stop-or-take-profit", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--profit-take-exit-multiple", type=float)
  parser.add_argument("--max-hold-bars", type=int)
  parser.add_argument("--stale-exit-bars", type=int)
  parser.add_argument("--min-hold-bars", type=int)
  parser.add_argument("--min-rebalance-abs-delta", type=float)
  parser.add_argument("--trend-filter-exit-enabled", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--clean-start", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


def _parse_dt(value: str) -> datetime:
  return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)

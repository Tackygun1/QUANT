from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.risk.sizing import PositionSizingConfig
from quant.utils.config import config_value, load_json_config


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  sizing_payload = payload.get("sizing", {})

  config = ModelBacktestConfig(
    artifact_dir=Path(config_value(args.artifact_dir, payload, "artifact_dir")),
    input_root=Path(config_value(args.input_root, payload, "input_root", "data/features")),
    output_root=Path(config_value(args.output_root, payload, "output_root", "reports/backtests")),
    dataset_name=config_value(args.dataset_name, payload, "dataset_name"),
    realized_return_column=config_value(args.realized_return_column, payload, "realized_return_column", "forward_return_1_bar"),
    fee_bps=float(args.fee_bps if args.fee_bps is not None else payload.get("fee_bps", 5.0)),
    slippage_bps=float(args.slippage_bps if args.slippage_bps is not None else payload.get("slippage_bps", 1.0)),
    sizing=PositionSizingConfig(
      volatility_column=config_value(args.volatility_column, sizing_payload, "volatility_column", "rolling_return_std_5_bar"),
      volatility_target=float(args.volatility_target if args.volatility_target is not None else sizing_payload.get("volatility_target", 0.0015)),
      volatility_floor=float(args.volatility_floor if args.volatility_floor is not None else sizing_payload.get("volatility_floor", 0.0005)),
      max_abs_position=float(args.max_abs_position if args.max_abs_position is not None else sizing_payload.get("max_abs_position", 1.0)),
      signal_threshold_bps=float(args.signal_threshold_bps if args.signal_threshold_bps is not None else sizing_payload.get("signal_threshold_bps", 2.0)),
      signal_scale_bps=float(args.signal_scale_bps if args.signal_scale_bps is not None else sizing_payload.get("signal_scale_bps", 10.0)),
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
      trend_filter_enabled=bool(config_value(args.trend_filter_enabled, sizing_payload, "trend_filter_enabled", False)),
      trend_filter_column=config_value(args.trend_filter_column, sizing_payload, "trend_filter_column", "trend_return_24_bar"),
      trend_filter_higher_timeframe_column=config_value(
        args.trend_filter_higher_timeframe_column,
        sizing_payload,
        "trend_filter_higher_timeframe_column",
        "trend_return_96_bar",
      ),
      trend_filter_threshold_bps=float(args.trend_filter_threshold_bps if args.trend_filter_threshold_bps is not None else sizing_payload.get("trend_filter_threshold_bps", 20.0)),
      trend_filter_countertrend_scale=float(args.trend_filter_countertrend_scale if args.trend_filter_countertrend_scale is not None else sizing_payload.get("trend_filter_countertrend_scale", 0.0)),
      trend_filter_require_alignment=bool(sizing_payload.get("trend_filter_require_alignment", True)),
      trend_filter_regime_bias_bps=float(args.trend_filter_regime_bias_bps if args.trend_filter_regime_bias_bps is not None else sizing_payload.get("trend_filter_regime_bias_bps", 0.0)),
      trend_filter_exit_enabled=bool(config_value(args.trend_filter_exit_enabled, sizing_payload, "trend_filter_exit_enabled", False)),
      trend_state_filter_enabled=bool(config_value(args.trend_state_filter_enabled, sizing_payload, "trend_state_filter_enabled", False)),
      trend_state_uptrend_probability_column=config_value(args.trend_state_uptrend_probability_column, sizing_payload, "trend_state_uptrend_probability_column", "oof_fast_trend_state_probability_uptrend"),
      trend_state_downtrend_probability_column=config_value(args.trend_state_downtrend_probability_column, sizing_payload, "trend_state_downtrend_probability_column", "oof_fast_trend_state_probability_downtrend"),
      trend_state_chop_probability_column=config_value(args.trend_state_chop_probability_column, sizing_payload, "trend_state_chop_probability_column", "oof_fast_trend_state_probability_chop"),
      trend_state_confidence_column=config_value(args.trend_state_confidence_column, sizing_payload, "trend_state_confidence_column", "oof_fast_trend_state_confidence"),
      trend_state_gate_threshold=float(args.trend_state_gate_threshold if args.trend_state_gate_threshold is not None else sizing_payload.get("trend_state_gate_threshold", 0.6)),
      trend_state_countertrend_scale=float(args.trend_state_countertrend_scale if args.trend_state_countertrend_scale is not None else sizing_payload.get("trend_state_countertrend_scale", 0.25)),
      trend_state_chop_threshold=float(args.trend_state_chop_threshold if args.trend_state_chop_threshold is not None else sizing_payload.get("trend_state_chop_threshold", 0.65)),
      trend_state_chop_scale=float(args.trend_state_chop_scale if args.trend_state_chop_scale is not None else sizing_payload.get("trend_state_chop_scale", 0.25)),
      trend_state_min_confidence=float(args.trend_state_min_confidence if args.trend_state_min_confidence is not None else sizing_payload.get("trend_state_min_confidence", 0.05)),
      trend_state_regime_bias_bps=float(args.trend_state_regime_bias_bps if args.trend_state_regime_bias_bps is not None else sizing_payload.get("trend_state_regime_bias_bps", 0.0)),
      trend_state_exit_enabled=bool(config_value(args.trend_state_exit_enabled, sizing_payload, "trend_state_exit_enabled", False)),
      trend_state_exit_gate_threshold=float(args.trend_state_exit_gate_threshold if args.trend_state_exit_gate_threshold is not None else sizing_payload.get("trend_state_exit_gate_threshold", 0.75)),
      trend_state_chop_exit_threshold=float(args.trend_state_chop_exit_threshold if args.trend_state_chop_exit_threshold is not None else sizing_payload.get("trend_state_chop_exit_threshold", 0.8)),
    ),
    config_path=args.config,
  )
  summary = run_model_backtest(config)
  logging.getLogger(__name__).info(
    "Backtest complete. status=%s output_dir=%s joined_rows=%s active_periods=%s trades=%s total_net_return=%s max_drawdown=%s sharpe_like=%s",
    summary.status,
    summary.output_dir,
    summary.joined_rows,
    summary.active_periods,
    summary.trades,
    summary.total_net_return,
    summary.max_drawdown,
    summary.sharpe_like,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run a fee-aware backtest on a saved model validation artifact.")
  parser.add_argument("--config", type=Path, help="Optional JSON config file.")
  parser.add_argument("--artifact-dir")
  parser.add_argument("--input-root")
  parser.add_argument("--output-root")
  parser.add_argument("--dataset-name")
  parser.add_argument("--realized-return-column")
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
  parser.add_argument("--trend-filter-enabled", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--trend-filter-column")
  parser.add_argument("--trend-filter-higher-timeframe-column")
  parser.add_argument("--trend-filter-threshold-bps", type=float)
  parser.add_argument("--trend-filter-countertrend-scale", type=float)
  parser.add_argument("--trend-filter-regime-bias-bps", type=float)
  parser.add_argument("--trend-filter-exit-enabled", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--trend-state-filter-enabled", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--trend-state-uptrend-probability-column")
  parser.add_argument("--trend-state-downtrend-probability-column")
  parser.add_argument("--trend-state-chop-probability-column")
  parser.add_argument("--trend-state-confidence-column")
  parser.add_argument("--trend-state-gate-threshold", type=float)
  parser.add_argument("--trend-state-countertrend-scale", type=float)
  parser.add_argument("--trend-state-chop-threshold", type=float)
  parser.add_argument("--trend-state-chop-scale", type=float)
  parser.add_argument("--trend-state-min-confidence", type=float)
  parser.add_argument("--trend-state-regime-bias-bps", type=float)
  parser.add_argument("--trend-state-exit-enabled", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument("--trend-state-exit-gate-threshold", type=float)
  parser.add_argument("--trend-state-chop-exit-threshold", type=float)
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )

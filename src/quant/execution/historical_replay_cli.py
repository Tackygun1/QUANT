from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.execution.historical_replay import HistoricalReplayConfig, run_historical_replay
from quant.execution.paper_trading import PaperTradingConfig
from quant.risk.sizing import PositionSizingConfig
from quant.utils.config import config_value, load_json_config, optional_path


def main() -> int:
  args = _build_parser().parse_args()
  _configure_logging(args.log_level)
  payload = load_json_config(args.config)
  paper_payload = payload.get("paper", {})
  sizing_payload = paper_payload.get("sizing", {})
  artifact_search_root = (
    args.artifact_search_root
    if args.artifact_search_root is not None
    else payload.get("artifact_search_root", paper_payload.get("artifact_search_root"))
  )

  paper_config = PaperTradingConfig(
    artifact_dir=Path(args.artifact_dir if args.artifact_dir is not None else payload.get("artifact_dir", paper_payload.get("artifact_dir"))),
    product=config_value(args.symbol, payload, "symbol", "BTC-USD"),
    artifact_search_root=optional_path(artifact_search_root),
    tradeability_artifact_dir=optional_path(
      config_value(
        args.tradeability_artifact_dir,
        payload,
        "tradeability_artifact_dir",
        paper_payload.get("tradeability_artifact_dir"),
      )
    ),
    output_root=Path(config_value(args.paper_root, payload, "paper_root", "data/replay/demo/paper")),
    candle_interval=config_value(args.candle_interval, paper_payload, "candle_interval", "5m"),
    lookback_hours=int(config_value(args.lookback_hours, paper_payload, "lookback_hours", 3)),
    poll_interval_seconds=float(config_value(args.poll_interval_seconds, paper_payload, "poll_interval_seconds", 300.0)),
    iterations=1,
    fee_bps=float(args.fee_bps if args.fee_bps is not None else paper_payload.get("fee_bps", 1.0)),
    slippage_bps=float(args.slippage_bps if args.slippage_bps is not None else paper_payload.get("slippage_bps", 0.5)),
    take_profit_multiple=float(args.take_profit_multiple if args.take_profit_multiple is not None else paper_payload.get("take_profit_multiple", 0.5)),
    request_timeout_seconds=float(paper_payload.get("request_timeout_seconds", 20.0)),
    request_pause_seconds=float(paper_payload.get("request_pause_seconds", 0.02)),
    sizing=PositionSizingConfig(
      volatility_column=config_value(args.volatility_column, sizing_payload, "volatility_column", "rolling_return_std_5_bar"),
      volatility_target=float(args.volatility_target if args.volatility_target is not None else sizing_payload.get("volatility_target", 0.0015)),
      volatility_floor=float(args.volatility_floor if args.volatility_floor is not None else sizing_payload.get("volatility_floor", 0.0005)),
      max_abs_position=float(args.max_abs_position if args.max_abs_position is not None else sizing_payload.get("max_abs_position", 1.0)),
      signal_threshold_bps=float(args.signal_threshold_bps if args.signal_threshold_bps is not None else sizing_payload.get("signal_threshold_bps", 0.5)),
      signal_scale_bps=float(args.signal_scale_bps if args.signal_scale_bps is not None else sizing_payload.get("signal_scale_bps", 5.0)),
      max_trade_risk_fraction=float(args.max_trade_risk_fraction if args.max_trade_risk_fraction is not None else sizing_payload.get("max_trade_risk_fraction", 0.01)),
      stop_loss_sigma=float(args.stop_loss_sigma if args.stop_loss_sigma is not None else sizing_payload.get("stop_loss_sigma", 2.0)),
      stop_loss_floor_bps=float(args.stop_loss_floor_bps if args.stop_loss_floor_bps is not None else sizing_payload.get("stop_loss_floor_bps", 20.0)),
      max_position_change=float(args.max_position_change if args.max_position_change is not None else sizing_payload.get("max_position_change", 1.0)),
      max_daily_loss_fraction=float(args.max_daily_loss_fraction if args.max_daily_loss_fraction is not None else sizing_payload.get("max_daily_loss_fraction", 0.02)),
      max_portfolio_drawdown_fraction=float(args.max_portfolio_drawdown_fraction if args.max_portfolio_drawdown_fraction is not None else sizing_payload.get("max_portfolio_drawdown_fraction", 0.1)),
      hold_signal_threshold_bps=float(args.hold_signal_threshold_bps if args.hold_signal_threshold_bps is not None else sizing_payload.get("hold_signal_threshold_bps", 0.0)),
      flip_signal_threshold_bps=float(args.flip_signal_threshold_bps if args.flip_signal_threshold_bps is not None else sizing_payload.get("flip_signal_threshold_bps", 0.0)),
      classification_stale_exit_margin_threshold=float(args.classification_stale_exit_margin_threshold if args.classification_stale_exit_margin_threshold is not None else sizing_payload.get("classification_stale_exit_margin_threshold", 0.0)),
      hold_until_stop_or_take_profit=bool(config_value(args.hold_until_stop_or_take_profit, sizing_payload, "hold_until_stop_or_take_profit", False)),
      profit_take_exit_multiple=float(
        args.profit_take_exit_multiple
        if args.profit_take_exit_multiple is not None
        else (
          args.take_profit_multiple
          if args.take_profit_multiple is not None
          else sizing_payload.get("profit_take_exit_multiple", paper_payload.get("take_profit_multiple", 0.0))
        )
      ),
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
    ),
    config_path=args.config,
  )

  config = HistoricalReplayConfig(
    artifact_dir=paper_config.artifact_dir,
    source_data_root=Path(config_value(args.source_data_root, payload, "source_data_root", "data/features")),
    provider=config_value(args.provider, payload, "provider", "coinbase"),
    symbol=config_value(args.symbol, payload, "symbol", "BTC-USD"),
    feature_date=config_value(args.feature_date, payload, "feature_date"),
    artifact_search_root=paper_config.artifact_search_root,
    replay_root=Path(config_value(args.replay_root, payload, "replay_root", "data/replay/demo")),
    step_delay_seconds=float(args.step_delay_seconds if args.step_delay_seconds is not None else payload.get("step_delay_seconds", 0.15)),
    start_row=int(args.start_row if args.start_row is not None else payload.get("start_row", 0)),
    max_rows=int(args.max_rows if args.max_rows is not None else payload.get("max_rows")) if (args.max_rows is not None or payload.get("max_rows") is not None) else None,
    paper=paper_config,
    config_path=args.config,
  )

  summary = run_historical_replay(config)
  logging.getLogger(__name__).info(
    "Historical replay complete. replay_root=%s rows_streamed=%s signals_written=%s orders_written=%s ledger_rows_written=%s",
    summary.replay_root,
    summary.rows_streamed,
    summary.signals_written,
    summary.orders_written,
    summary.ledger_rows_written,
  )
  return 0


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Replay historical supervised rows through a saved model artifact at accelerated speed.")
  parser.add_argument("--config", type=Path)
  parser.add_argument("--artifact-dir")
  parser.add_argument("--artifact-search-root")
  parser.add_argument("--tradeability-artifact-dir")
  parser.add_argument("--source-data-root")
  parser.add_argument("--replay-root")
  parser.add_argument("--paper-root")
  parser.add_argument("--provider")
  parser.add_argument("--symbol")
  parser.add_argument("--feature-date")
  parser.add_argument("--step-delay-seconds", type=float)
  parser.add_argument("--start-row", type=int)
  parser.add_argument("--max-rows", type=int)
  parser.add_argument("--candle-interval")
  parser.add_argument("--lookback-hours", type=int)
  parser.add_argument("--poll-interval-seconds", type=float)
  parser.add_argument("--fee-bps", type=float)
  parser.add_argument("--slippage-bps", type=float)
  parser.add_argument("--take-profit-multiple", type=float)
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
  parser.add_argument("--log-level", default="INFO")
  return parser


def _configure_logging(level: str) -> None:
  logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )


if __name__ == "__main__":
  raise SystemExit(main())

from __future__ import annotations

import unittest

from quant.risk.sizing import (
  PositionState,
  PositionSizingConfig,
  RiskState,
  next_active_position_bars,
  target_position_from_prediction,
)


class PositionSizingConstraintTests(unittest.TestCase):
  def test_one_percent_rule_caps_position_size(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=0.02,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        max_abs_position=1.0,
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_trade_risk_fraction=0.01,
        stop_loss_sigma=2.0,
        stop_loss_floor_bps=200.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertAlmostEqual(decision.raw_target_position, 1.0, places=9)
    self.assertAlmostEqual(decision.target_position, 0.5, places=9)
    self.assertIn("risk_budget_cap", decision.constraint_flags)

  def test_daily_loss_halt_flattens_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=0.01,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=0.02,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      risk_state=RiskState(
        equity=0.97,
        peak_equity=1.0,
        session_start_equity=1.0,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("daily_loss_halt", decision.constraint_flags)

  def test_position_change_cap_limits_turnover(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=1.0,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        max_abs_position=1.0,
        max_position_change=0.25,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
    )

    self.assertAlmostEqual(decision.raw_target_position, 1.0, places=9)
    self.assertAlmostEqual(decision.target_position, -0.25, places=9)
    self.assertIn("turnover_cap", decision.constraint_flags)

  def test_classification_margin_threshold_blocks_weak_probability_edge(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.08,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.15,
        classification_margin_scale=0.35,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertEqual(decision.confidence, 0.0)

  def test_classification_margin_sizes_by_probability_edge(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.325,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.15,
        classification_margin_scale=0.35,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertAlmostEqual(decision.confidence, 0.5, places=9)
    self.assertAlmostEqual(decision.target_position, 0.5, places=9)

  def test_classification_fixed_target_position_uses_full_notional(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.325,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.15,
        classification_margin_scale=0.35,
        classification_fixed_target_position=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertAlmostEqual(decision.confidence, 0.5, places=9)
    self.assertAlmostEqual(decision.raw_target_position, 1.0, places=9)
    self.assertAlmostEqual(decision.target_position, 1.0, places=9)

  def test_hold_flat_until_time_stop_preserves_position_on_weak_flat_signal(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.04,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.12,
        classification_margin_scale=0.35,
        hold_flat_until_time_stop=True,
        max_hold_bars=8,
        stale_exit_bars=6,
        classification_stale_exit_margin_threshold=0.05,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=1.0,
      active_position_bars=4,
    )

    self.assertEqual(decision.target_position, 1.0)
    self.assertIn("time_stop_hold", decision.constraint_flags)

  def test_hold_flat_until_time_stop_allows_duration_exit(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.04,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.12,
        classification_margin_scale=0.35,
        hold_flat_until_time_stop=True,
        max_hold_bars=6,
        stale_exit_bars=4,
        classification_stale_exit_margin_threshold=0.05,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=1.0,
      active_position_bars=6,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("max_hold_exit", decision.constraint_flags)

  def test_trend_filter_blocks_countertrend_short(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "trend_return_24_bar": "0.004",
        "trend_return_96_bar": "0.006",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_filter_enabled=True,
        trend_filter_countertrend_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_filter", decision.constraint_flags)

  def test_trend_filter_keeps_aligned_short(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "trend_return_24_bar": "-0.004",
        "trend_return_96_bar": "-0.006",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_filter_enabled=True,
        trend_filter_countertrend_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertLess(decision.target_position, 0.0)
    self.assertNotIn("trend_filter", decision.constraint_flags)

  def test_trend_filter_bias_can_flip_clear_uptrend_long(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.001,
      row={
        "rolling_return_std_5_bar": "0.001",
        "trend_return_24_bar": "0.004",
        "trend_return_96_bar": "0.006",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_filter_enabled=True,
        trend_filter_countertrend_scale=0.0,
        trend_filter_regime_bias_bps=20.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertGreater(decision.target_position, 0.0)
    self.assertIn("trend_filter", decision.constraint_flags)
    self.assertIn("trend_regime_bias", decision.constraint_flags)

  def test_trend_state_filter_blocks_countertrend_short(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.1",
        "oof_trend_state_probability_uptrend": "0.82",
        "oof_trend_state_confidence": "0.82",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_state_filter_enabled=True,
        trend_state_countertrend_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_state_filter", decision.constraint_flags)

  def test_trend_state_filter_keeps_aligned_long(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.12",
        "oof_trend_state_probability_uptrend": "0.79",
        "oof_trend_state_confidence": "0.79",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_state_filter_enabled=True,
        trend_state_countertrend_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertGreater(decision.target_position, 0.0)
    self.assertNotIn("trend_state_filter", decision.constraint_flags)

  def test_trend_state_filter_respects_gate_threshold(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.2",
        "oof_trend_state_probability_uptrend": "0.58",
        "oof_trend_state_confidence": "0.58",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_state_filter_enabled=True,
        trend_state_gate_threshold=0.65,
        trend_state_countertrend_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertLess(decision.target_position, 0.0)
    self.assertNotIn("trend_state_filter", decision.constraint_flags)

  def test_trend_state_chop_filter_flattens_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=0.01,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_chop": "0.88",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_state_filter_enabled=True,
        trend_state_chop_threshold=0.7,
        trend_state_chop_scale=0.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_state_chop", decision.constraint_flags)

  def test_trend_state_bias_can_flip_clear_uptrend_long(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.0005,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.1",
        "oof_trend_state_probability_uptrend": "0.8",
        "oof_trend_state_confidence": "0.8",
      },
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        trend_state_filter_enabled=True,
        trend_state_countertrend_scale=0.0,
        trend_state_regime_bias_bps=10.0,
      ),
      execution_cost_rate=0.0,
    )

    self.assertGreater(decision.target_position, 0.0)
    self.assertIn("trend_state_filter", decision.constraint_flags)
    self.assertIn("trend_state_regime_bias", decision.constraint_flags)

  def test_hold_hysteresis_keeps_same_direction_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.0003,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        signal_threshold_bps=5.0,
        hold_signal_threshold_bps=2.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.6,
      active_position_bars=3,
    )

    self.assertAlmostEqual(decision.target_position, -0.6, places=9)
    self.assertIn("hold_hysteresis", decision.constraint_flags)

  def test_min_hold_blocks_early_flip(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=0.01,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        flip_signal_threshold_bps=0.0,
        signal_scale_bps=1.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        min_hold_bars=4,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=2,
    )

    self.assertAlmostEqual(decision.target_position, -0.5, places=9)
    self.assertIn("min_hold", decision.constraint_flags)

  def test_class_change_persistence_requires_repeated_flip_signal(self) -> None:
    first = target_position_from_prediction(
      task_type="classification",
      prediction=0.8,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        class_change_persistence_bars=2,
        classification_margin_threshold=0.1,
        classification_flip_margin_threshold=0.2,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=8,
    )

    second = target_position_from_prediction(
      task_type="classification",
      prediction=0.8,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        class_change_persistence_bars=2,
        classification_margin_threshold=0.1,
        classification_flip_margin_threshold=0.2,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=9,
      pending_direction=first.pending_direction,
      pending_direction_bars=first.pending_direction_bars,
    )

    self.assertAlmostEqual(first.target_position, -0.5, places=9)
    self.assertIn("class_change_persistence", first.constraint_flags)
    self.assertEqual(first.pending_direction, 1)
    self.assertEqual(first.pending_direction_bars, 1)
    self.assertGreater(second.target_position, 0.0)
    self.assertEqual(second.pending_direction, 0)

  def test_weak_opposite_classification_signal_flattens_instead_of_holding(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.18,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        classification_hold_margin_threshold=0.05,
        classification_flip_margin_threshold=0.25,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=8,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("flip_exit_hysteresis", decision.constraint_flags)

  def test_tradeability_gate_blocks_new_entry(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.7,
      row={
        "rolling_return_std_5_bar": "0.001",
        "tradeability_prediction_probability_tradeable": "0.42",
        "tradeability_prediction_confidence": "0.7",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        tradeability_gate_enabled=True,
        tradeability_gate_threshold=0.55,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("tradeability_gate_entry", decision.constraint_flags)

  def test_tradeability_gate_flattens_blocked_reverse(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.7,
      row={
        "rolling_return_std_5_bar": "0.001",
        "tradeability_prediction_probability_tradeable": "0.4",
        "tradeability_prediction_confidence": "0.8",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        classification_flip_margin_threshold=0.2,
        tradeability_gate_enabled=True,
        tradeability_gate_threshold=0.55,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=10,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("tradeability_gate_reverse", decision.constraint_flags)

  def test_tradeability_gate_blocks_scale_in_only(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "tradeability_prediction_probability_tradeable": "0.45",
        "tradeability_prediction_confidence": "0.75",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        tradeability_gate_enabled=True,
        tradeability_gate_threshold=0.55,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.3,
      active_position_bars=6,
    )

    self.assertAlmostEqual(decision.target_position, 0.3, places=9)
    self.assertIn("tradeability_gate_scale_in", decision.constraint_flags)

  def test_forecast_gate_blocks_misaligned_entry(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.7,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_hybrid_transformer_return_q50_6_bar": "-0.0008",
        "oof_hybrid_transformer_return_q50_24_bar": "-0.0012",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        forecast_gate_enabled=True,
        forecast_gate_fast_threshold_bps=3.0,
        forecast_gate_slow_threshold_bps=5.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.0,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("forecast_gate_entry", decision.constraint_flags)

  def test_regime_chop_overlay_caps_position_size(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "regime_hmm_p_chop": "0.8",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        regime_chop_gate_enabled=True,
        regime_chop_threshold=0.55,
        regime_chop_max_position_scale=0.4,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.0,
    )

    self.assertAlmostEqual(decision.target_position, 0.4, places=9)
    self.assertIn("regime_chop_size_cap", decision.constraint_flags)

  def test_target_smoothing_blends_toward_new_target(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.8,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.1,
        target_position_smoothing=0.5,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.2,
    )

    self.assertAlmostEqual(decision.target_position, 0.6, places=9)
    self.assertIn("target_smoothing", decision.constraint_flags)

  def test_min_rebalance_ignores_small_same_direction_delta(self) -> None:
    decision = target_position_from_prediction(
      task_type="regression",
      prediction=-0.0018,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        signal_threshold_bps=0.0,
        signal_scale_bps=20.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        min_rebalance_abs_delta=0.1,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.85,
      active_position_bars=5,
    )

    self.assertAlmostEqual(decision.target_position, -0.85, places=9)
    self.assertIn("min_rebalance", decision.constraint_flags)

  def test_max_hold_exit_flattens_stale_same_direction_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.4,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.15,
        classification_margin_scale=0.35,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
        min_hold_bars=2,
        max_hold_bars=4,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=4,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("max_hold_exit", decision.constraint_flags)

  def test_stale_exit_flattens_weak_same_direction_margin(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.08,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        classification_stale_exit_margin_threshold=0.1,
        stale_exit_bars=3,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=3,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("stale_exit", decision.constraint_flags)

  def test_profit_take_exit_flattens_winning_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.3,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        profit_take_exit_multiple=1.5,
        stop_loss_floor_bps=20.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=3,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.4,
        open_return=0.004,
        max_favorable_return=0.004,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("profit_take_exit", decision.constraint_flags)

  def test_stop_loss_exit_flattens_losing_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.3,
      row={
        "rolling_return_std_5_bar": "0.001",
        "close": "99.85",
        "high": "100.05",
        "low": "99.75",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        stop_loss_floor_bps=20.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=3,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=99.85,
        open_return=-0.0015,
        max_adverse_return=-0.0025,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("stop_loss_exit", decision.constraint_flags)

  def test_hold_until_stop_or_take_profit_blocks_counter_signal(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "close": "100.1",
        "high": "100.2",
        "low": "99.95",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=2.0,
        stop_loss_floor_bps=50.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=3,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.1,
        open_return=0.001,
        max_favorable_return=0.002,
      ),
    )

    self.assertEqual(decision.target_position, 0.5)
    self.assertIn("hold_until_exit", decision.constraint_flags)
    self.assertEqual(decision.pending_direction, 0)
    self.assertEqual(decision.pending_direction_bars, 0)

  def test_hold_until_stop_or_take_profit_releases_on_target_hit(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "close": "100.25",
        "high": "100.35",
        "low": "100.1",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=1.5,
        stop_loss_floor_bps=20.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=3,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.25,
        open_return=0.0025,
        max_favorable_return=0.0035,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("profit_take_exit", decision.constraint_flags)
    self.assertNotIn("hold_until_exit", decision.constraint_flags)

  def test_hold_until_stop_or_take_profit_releases_on_max_hold(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.8,
      row={"rolling_return_std_5_bar": "0.001", "close": "100.05", "high": "100.1", "low": "99.95"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=1.0,
        stop_loss_floor_bps=20.0,
        max_hold_bars=8,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=8,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.05,
        open_return=0.0005,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("max_hold_exit", decision.constraint_flags)

  def test_hold_until_stop_or_take_profit_releases_on_stale_signal(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.05,
      row={"rolling_return_std_5_bar": "0.001", "close": "100.02", "high": "100.08", "low": "99.98"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        classification_stale_exit_margin_threshold=0.1,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=1.0,
        stop_loss_floor_bps=20.0,
        stale_exit_bars=6,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=6,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.02,
        open_return=0.0002,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("stale_exit", decision.constraint_flags)

  def test_hold_until_stop_or_take_profit_releases_on_trend_filter_regime_flip(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "trend_return_24_bar": "-0.005",
        "trend_return_96_bar": "-0.006",
        "close": "100.0",
        "high": "100.1",
        "low": "99.9",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=1.0,
        stop_loss_floor_bps=20.0,
        trend_filter_exit_enabled=True,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=4,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.0,
        open_return=0.0,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_filter_exit", decision.constraint_flags)

  def test_hold_until_stop_or_take_profit_releases_on_trend_state_regime_flip(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.8,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.85",
        "oof_trend_state_probability_uptrend": "0.05",
        "oof_trend_state_probability_chop": "0.05",
        "close": "100.0",
        "high": "100.1",
        "low": "99.9",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        hold_until_stop_or_take_profit=True,
        profit_take_exit_multiple=1.0,
        stop_loss_floor_bps=20.0,
        trend_state_exit_enabled=True,
        trend_state_exit_gate_threshold=0.75,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=4,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.0,
        open_return=0.0,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_state_exit", decision.constraint_flags)

  def test_trailing_profit_exit_flattens_after_giveback(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.3,
      row={"rolling_return_std_5_bar": "0.001"},
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        trailing_profit_activation_multiple=1.5,
        trailing_profit_retrace_multiple=1.0,
        stop_loss_floor_bps=20.0,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=3,
      position_state=PositionState(
        entry_price=100.0,
        mark_price=100.1,
        open_return=0.001,
        max_favorable_return=0.0045,
      ),
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trailing_profit_exit", decision.constraint_flags)

  def test_trend_state_exit_flattens_position_when_regime_flips(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=-0.3,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.05",
        "oof_trend_state_probability_uptrend": "0.82",
        "oof_trend_state_probability_chop": "0.1",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        stale_exit_bars=3,
        trend_state_exit_enabled=True,
        trend_state_exit_gate_threshold=0.75,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=-0.5,
      active_position_bars=3,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_state_exit", decision.constraint_flags)

  def test_trend_state_chop_exit_flattens_position(self) -> None:
    decision = target_position_from_prediction(
      task_type="classification",
      prediction=0.3,
      row={
        "rolling_return_std_5_bar": "0.001",
        "oof_trend_state_probability_downtrend": "0.1",
        "oof_trend_state_probability_uptrend": "0.2",
        "oof_trend_state_probability_chop": "0.9",
      },
      sizing=PositionSizingConfig(
        classification_margin_threshold=0.05,
        classification_margin_scale=0.35,
        stale_exit_bars=2,
        trend_state_exit_enabled=True,
        trend_state_chop_exit_threshold=0.8,
        max_position_change=2.0,
        max_daily_loss_fraction=1.0,
        max_portfolio_drawdown_fraction=1.0,
      ),
      execution_cost_rate=0.0,
      previous_position=0.5,
      active_position_bars=2,
    )

    self.assertEqual(decision.target_position, 0.0)
    self.assertIn("trend_state_chop_exit", decision.constraint_flags)

  def test_tracks_active_position_bars(self) -> None:
    self.assertEqual(
      next_active_position_bars(
        previous_position=-0.5,
        previous_active_position_bars=3,
        target_position=-0.7,
      ),
      4,
    )
    self.assertEqual(
      next_active_position_bars(
        previous_position=-0.5,
        previous_active_position_bars=3,
        target_position=0.0,
      ),
      0,
    )


if __name__ == "__main__":
  unittest.main()

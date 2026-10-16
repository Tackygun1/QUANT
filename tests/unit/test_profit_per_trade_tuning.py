from __future__ import annotations

import unittest

from quant.research.profit_per_trade_tuning import (
  ProfitPerTradeRequirements,
  ProfitPerTradeSweepConfig,
  _requirement_failures,
  build_profit_per_trade_variants,
)
from quant.risk.sizing import PositionSizingConfig


class ProfitPerTradeTuningTests(unittest.TestCase):
  def test_grid_includes_forecast_gates_and_fixed_position_candidates(self) -> None:
    config = ProfitPerTradeSweepConfig(
      reference_artifact_dir=None,  # type: ignore[arg-type]
      reference_backtest_dir=None,  # type: ignore[arg-type]
      output_root=None,  # type: ignore[arg-type]
      margin_thresholds=(0.12, 0.2),
      minimum_hold_bars=(2,),
      minimum_rebalance_deltas=(0.025,),
      fixed_target_positions=(0.0, 0.25),
      forecast_gate_profiles=((False, 3.0, 5.0), (True, 5.0, 8.0)),
    )

    variants = build_profit_per_trade_variants(
      config=config,
      base_sizing=PositionSizingConfig(),
    )

    self.assertEqual(len(variants), 8)
    self.assertTrue(any(variant.sizing.forecast_gate_enabled for variant in variants))
    self.assertTrue(any(variant.sizing.classification_fixed_target_position == 0.25 for variant in variants))

  def test_minimum_entry_requirement_is_a_hard_failure(self) -> None:
    metrics = {
      "entries": 99,
      "traded_folds": 14,
      "positive_folds": 10,
      "median_fold_return": 0.001,
      "best_fold_profit_share": 0.3,
      "max_drawdown": 0.005,
      "net_edge_per_turnover_bps": 4.0,
      "stress_net_return": 0.002,
    }

    failures = _requirement_failures(
      metrics,
      requirements=ProfitPerTradeRequirements(minimum_entries=100),
      include_stress=True,
    )

    self.assertIn("minimum_entries", failures)

  def test_candidate_passing_all_requirements_has_no_failures(self) -> None:
    metrics = {
      "entries": 120,
      "traded_folds": 14,
      "positive_folds": 10,
      "median_fold_return": 0.001,
      "best_fold_profit_share": 0.3,
      "max_drawdown": 0.005,
      "net_edge_per_turnover_bps": 4.0,
      "stress_net_return": 0.002,
    }

    failures = _requirement_failures(
      metrics,
      requirements=ProfitPerTradeRequirements(),
      include_stress=True,
    )

    self.assertEqual(failures, [])


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from quant.research.rolling_holdout import aggregate_model_window_results, build_rolling_windows


class RollingHoldoutTests(unittest.TestCase):
  def test_builds_non_overlapping_windows(self) -> None:
    windows = build_rolling_windows(
      evaluation_start=datetime(2026, 4, 16, 2, 30, tzinfo=UTC),
      evaluation_end=datetime(2026, 4, 16, 10, 30, tzinfo=UTC),
      window_hours=2.0,
      step_hours=2.0,
    )

    self.assertEqual(len(windows), 4)
    self.assertEqual(windows[0].slug, "01_20260416T023000Z_20260416T043000Z")
    self.assertEqual(windows[-1].slug, "04_20260416T083000Z_20260416T103000Z")

  def test_aggregates_model_results(self) -> None:
    summary = aggregate_model_window_results(
      "lightgbm_hybrid_direction_meta",
      [
        {
          "evaluation_rows": 24,
          "metrics": {"test_accuracy": 0.55},
          "backtest": {
            "total_net_return": 0.01,
            "max_drawdown": 0.02,
            "trades": 10,
            "active_periods": 12,
          },
          "aggregate_metrics": {
            "total_gross_return": 0.03,
            "total_transaction_cost": 0.01,
            "total_turnover": 20.0,
            "average_turnover_per_period": 0.5,
          },
          "weakness": {
            "cost_share_of_gross": 0.3333333333,
            "average_resize_rate": 0.4,
            "primary_weakness": "timing_churn",
            "secondary_weakness": "trend_recognition",
          },
        },
        {
          "evaluation_rows": 24,
          "metrics": {"test_accuracy": 0.45},
          "backtest": {
            "total_net_return": -0.005,
            "max_drawdown": 0.01,
            "trades": 5,
            "active_periods": 8,
          },
          "aggregate_metrics": {
            "total_gross_return": 0.01,
            "total_transaction_cost": 0.012,
            "total_turnover": 10.0,
            "average_turnover_per_period": 0.25,
          },
          "weakness": {
            "cost_share_of_gross": 1.2,
            "average_resize_rate": 0.2,
            "primary_weakness": "timing_churn",
            "secondary_weakness": "hold_management",
          },
        },
      ],
    )

    self.assertEqual(summary["window_count"], 2)
    self.assertEqual(summary["positive_windows"], 1)
    self.assertEqual(summary["total_trades"], 15)
    self.assertAlmostEqual(summary["mean_net_return"], 0.0025)
    self.assertAlmostEqual(summary["total_transaction_cost"], 0.022)
    self.assertEqual(summary["primary_weakness_counts"]["timing_churn"], 2)
    self.assertAlmostEqual(summary["metrics_summary"]["test_accuracy"], 0.5)


if __name__ == "__main__":
  unittest.main()

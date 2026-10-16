from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quant.research.execution_tuning import (
  annotate_candidate_selection_score,
  build_artifact_execution_variants,
  build_execution_variants,
  load_sizing_from_backtest_config,
  load_window_evaluations,
)
from quant.risk.sizing import PositionSizingConfig


class ExecutionTuningTests(unittest.TestCase):
  def test_load_window_evaluations_filters_by_model(self) -> None:
    summary = {
      "windows": [
        {
          "window_index": 1,
          "holdout_root": "/tmp/window1",
          "results": {
            "histgb_return": {
              "evaluation_artifact_dir": "/tmp/window1/eval/histgb",
            },
          },
        },
        {
          "window_index": 2,
          "holdout_root": "/tmp/window2",
          "results": {
            "lightgbm_hybrid_direction_meta": {
              "evaluation_artifact_dir": "/tmp/window2/eval/hybrid",
            },
          },
        },
      ]
    }

    windows = load_window_evaluations(summary, "histgb_return")

    self.assertEqual(len(windows), 1)
    self.assertEqual(windows[0].window_index, 1)
    self.assertEqual(windows[0].holdout_root, Path("/tmp/window1"))

  def test_build_execution_variants_expands_regression_threshold_grid(self) -> None:
    variants = build_execution_variants(
      model_name="histgb_return",
      base_sizing=PositionSizingConfig(signal_threshold_bps=0.5, signal_scale_bps=5.0),
    )

    slugs = {variant.slug for variant in variants}
    self.assertIn("thr0_scale1_hold0_keep0_rebal0", slugs)
    self.assertIn("thr0.5_scale5_hold3_keep0.5_rebal0.05", slugs)
    self.assertEqual(len(slugs), len(variants))

  def test_build_artifact_execution_variants_uses_executable_profiles_when_oof_columns_present(self) -> None:
    variants = build_artifact_execution_variants(
      model_name="lightgbm_hybrid_direction_meta",
      base_sizing=PositionSizingConfig(),
      feature_columns=(
        "close",
        "oof_fast_trend_state_probability_uptrend",
        "oof_fast_trend_state_probability_downtrend",
        "oof_fast_trend_state_probability_chop",
        "oof_fast_trend_state_confidence",
        "oof_slow_trend_state_probability_uptrend",
        "oof_slow_trend_state_probability_downtrend",
        "oof_slow_trend_state_probability_chop",
        "oof_slow_trend_state_confidence",
        "oof_fast_triple_barrier_probability_profit",
      ),
    )

    slugs = {variant.slug for variant in variants}
    self.assertTrue(any(slug.startswith("plain_") for slug in slugs))
    self.assertTrue(any(slug.startswith("fast_gate_soft_") for slug in slugs))
    self.assertTrue(any(slug.startswith("slow_gate_strict_") for slug in slugs))
    self.assertTrue(any(slug.startswith("slow_gate_with_rawtrend_") for slug in slugs))
    self.assertEqual(len(slugs), len(variants))

  def test_build_artifact_execution_variants_uses_focused_hybrid_profiles_without_oof_columns(self) -> None:
    variants = build_artifact_execution_variants(
      model_name="lightgbm_hybrid_direction_meta",
      base_sizing=PositionSizingConfig(
        hold_until_stop_or_take_profit=True,
        trend_filter_column="trend_return_24_bar",
        trend_filter_higher_timeframe_column="trend_return_96_bar",
      ),
      feature_columns=(
        "close",
        "trend_return_24_bar",
        "trend_return_96_bar",
      ),
    )

    slugs = {variant.slug for variant in variants}
    self.assertTrue(any("_nolock_" in slug for slug in slugs))
    self.assertTrue(any("lock_xhold4_stale2_pt0p5" in slug for slug in slugs))
    self.assertTrue(any(slug.startswith("rawtrend_") for slug in slugs))
    self.assertEqual(len(slugs), len(variants))

  def test_load_sizing_from_backtest_config_reads_sizing_payload(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "backtest_config.json"
      path.write_text(
        json.dumps(
          {
            "sizing": {
              "min_hold_bars": 4,
              "min_rebalance_abs_delta": 0.05,
              "classification_margin_threshold": 0.2,
              "max_position_change": 0.25,
            }
          }
        ),
        encoding="utf-8",
      )

      sizing = load_sizing_from_backtest_config(path)

      self.assertEqual(sizing.min_hold_bars, 4)
      self.assertAlmostEqual(sizing.min_rebalance_abs_delta, 0.05)
      self.assertAlmostEqual(sizing.classification_margin_threshold, 0.2)
      self.assertAlmostEqual(sizing.max_position_change, 0.25)

  def test_annotate_candidate_selection_score_penalizes_inactivity(self) -> None:
    candidate = annotate_candidate_selection_score(
      {
        "after_cost_trade_quality": 0.001,
        "trades": 4,
        "active_periods": 2,
        "joined_rows": 200,
      },
      minimum_trade_count=20,
      minimum_active_coverage=0.05,
      inactivity_trade_penalty=0.01,
      inactivity_coverage_penalty=0.02,
    )

    self.assertAlmostEqual(candidate["active_coverage"], 0.01)
    self.assertGreater(candidate["inactivity_penalty"], 0.0)
    self.assertLess(candidate["selection_score"], candidate["after_cost_trade_quality"])


if __name__ == "__main__":
  unittest.main()

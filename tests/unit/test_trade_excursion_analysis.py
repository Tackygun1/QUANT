from __future__ import annotations

import unittest

from quant.research.trade_excursion_analysis import (
  build_excursion_informed_sizing_candidates,
  extract_trade_excursions,
  suggest_excursion_informed_exit_parameters,
  summarize_trade_excursions,
)
from quant.risk.sizing import PositionSizingConfig


class TradeExcursionAnalysisTests(unittest.TestCase):
  def test_extracts_trade_excursions_from_period_rows(self) -> None:
    periods = [
      {
        "candle_start": "2026-04-12T00:00:00.000Z",
        "target_position": "1.0",
        "open_return": "0.001",
        "net_return": "0.0008",
        "max_favorable_return": "0.001",
        "max_adverse_return": "-0.0005",
        "stop_distance_rate": "0.002",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:05:00.000Z",
        "target_position": "1.0",
        "open_return": "0.0015",
        "net_return": "0.0004",
        "max_favorable_return": "0.003",
        "max_adverse_return": "-0.0005",
        "stop_distance_rate": "0.002",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:10:00.000Z",
        "target_position": "0.0",
        "open_return": "0.0",
        "net_return": "-0.0001",
        "max_favorable_return": "0.0",
        "max_adverse_return": "0.0",
        "stop_distance_rate": "0.0",
        "constraint_flags": "profit_take_exit",
      },
      {
        "candle_start": "2026-04-12T00:15:00.000Z",
        "target_position": "-1.0",
        "open_return": "0.0005",
        "net_return": "0.0002",
        "max_favorable_return": "0.001",
        "max_adverse_return": "-0.0007",
        "stop_distance_rate": "0.002",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:20:00.000Z",
        "target_position": "-1.0",
        "open_return": "-0.0002",
        "net_return": "-0.0003",
        "max_favorable_return": "0.001",
        "max_adverse_return": "-0.0012",
        "stop_distance_rate": "0.002",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:25:00.000Z",
        "target_position": "0.0",
        "open_return": "0.0",
        "net_return": "0.0",
        "max_favorable_return": "0.0",
        "max_adverse_return": "0.0",
        "stop_distance_rate": "0.0",
        "constraint_flags": "stale_exit",
      },
    ]

    excursions = extract_trade_excursions(periods)

    self.assertEqual(len(excursions), 2)
    self.assertEqual(excursions[0].bars, 2)
    self.assertAlmostEqual(excursions[0].max_favorable_return, 0.003)
    self.assertAlmostEqual(excursions[0].peak_giveback_return, 0.0015)
    self.assertEqual(excursions[0].time_to_peak_bars, 2)
    self.assertEqual(excursions[0].exit_flags, ("profit_take_exit",))
    self.assertEqual(excursions[1].direction, -1)
    self.assertEqual(excursions[1].exit_flags, ("stale_exit",))

  def test_suggests_smaller_exit_geometry_when_excursions_are_tight(self) -> None:
    periods = [
      {
        "candle_start": "2026-04-12T00:00:00.000Z",
        "target_position": "1.0",
        "open_return": "0.0009",
        "net_return": "0.0008",
        "max_favorable_return": "0.0012",
        "max_adverse_return": "-0.0004",
        "stop_distance_rate": "0.01",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:05:00.000Z",
        "target_position": "0.0",
        "open_return": "0.0",
        "net_return": "0.0",
        "max_favorable_return": "0.0",
        "max_adverse_return": "0.0",
        "stop_distance_rate": "0.0",
        "constraint_flags": "exit",
      },
      {
        "candle_start": "2026-04-12T00:10:00.000Z",
        "target_position": "-1.0",
        "open_return": "0.0011",
        "net_return": "0.001",
        "max_favorable_return": "0.0018",
        "max_adverse_return": "-0.0005",
        "stop_distance_rate": "0.01",
        "constraint_flags": "",
      },
      {
        "candle_start": "2026-04-12T00:15:00.000Z",
        "target_position": "0.0",
        "open_return": "0.0",
        "net_return": "0.0",
        "max_favorable_return": "0.0",
        "max_adverse_return": "0.0",
        "stop_distance_rate": "0.0",
        "constraint_flags": "exit",
      },
    ]
    excursions = extract_trade_excursions(periods)

    recommendations = suggest_excursion_informed_exit_parameters(
      trades=excursions,
      base_sizing=PositionSizingConfig(stop_loss_floor_bps=100.0),
    )
    summary = summarize_trade_excursions(trades=excursions, recommendations=recommendations)
    candidates = build_excursion_informed_sizing_candidates(
      base_sizing=PositionSizingConfig(stop_loss_floor_bps=100.0),
      stop_loss_floor_bps_candidates=recommendations["stop_loss_floor_bps_candidates"],
      profit_take_multiple_candidates=recommendations["profit_take_multiple_candidates"],
      trailing_multiple_candidates=[tuple(candidate) for candidate in recommendations["trailing_multiple_candidates"]],
    )

    self.assertEqual(summary.trade_count, 2)
    self.assertGreater(summary.mfe_bps_quantiles["p50"], 0.0)
    self.assertTrue(recommendations["stop_loss_floor_bps_candidates"])
    self.assertLess(min(recommendations["stop_loss_floor_bps_candidates"]), 100.0)
    self.assertGreater(len(candidates), 1)


if __name__ == "__main__":
  unittest.main()

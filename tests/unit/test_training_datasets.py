from __future__ import annotations

import unittest

from quant.training.datasets import (
  build_multi_target_sequence_dataset,
  build_sequence_dataset,
  build_tabular_dataset,
  select_feature_columns,
)


class TrainingDatasetTests(unittest.TestCase):
  def test_selects_numeric_feature_columns_and_excludes_labels(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": "2026-04-12T19:29:00.000Z",
        "close": "100",
        "volume": "10",
        "trades_window_count": "4",
        "forward_return_1_bar": "0.01",
        "direction_label_1_bar": "1",
        "tradeable_label_1_bar": "1",
        "legacy_tradeable_label_1_bar": "0",
        "oracle_position_label_1_bar": "1",
        "oracle_tradeability_score_1_bar": "0.95",
        "future_trend_state_label_24_bar": "1",
        "triple_barrier_label_24_bar": "1",
      }
    ]
    feature_columns = select_feature_columns(rows)
    self.assertIn("close", feature_columns)
    self.assertIn("volume", feature_columns)
    self.assertIn("trades_window_count", feature_columns)
    self.assertNotIn("forward_return_1_bar", feature_columns)
    self.assertNotIn("direction_label_1_bar", feature_columns)
    self.assertNotIn("tradeable_label_1_bar", feature_columns)
    self.assertNotIn("legacy_tradeable_label_1_bar", feature_columns)
    self.assertNotIn("oracle_position_label_1_bar", feature_columns)
    self.assertNotIn("oracle_tradeability_score_1_bar", feature_columns)
    self.assertNotIn("future_trend_state_label_24_bar", feature_columns)
    self.assertNotIn("triple_barrier_label_24_bar", feature_columns)

  def test_builds_tabular_dataset(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": "2026-04-12T19:29:00.000Z",
        "is_closed": "true",
        "close": "100",
        "volume": "10",
        "forward_return_1_bar": "0.01",
      },
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": "2026-04-12T19:30:00.000Z",
        "is_closed": "false",
        "close": "101",
        "volume": "11",
        "forward_return_1_bar": "0.02",
      },
    ]
    dataset = build_tabular_dataset(
      rows,
      target_column="forward_return_1_bar",
      classification=False,
    )
    self.assertEqual(dataset.feature_names, ["close", "volume"])
    self.assertEqual(len(dataset.rows), 1)
    self.assertEqual(dataset.targets, [0.01])

  def test_filters_features_by_group_and_column(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": "2026-04-12T19:29:00.000Z",
        "is_closed": "true",
        "close": "100",
        "volume": "10",
        "trend_return_6_bar": "0.01",
        "trend_return_24_bar": "0.02",
        "trend_return_96_bar": "0.05",
        "trend_close_vs_ema_24_pct": "0.003",
        "trend_close_vs_ema_96_pct": "0.005",
        "trend_ema_spread_24_96_pct": "0.002",
        "trend_realized_volatility_20_bar": "0.004",
        "structure_close_vs_session_vwap_pct": "0.002",
        "trades_window_count": "4",
        "top_window_mid_mean": "100.1",
        "oof_hybrid_transformer_return_q50_1_bar": "0.003",
        "oof_trend_state_probability_uptrend": "0.7",
        "oof_fast_trend_state_probability_uptrend": "0.8",
        "oof_slow_trend_state_probability_uptrend": "0.6",
        "oof_fast_triple_barrier_probability_profit": "0.65",
        "forward_return_1_bar": "0.01",
      }
    ]

    baseline_columns = select_feature_columns(rows, feature_groups=("baseline_core",))
    self.assertEqual(
      baseline_columns,
      ["close", "volume", "trades_window_count", "top_window_mid_mean"],
    )

    trend_regime_columns = select_feature_columns(rows, feature_groups=("trend_regime_layer",))
    self.assertEqual(
      trend_regime_columns,
      [
        "trend_return_24_bar",
        "trend_return_96_bar",
        "trend_close_vs_ema_24_pct",
        "trend_close_vs_ema_96_pct",
        "trend_ema_spread_24_96_pct",
        "trend_realized_volatility_20_bar",
      ],
    )

    pruned_columns = select_feature_columns(
      rows,
      feature_groups=("baseline_core", "trend_layer"),
      include_feature_columns=("structure_close_vs_session_vwap_pct",),
    )
    self.assertEqual(
      pruned_columns,
      [
        "close",
        "volume",
        "trend_return_6_bar",
        "trend_return_24_bar",
        "trend_return_96_bar",
        "trend_close_vs_ema_24_pct",
        "trend_close_vs_ema_96_pct",
        "trend_ema_spread_24_96_pct",
        "trend_realized_volatility_20_bar",
        "structure_close_vs_session_vwap_pct",
        "trades_window_count",
        "top_window_mid_mean",
      ],
    )

    dataset = build_tabular_dataset(
      rows,
      target_column="forward_return_1_bar",
      classification=False,
      feature_groups=(
        "baseline_core",
        "hybrid_transformer_oof",
        "trend_state_oof",
        "fast_trend_state_oof",
        "slow_trend_state_oof",
        "fast_triple_barrier_oof",
      ),
      exclude_feature_columns=("volume",),
    )
    self.assertEqual(
      dataset.feature_names,
      [
        "close",
        "trades_window_count",
        "top_window_mid_mean",
        "oof_hybrid_transformer_return_q50_1_bar",
        "oof_trend_state_probability_uptrend",
        "oof_fast_trend_state_probability_uptrend",
        "oof_slow_trend_state_probability_uptrend",
        "oof_fast_triple_barrier_probability_profit",
      ],
    )

  def test_rejects_unknown_feature_groups(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": "2026-04-12T19:29:00.000Z",
        "close": "100",
        "forward_return_1_bar": "0.01",
      }
    ]

    with self.assertRaisesRegex(ValueError, "Unknown feature group"):
      select_feature_columns(rows, feature_groups=("does_not_exist",))

  def test_builds_sequence_dataset(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": f"2026-04-12T19:2{i}:00.000Z",
        "is_closed": "true",
        "close": str(100 + i),
        "volume": str(10 + i),
        "forward_return_1_bar": "0.01" if i >= 2 else "",
      }
      for i in range(4)
    ]
    dataset = build_sequence_dataset(
      rows,
      target_column="forward_return_1_bar",
      lookback=2,
    )
    self.assertEqual(dataset.feature_names, ["close", "volume"])
    self.assertEqual(len(dataset.rows), 2)
    self.assertEqual(len(dataset.sequences[0]), 2)
    self.assertEqual(dataset.targets, [0.01, 0.01])

  def test_builds_multi_target_sequence_dataset_with_auxiliary_targets(self) -> None:
    rows = [
      {
        "provider": "binance",
        "symbol": "BTCUSDT",
        "feature_date": "2026-04-12",
        "candle_start": f"2026-04-12T19:2{i}:00.000Z",
        "is_closed": "true",
        "close": str(100 + i),
        "volume": str(10 + i),
        "forward_return_1_bar": "0.01" if i >= 2 else "",
        "forward_return_3_bar": "0.03" if i >= 2 else "",
        "future_trend_state_label_3_bar": str((i % 3) - 1) if i >= 2 else "",
      }
      for i in range(4)
    ]

    dataset = build_multi_target_sequence_dataset(
      rows,
      target_columns=("forward_return_1_bar", "forward_return_3_bar"),
      auxiliary_target_columns=("future_trend_state_label_3_bar",),
      lookback=2,
    )

    self.assertEqual(dataset.target_columns, ["forward_return_1_bar", "forward_return_3_bar"])
    self.assertEqual(dataset.auxiliary_target_columns, ["future_trend_state_label_3_bar"])
    self.assertEqual(len(dataset.rows), 2)
    self.assertEqual(dataset.targets[0], [0.01, 0.03])
    self.assertEqual(dataset.auxiliary_targets[0], [1])


if __name__ == "__main__":
  unittest.main()

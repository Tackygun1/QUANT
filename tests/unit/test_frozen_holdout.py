from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path

from quant.models.baselines import RidgeReturnRegressor
from quant.research.frozen_holdout import _resolve_artifact_execution_sizing, build_frozen_predictions
from quant.risk.sizing import PositionSizingConfig


class FrozenHoldoutTests(unittest.TestCase):
  def test_resolves_execution_sizing_from_artifact_metadata(self) -> None:
    base = PositionSizingConfig(
      signal_threshold_bps=0.5,
      signal_scale_bps=5.0,
      min_hold_bars=0,
      min_rebalance_abs_delta=0.0,
    )
    resolved, source = _resolve_artifact_execution_sizing(
      source_metadata={
        "extra": {
          "execution_sizing": {
            "min_hold_bars": 3,
            "min_rebalance_abs_delta": 0.1,
            "signal_threshold_bps": 1.5,
          }
        }
      },
      base_sizing=base,
      use_artifact_execution_sizing=True,
    )

    self.assertEqual(source, "artifact_execution_sizing")
    self.assertEqual(resolved.min_hold_bars, 3)
    self.assertAlmostEqual(resolved.min_rebalance_abs_delta, 0.1)
    self.assertAlmostEqual(resolved.signal_threshold_bps, 1.5)
    self.assertAlmostEqual(resolved.signal_scale_bps, 5.0)

  def test_falls_back_to_base_sizing_when_artifact_execution_disabled(self) -> None:
    base = PositionSizingConfig(
      signal_threshold_bps=0.5,
      signal_scale_bps=5.0,
      min_hold_bars=1,
    )
    resolved, source = _resolve_artifact_execution_sizing(
      source_metadata={
        "extra": {
          "execution_params": {
            "min_hold_bars": 5,
            "min_rebalance_abs_delta": 0.2,
          }
        }
      },
      base_sizing=base,
      use_artifact_execution_sizing=False,
    )

    self.assertEqual(source, "config")
    self.assertEqual(resolved, base)

  def test_builds_tabular_predictions_for_evaluation_window(self) -> None:
    model = RidgeReturnRegressor(alpha=1.0)
    model.fit(
      [
        [100.0, 101.0, 0.001],
        [101.0, 102.0, 0.001],
        [102.0, 103.0, 0.001],
      ],
      [0.0001, 0.0002, 0.0003],
    )
    rows = [
      {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T03:50:00.000Z",
        "is_closed": "true",
        "open": "100.0",
        "close": "101.0",
        "rolling_return_std_5_bar": "0.001",
        "forward_return_1_bar": "0.0001",
      },
      {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T03:55:00.000Z",
        "is_closed": "true",
        "open": "101.0",
        "close": "102.0",
        "rolling_return_std_5_bar": "0.001",
        "forward_return_1_bar": "0.0002",
      },
      {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T04:00:00.000Z",
        "is_closed": "true",
        "open": "102.0",
        "close": "103.0",
        "rolling_return_std_5_bar": "0.001",
        "forward_return_1_bar": "0.0003",
      },
    ]
    prediction_rows, targets, predictions = build_frozen_predictions(
      rows=rows,
      source_metadata={
        "model_name": "ridge_return",
        "task_type": "regression",
        "target_column": "forward_return_1_bar",
        "feature_columns": ("open", "close", "rolling_return_std_5_bar"),
      },
      model=model,
      evaluation_start=datetime(2026, 4, 13, 3, 55, tzinfo=UTC),
      evaluation_end=datetime(2026, 4, 13, 4, 5, tzinfo=UTC),
    )

    self.assertEqual(len(prediction_rows), 2)
    self.assertEqual([row["candle_start"] for row in prediction_rows], [
      "2026-04-13T03:55:00.000Z",
      "2026-04-13T04:00:00.000Z",
    ])
    self.assertEqual(targets, [0.0002, 0.0003])
    self.assertEqual(len(predictions), 2)

  def test_builds_meta_model_predictions_for_evaluation_window(self) -> None:
    class DummyMetaModel:
      def predict(self, rows: list[list[float]]) -> list[float]:
        return [sum(row) for row in rows]

    rows = [
      {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T03:50:00.000Z",
        "is_closed": "true",
        "close": "101.0",
        "rolling_return_std_5_bar": "0.001",
        "forward_return_1_bar": "0.0001",
      },
      {
        "provider": "coinbase",
        "symbol": "BTC-USD",
        "candle_start": "2026-04-13T03:55:00.000Z",
        "is_closed": "true",
        "close": "102.0",
        "rolling_return_std_5_bar": "0.001",
        "forward_return_1_bar": "0.0002",
      },
    ]

    prediction_rows, targets, predictions = build_frozen_predictions(
      rows=rows,
      source_metadata={
        "model_name": "lightgbm_hybrid_return_meta",
        "task_type": "regression",
        "target_column": "forward_return_1_bar",
        "feature_columns": ("close", "rolling_return_std_5_bar"),
      },
      model=DummyMetaModel(),
      evaluation_start=datetime(2026, 4, 13, 3, 55, tzinfo=UTC),
      evaluation_end=datetime(2026, 4, 13, 4, 0, tzinfo=UTC),
      artifact_dir=Path("/tmp/fake_artifact"),
    )

    self.assertEqual(len(prediction_rows), 1)
    self.assertEqual(targets, [0.0002])
    self.assertAlmostEqual(predictions[0], 102.001)


if __name__ == "__main__":
  unittest.main()

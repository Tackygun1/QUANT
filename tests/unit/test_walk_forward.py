from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from quant.models.transformers.return_forecaster import TransformerForecasterConfig
from quant.validation.walk_forward import (
  TabularWalkForwardConfig,
  TransformerWalkForwardConfig,
  evaluate_tabular_walk_forward,
  evaluate_transformer_walk_forward,
  expanding_window_splits,
)


class WalkForwardSplitTests(unittest.TestCase):
  def test_builds_expanding_window_splits(self) -> None:
    splits = expanding_window_splits(
      12,
      initial_train_rows=4,
      validation_rows=2,
      step_rows=2,
      embargo_rows=1,
    )
    self.assertEqual(
      splits,
      [
        ([0, 1, 2, 3], [5, 6]),
        ([0, 1, 2, 3, 4, 5], [7, 8]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [9, 10]),
      ],
    )


class WalkForwardTabularTests(unittest.TestCase):
  def test_evaluates_tabular_models_and_writes_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_tabular_dataset(root / "features")

      summary = evaluate_tabular_walk_forward(
        TabularWalkForwardConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("ridge_return",),
          initial_train_rows=12,
          validation_rows=4,
          step_rows=4,
          minimum_rows=16,
        )
      )

      self.assertEqual(summary.runs_attempted, 1)
      self.assertEqual(summary.runs_completed, 1)
      self.assertEqual(summary.folds_completed, 3)

      artifact_root = next((root / "models").rglob("summary.json")).parent
      payload = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(payload["status"], "completed")
      self.assertEqual(payload["folds_completed"], 3)
      self.assertIn("overall_validation_rmse", payload["aggregate_metrics"])

      with (artifact_root / "fold_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
      self.assertEqual(len(fold_rows), 3)

  def test_writes_trend_state_probability_outputs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_tabular_dataset(root / "features", row_count=64)

      summary = evaluate_tabular_walk_forward(
        TabularWalkForwardConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("histgb_trend_state",),
          initial_train_rows=32,
          validation_rows=4,
          step_rows=4,
          minimum_rows=40,
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_root = next((root / "models").rglob("summary.json")).parent
      with (artifact_root / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))

      self.assertTrue(prediction_rows)
      self.assertIn("prediction_probability_downtrend", prediction_rows[0])
      self.assertIn("prediction_probability_chop", prediction_rows[0])
      self.assertIn("prediction_probability_uptrend", prediction_rows[0])
      self.assertIn("prediction_trend_state_margin", prediction_rows[0])

  @staticmethod
  def _write_tabular_dataset(input_root: Path, row_count: int = 24) -> None:
    path = (
      input_root
      / "training_datasets"
      / "market_supervised"
      / "binance"
      / "BTCUSDT"
      / "2026"
      / "04"
      / "12.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "candle_end",
          "is_closed",
          "close",
          "volume",
          "trades_window_count",
          "trade_imbalance",
          "forward_return_1_bar",
          "direction_label_1_bar",
          "future_trend_state_label_24_bar",
        ]
      )
      for minute in range(row_count):
        label = 1 if minute % 3 == 0 else (-1 if minute % 3 == 1 else 0)
        hour = 19 + minute // 60
        clock_minute = minute % 60
        end_hour = 19 + (minute + 1) // 60
        end_clock_minute = (minute + 1) % 60
        writer.writerow(
          [
            "binance",
            "BTCUSDT",
            "2026-04-12",
            f"2026-04-12T{hour:02d}:{clock_minute:02d}:00.000Z",
            f"2026-04-12T{end_hour:02d}:{end_clock_minute:02d}:00.000Z",
            "true",
            str(100 + minute),
            str(10 + minute),
            str(20 + minute),
            str(((minute % 5) - 2) / 10),
            str(0.001 * ((minute % 4) - 1.5)),
            str(label),
            str(label),
          ]
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class WalkForwardTransformerTests(unittest.TestCase):
  def test_evaluates_transformer_and_writes_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      WalkForwardTabularTests._write_tabular_dataset(root / "features")

      summary = evaluate_transformer_walk_forward(
        TransformerWalkForwardConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          initial_train_rows=12,
          validation_rows=4,
          step_rows=4,
          minimum_rows=16,
          transformer=TransformerForecasterConfig(
            lookback=4,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            batch_size=4,
            epochs=2,
          ),
        )
      )

      self.assertEqual(summary.runs_attempted, 1)
      self.assertEqual(summary.runs_completed, 1)
      self.assertGreaterEqual(summary.folds_completed, 1)

      artifact_root = next((root / "models").rglob("summary.json")).parent
      payload = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(payload["status"], "completed")
      self.assertIn("overall_validation_rmse", payload["aggregate_metrics"])


if __name__ == "__main__":
  unittest.main()

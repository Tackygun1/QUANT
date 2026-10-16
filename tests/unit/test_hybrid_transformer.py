from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from quant.models.transformers.hybrid_forecaster import HybridTransformerConfig
from quant.training.train_hybrid_transformer import (
  HybridTransformerTrainingConfig,
  train_hybrid_transformers,
)
from quant.validation.hybrid_walk_forward import (
  HybridTransformerWalkForwardConfig,
  evaluate_hybrid_transformer_walk_forward,
  purged_expanding_window_splits,
)


class HybridWalkForwardSplitTests(unittest.TestCase):
  def test_builds_purged_expanding_window_splits(self) -> None:
    splits = purged_expanding_window_splits(
      20,
      initial_train_rows=8,
      validation_rows=3,
      step_rows=3,
      embargo_rows=1,
      purge_rows=2,
    )
    self.assertEqual(
      splits,
      [
        ([0, 1, 2, 3, 4, 5], [9, 10, 11]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [12, 13, 14]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [15, 16, 17]),
      ],
    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class HybridTransformerTrainingTests(unittest.TestCase):
  def test_trains_and_writes_multi_output_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_dataset(root / "features")

      summary = train_hybrid_transformers(
        HybridTransformerTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          minimum_rows=12,
          target_columns=("forward_return_1_bar", "forward_return_3_bar"),
          feature_groups=("baseline_core",),
          transformer=HybridTransformerConfig(
            lookback=6,
            backbone="timemixer",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            scales=(1, 2),
            mixer_layers=1,
            dropout=0.0,
            batch_size=8,
            epochs=2,
            validation_fraction=0.25,
            horizons=(1, 3),
            quantiles=(0.1, 0.5, 0.9),
          ),
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_root = next((root / "models").rglob("metadata.json")).parent
      metadata = json.loads((artifact_root / "metadata.json").read_text(encoding="utf-8"))
      self.assertEqual(metadata["model_name"], "hybrid_quantile_transformer")
      self.assertIn("prediction_return_q50_1_bar", metadata["extra"]["prediction_columns"])
      self.assertEqual(tuple(metadata["extra"]["feature_groups"]), ("baseline_core",))
      self.assertEqual(metadata["extra"]["transformer_config"]["backbone"], "timemixer")
      self.assertNotIn("trade_imbalance", metadata["feature_columns"])

      with (artifact_root / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertTrue(rows)
      self.assertIn("prediction_return_q10_1_bar", rows[0])
      self.assertIn("prediction_return_spread_3_bar", rows[0])

  def test_trains_with_auxiliary_trend_targets_and_sample_weights(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_dataset(root / "features")

      summary = train_hybrid_transformers(
        HybridTransformerTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          minimum_rows=12,
          target_columns=("forward_return_1_bar", "forward_return_3_bar"),
          auxiliary_target_columns=(
            "future_trend_state_label_3_bar",
            "future_trend_break_label_3_bar",
            "triple_barrier_label_3_bar",
          ),
          feature_groups=("baseline_core",),
          sample_weight_scheme="hmm_regime_target_balance",
          sample_weight_target_column="future_trend_state_label_3_bar",
          regime_change_boost=1.25,
          transformer=HybridTransformerConfig(
            lookback=6,
            backbone="timemixer",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            scales=(1, 2),
            mixer_layers=1,
            dropout=0.0,
            batch_size=8,
            epochs=2,
            validation_fraction=0.25,
            horizons=(1, 3),
            quantiles=(0.1, 0.5, 0.9),
            auxiliary_loss_weight=0.2,
            auxiliary_loss_weights={
              "future_trend_state_label_3_bar": 1.5,
              "future_trend_break_label_3_bar": 0.0,
              "triple_barrier_label_3_bar": 0.5,
            },
          ),
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_root = next((root / "models").rglob("metadata.json")).parent
      metadata = json.loads((artifact_root / "metadata.json").read_text(encoding="utf-8"))
      self.assertEqual(
        metadata["extra"]["auxiliary_target_columns"],
        [
          "future_trend_state_label_3_bar",
          "future_trend_break_label_3_bar",
          "triple_barrier_label_3_bar",
        ],
      )
      self.assertIn(
        "validation_auxiliary_accuracy_future_trend_state_label_3_bar",
        metadata["metrics"],
      )
      self.assertEqual(
        metadata["extra"]["transformer_config"]["auxiliary_loss_weights"]["future_trend_break_label_3_bar"],
        0.0,
      )

      with (artifact_root / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertTrue(rows)
      self.assertIn("target_future_trend_state_label_3_bar", rows[0])
      self.assertIn("prediction_trend_state_3_bar", rows[0])
      self.assertIn("prediction_probability_uptrend_3_bar", rows[0])

  def _write_dataset(self, input_root: Path) -> None:
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
          "rolling_return_std_5_bar",
          "trend_return_24_bar",
          "regime_hmm_state_index",
          "regime_bocpd_change_prob",
          "forward_return_1_bar",
          "forward_return_3_bar",
          "future_trend_state_label_3_bar",
          "future_trend_break_label_3_bar",
          "triple_barrier_label_3_bar",
        ]
      )
      for minute in range(40):
        writer.writerow(
          [
            "binance",
            "BTCUSDT",
            "2026-04-12",
            f"2026-04-12T19:{minute:02d}:00.000Z",
            f"2026-04-12T19:{minute + 1:02d}:00.000Z",
            "true",
            str(100 + minute),
            str(10 + minute),
            str(20 + minute),
            str(((minute % 5) - 2) / 10),
            str(0.0005 + (minute % 4) * 0.0001),
            str(0.001 * ((minute % 7) - 3)),
            str(minute % 3),
            str(0.5 if minute % 9 == 0 else 0.1),
            str(0.001 * ((minute % 4) - 1.5)),
            str(0.0015 * ((minute % 5) - 2)),
            str((minute % 3) - 1),
            str(1 if minute % 5 == 0 else 0),
            str((minute % 3) - 1),
          ]
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class HybridTransformerWalkForwardTests(unittest.TestCase):
  def test_runs_purged_walk_forward_and_writes_multi_output_predictions(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      HybridTransformerTrainingTests()._write_dataset(root / "features")

      summary = evaluate_hybrid_transformer_walk_forward(
        HybridTransformerWalkForwardConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          dataset_name="market_supervised",
          target_columns=("forward_return_1_bar", "forward_return_3_bar"),
          feature_groups=("baseline_core",),
          initial_train_rows=12,
          validation_rows=6,
          step_rows=6,
          embargo_rows=1,
          purge_rows=3,
          minimum_rows=20,
          transformer=HybridTransformerConfig(
            lookback=6,
            backbone="timemixer",
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            scales=(1, 2),
            mixer_layers=1,
            dropout=0.0,
            batch_size=8,
            epochs=2,
            horizons=(1, 3),
            quantiles=(0.1, 0.5, 0.9),
          ),
        )
      )

      self.assertEqual(summary.runs_attempted, 1)
      self.assertEqual(summary.runs_completed, 1)
      self.assertGreaterEqual(summary.folds_completed, 1)

      artifact_root = next((root / "models").rglob("summary.json")).parent
      payload = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(payload["family"], "hybrid_transformers")
      self.assertIn("overall_validation_forward_return_1_bar_rmse", payload["aggregate_metrics"])

      with (artifact_root / "validation_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
      self.assertTrue(rows)
      self.assertIn("prediction_return_q50_1_bar", rows[0])


if __name__ == "__main__":
  unittest.main()

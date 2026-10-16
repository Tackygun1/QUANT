from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant.validation.meta_walk_forward import MetaWalkForwardConfig, evaluate_meta_walk_forward


class MetaWalkForwardTests(unittest.TestCase):
  def test_writes_purged_chronological_oof_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      input_root = root / "features"
      dataset_dir = input_root / "training_datasets" / "market_hybrid_stacked" / "coinbase" / "BTC-USD" / "2026" / "04"
      dataset_dir.mkdir(parents=True)
      path = dataset_dir / "01.csv"
      start = datetime(2026, 4, 1, tzinfo=UTC)
      with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
          "provider",
          "symbol",
          "candle_start",
          "is_closed",
          "close",
          "rolling_return_std_5_bar",
          "direction_label_1_bar",
        ])
        writer.writeheader()
        for index in range(96):
          writer.writerow({
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "candle_start": (start + timedelta(minutes=5 * index)).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "is_closed": "true",
            "close": 100 + index % 7,
            "rolling_return_std_5_bar": 0.001 + (index % 5) * 0.0001,
            "direction_label_1_bar": (-1, 0, 1)[index % 3],
          })

      summary = evaluate_meta_walk_forward(MetaWalkForwardConfig(
        input_root=input_root,
        output_root=root / "models",
        dataset_name="market_hybrid_stacked",
        providers=("coinbase",),
        symbols=("BTC-USD",),
        initial_train_rows=48,
        validation_rows=16,
        step_rows=16,
        embargo_rows=1,
        purge_rows=1,
        minimum_rows=65,
        model_params={
          "lightgbm_hybrid_direction_meta": {
            "n_estimators": 5,
            "num_leaves": 3,
            "max_depth": 2,
            "min_child_samples": 4,
            "class_weight": "balanced",
            "random_state": 11,
          }
        },
        probability_calibration_fraction=0.2,
        minimum_calibration_rows=8,
      ))

      self.assertEqual(summary.runs_completed, 1)
      self.assertEqual(summary.folds_completed, 2)
      self.assertEqual(summary.prediction_rows, 32)
      output_dir = summary.output_dirs[0]
      payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(payload["window"]["purge_rows"], 1)
      self.assertLess(payload["folds"][0]["calibration_end"], payload["folds"][0]["validation_start"])
      with (output_dir / "validation_predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
      self.assertEqual(len(rows), 32)
      self.assertIn("prediction_margin", rows[0])
      self.assertEqual([row["fold_index"] for row in rows[:16]], ["1"] * 16)
      self.assertEqual([row["fold_index"] for row in rows[16:]], ["2"] * 16)


if __name__ == "__main__":
  unittest.main()

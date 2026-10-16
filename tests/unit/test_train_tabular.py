from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from quant.training.train_tabular import TabularTrainingConfig, train_tabular_models


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
class TrainTabularSmokeTests(unittest.TestCase):
  def test_trains_and_writes_artifacts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_supervised"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_csv(
        dataset_path,
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
          "forward_return_1_bar",
          "direction_label_1_bar",
        ],
        [
          ["binance", "BTCUSDT", "2026-04-12", f"2026-04-12T19:{minute:02d}:00.000Z", f"2026-04-12T19:{minute+1:02d}:00.000Z", "true", str(100 + minute), str(10 + minute), str(20 + minute), str(0.001 * ((minute % 3) - 1)), str((minute % 3) - 1)]
          for minute in range(18)
        ],
      )

      summary = train_tabular_models(
        TabularTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("ridge_return",),
          minimum_rows=8,
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_files = sorted((root / "models").rglob("*"))
      self.assertTrue(any(path.name == "model.pkl" for path in artifact_files))
      self.assertTrue(any(path.name == "metadata.json" for path in artifact_files))

  def test_skips_single_class_classification_runs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_supervised"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_csv(
        dataset_path,
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
          "forward_return_1_bar",
          "direction_label_1_bar",
        ],
        [
          ["binance", "BTCUSDT", "2026-04-12", f"2026-04-12T19:{minute:02d}:00.000Z", f"2026-04-12T19:{minute+1:02d}:00.000Z", "true", str(100 + minute), str(10 + minute), str(20 + minute), str(0.001 * minute), "1"]
          for minute in range(18)
        ],
      )

      summary = train_tabular_models(
        TabularTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          model_names=("logistic_direction",),
          minimum_rows=8,
        )
      )

      self.assertEqual(summary.runs_attempted, 1)
      self.assertEqual(summary.runs_completed, 0)

  @staticmethod
  def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(fieldnames)
      writer.writerows(rows)


if __name__ == "__main__":
  unittest.main()

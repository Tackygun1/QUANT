from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from quant.models.transformers.return_forecaster import TransformerForecasterConfig
from quant.training.train_transformer import (
  TransformerTrainingConfig,
  train_transformer_forecasters,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class TrainTransformerSmokeTests(unittest.TestCase):
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
          "trade_imbalance",
          "forward_return_1_bar",
        ],
        [
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
            str(0.001 * ((minute % 4) - 1.5)),
          ]
          for minute in range(40)
        ],
      )

      summary = train_transformer_forecasters(
        TransformerTrainingConfig(
          input_root=root / "features",
          output_root=root / "models",
          providers=("binance",),
          symbols=("BTCUSDT",),
          minimum_rows=8,
          transformer=TransformerForecasterConfig(
            lookback=8,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            batch_size=8,
            epochs=2,
            validation_fraction=0.25,
          ),
        )
      )

      self.assertEqual(summary.runs_completed, 1)
      artifact_files = sorted((root / "models").rglob("*"))
      self.assertTrue(any(path.name == "model.pt" for path in artifact_files))
      self.assertTrue(any(path.name == "metadata.json" for path in artifact_files))

  @staticmethod
  def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(fieldnames)
      writer.writerows(rows)


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from quant.training.supervised_dataset import SupervisedDatasetConfig, build_supervised_datasets


class SupervisedDatasetTests(unittest.TestCase):
  def test_joins_feature_and_label_tables(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_csv(
        root / "features" / "market_features" / "binance" / "BTCUSDT" / "2026" / "04" / "12.csv",
        ["provider", "symbol", "feature_date", "candle_start", "candle_end", "close", "volume"],
        [
          ["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:29:00.000Z", "2026-04-12T19:30:00.000Z", "100", "10"],
          ["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:30:00.000Z", "2026-04-12T19:31:00.000Z", "101", "11"],
        ],
      )
      self._write_csv(
        root / "features" / "market_labels" / "binance" / "BTCUSDT" / "2026" / "04" / "12.csv",
        ["provider", "symbol", "feature_date", "candle_start", "candle_end", "forward_return_1_bar"],
        [
          ["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:29:00.000Z", "2026-04-12T19:30:00.000Z", "0.01"],
          ["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:30:00.000Z", "2026-04-12T19:31:00.000Z", ""],
        ],
      )

      summary = build_supervised_datasets(
        SupervisedDatasetConfig(
          input_root=root / "features",
          output_root=root / "features",
        )
      )

      self.assertEqual(summary.datasets_processed, 1)
      output_path = (
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
      rows = self._read_csv(output_path)
      self.assertEqual(len(rows), 2)
      self.assertEqual(rows[0]["forward_return_1_bar"], "0.01")
      self.assertEqual(rows[0]["has_any_forward_label"], "true")
      self.assertEqual(rows[0]["has_1_bar_label"], "true")
      self.assertEqual(rows[1]["has_any_forward_label"], "false")

  @staticmethod
  def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(fieldnames)
      writer.writerows(rows)

  @staticmethod
  def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
      return list(csv.DictReader(handle))


if __name__ == "__main__":
  unittest.main()

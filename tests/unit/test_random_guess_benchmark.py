from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant.research.random_guess_benchmark import RandomGuessBenchmarkConfig, run_random_guess_benchmark
from quant.risk.sizing import PositionSizingConfig


class RandomGuessBenchmarkTests(unittest.TestCase):
  def test_compares_seeded_random_and_shuffled_predictions(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      artifact_dir = root / "reference_artifact"
      backtest_dir = root / "reference_backtest"
      input_root = root / "features"
      dataset_name = "market_stacked"
      self._write_reference_artifact(artifact_dir)
      self._write_reference_backtest(backtest_dir, input_root, dataset_name)
      self._write_dataset(input_root, dataset_name)

      summary = run_random_guess_benchmark(RandomGuessBenchmarkConfig(
        reference_artifact_dir=artifact_dir,
        reference_backtest_dir=backtest_dir,
        output_root=root / "reports",
        simulations_per_baseline=3,
        seed=17,
      ))

      self.assertEqual(summary.status, "completed")
      self.assertEqual(summary.simulations_completed, 6)
      self.assertEqual(len(summary.representative_artifact_dirs), 2)
      payload = json.loads((summary.output_dir / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(payload["baselines"]["uniform_random_class"]["runs"], 3)
      self.assertEqual(payload["baselines"]["shuffled_model_signal"]["runs"], 3)
      with (summary.output_dir / "simulations.csv").open(encoding="utf-8", newline="") as handle:
        simulation_rows = list(csv.DictReader(handle))
      self.assertEqual(len(simulation_rows), 6)
      with (
        summary.representative_artifact_dirs[0] / "validation_predictions.csv"
      ).open(encoding="utf-8", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
      self.assertEqual(len(prediction_rows), 30)
      self.assertEqual({row["prediction"] for row in prediction_rows} - {"-1", "0", "1"}, set())

  @staticmethod
  def _write_reference_artifact(path: Path) -> None:
    path.mkdir(parents=True)
    metadata = {
      "model_name": "reference_direction_model",
      "task_type": "classification",
      "target_column": "direction_label_1_bar",
      "provider": "coinbase",
      "symbol": "BTC-USD",
      "feature_columns": [],
      "train_rows": 0,
      "validation_rows": 30,
      "metrics": {
        "overall_oof_accuracy": 0.5,
        "overall_oof_macro_recall": 0.5,
      },
      "extra": {
        "dataset_name": "market_stacked",
        "primary_prediction_column": "prediction_margin",
      },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (path / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow([
        "provider", "symbol", "candle_start", "target", "prediction",
        "prediction_probability_max", "prediction_confidence",
        "prediction_probability_short", "prediction_probability_flat",
        "prediction_probability_long", "prediction_margin", "fold_index",
      ])
      start = datetime(2026, 4, 1, tzinfo=UTC)
      for index in range(30):
        target = (-1, 0, 1)[index % 3]
        prediction = (-1, 0, 1)[(index + 1) % 3]
        margin = (-0.25, 0.0, 0.25)[(index + 1) % 3]
        probabilities = {
          -1: (0.6, 0.25, 0.15),
          0: (0.25, 0.5, 0.25),
          1: (0.15, 0.25, 0.6),
        }[prediction]
        writer.writerow([
          "coinbase",
          "BTC-USD",
          (start + timedelta(minutes=5 * index)).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
          target,
          prediction,
          max(probabilities),
          max(probabilities) - sorted(probabilities)[-2],
          *probabilities,
          margin,
          1,
        ])

  @staticmethod
  def _write_reference_backtest(path: Path, input_root: Path, dataset_name: str) -> None:
    path.mkdir(parents=True)
    sizing = PositionSizingConfig(
      classification_margin_threshold=0.12,
      classification_margin_scale=0.35,
      max_portfolio_drawdown_fraction=0.01,
      max_daily_loss_fraction=0.01,
      trend_filter_enabled=False,
    )
    (path / "backtest_config.json").write_text(json.dumps({
      "input_root": str(input_root),
      "dataset_name": dataset_name,
      "realized_return_column": "forward_return_1_bar",
      "fee_bps": 1.0,
      "slippage_bps": 0.5,
      "sizing": asdict(sizing),
    }), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({
      "total_net_return": 0.01,
      "total_transaction_cost": 0.001,
      "max_drawdown": 0.002,
      "sharpe_like": 1.5,
      "profit_factor_after_cost": 1.2,
      "active_hit_rate": 0.55,
      "active_periods": 20,
      "trades": 10,
      "total_turnover": 4.0,
      "aggregate_metrics": {
        "total_gross_return": 0.011,
        "entry_count": 5,
      },
    }), encoding="utf-8")

  @staticmethod
  def _write_dataset(input_root: Path, dataset_name: str) -> None:
    path = (
      input_root / "training_datasets" / dataset_name / "coinbase" / "BTC-USD" / "2026" / "04" / "01.csv"
    )
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow([
        "provider", "symbol", "candle_start", "is_closed", "close",
        "rolling_return_std_5_bar", "forward_return_1_bar", "trend_return_24_bar",
        "trend_return_96_bar",
      ])
      start = datetime(2026, 4, 1, tzinfo=UTC)
      for index in range(30):
        writer.writerow([
          "coinbase",
          "BTC-USD",
          (start + timedelta(minutes=5 * index)).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
          "true",
          100 + index * 0.1,
          0.001,
          (1 if index % 2 == 0 else -1) * 0.0005,
          0.003,
          0.004,
        ])


if __name__ == "__main__":
  unittest.main()

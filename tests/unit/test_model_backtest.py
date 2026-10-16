from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.backtesting.model_backtest import ModelBacktestConfig, run_model_backtest
from quant.risk.sizing import PositionSizingConfig


class ModelBacktestTests(unittest.TestCase):
  def test_runs_fee_aware_backtest_from_validation_artifact(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      artifact_dir = root / "models" / "ensembles" / "lightgbm_return_meta" / "binance" / "BTCUSDT" / "20260413T010000Z"
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_metadata(artifact_dir / "metadata.json")
      self._write_predictions(artifact_dir / "validation_predictions.csv")
      self._write_dataset(dataset_path)

      summary = run_model_backtest(
        ModelBacktestConfig(
          artifact_dir=artifact_dir,
          input_root=root / "features",
          output_root=root / "reports",
          fee_bps=0.0,
          slippage_bps=0.0,
          sizing=PositionSizingConfig(
            volatility_column="rolling_return_std_5_bar",
            volatility_target=0.0015,
            volatility_floor=0.0005,
            max_abs_position=1.0,
            signal_threshold_bps=0.0,
            signal_scale_bps=1.0,
            max_position_change=2.0,
            max_daily_loss_fraction=1.0,
            max_portfolio_drawdown_fraction=1.0,
          ),
        )
      )

      self.assertEqual(summary.status, "completed")
      self.assertEqual(summary.joined_rows, 4)
      self.assertEqual(summary.active_periods, 3)
      self.assertEqual(summary.trades, 3)
      self.assertAlmostEqual(summary.total_net_return, 0.00700196, places=9)

      summary_path = summary.output_dir / "summary.json"
      payload = json.loads(summary_path.read_text(encoding="utf-8"))
      self.assertAlmostEqual(payload["aggregate_metrics"]["total_net_return"], 0.00700196, places=9)

      with (summary.output_dir / "periods.csv").open("r", encoding="utf-8", newline="") as handle:
        periods = list(csv.DictReader(handle))
      self.assertEqual(len(periods), 4)
      self.assertEqual(periods[0]["target_position"], "1.0")
      self.assertEqual(periods[0]["active_position_bars"], "1")
      self.assertEqual(periods[2]["target_position"], "-1.0")

  def test_profit_take_exit_reduces_hold_duration_in_backtest(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      artifact_dir = root / "models" / "ensembles" / "lightgbm_return_meta" / "binance" / "BTCUSDT" / "20260413T010000Z"
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_metadata(artifact_dir / "metadata.json")
      self._write_predictions(artifact_dir / "validation_predictions.csv")
      self._write_dataset(dataset_path)

      summary = run_model_backtest(
        ModelBacktestConfig(
          artifact_dir=artifact_dir,
          input_root=root / "features",
          output_root=root / "reports",
          fee_bps=0.0,
          slippage_bps=0.0,
          sizing=PositionSizingConfig(
            volatility_column="rolling_return_std_5_bar",
            volatility_target=0.0015,
            volatility_floor=0.0005,
            max_abs_position=1.0,
            signal_threshold_bps=0.0,
            signal_scale_bps=1.0,
            stop_loss_floor_bps=50.0,
            profit_take_exit_multiple=1.5,
            max_position_change=2.0,
            max_daily_loss_fraction=1.0,
            max_portfolio_drawdown_fraction=1.0,
          ),
        )
      )

      self.assertEqual(summary.status, "completed")
      self.assertEqual(summary.active_periods, 2)
      self.assertEqual(summary.trades, 4)
      self.assertAlmostEqual(summary.total_net_return, 0.00902, places=9)

      with (summary.output_dir / "periods.csv").open("r", encoding="utf-8", newline="") as handle:
        periods = list(csv.DictReader(handle))
      self.assertEqual(periods[1]["target_position"], "0.0")
      self.assertIn("profit_take_exit", periods[1]["constraint_flags"])
      self.assertEqual(periods[1]["entry_price"], "")

  def test_hold_until_stop_or_take_profit_reduces_flip_churn(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      artifact_dir = root / "models" / "ensembles" / "lightgbm_return_meta" / "binance" / "BTCUSDT" / "20260413T010000Z"
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_stacked"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      self._write_metadata(artifact_dir / "metadata.json")
      self._write_hold_until_predictions(artifact_dir / "validation_predictions.csv")
      self._write_hold_until_dataset(dataset_path)

      summary = run_model_backtest(
        ModelBacktestConfig(
          artifact_dir=artifact_dir,
          input_root=root / "features",
          output_root=root / "reports",
          fee_bps=0.0,
          slippage_bps=0.0,
          sizing=PositionSizingConfig(
            volatility_column="rolling_return_std_5_bar",
            volatility_target=0.0015,
            volatility_floor=0.0005,
            max_abs_position=1.0,
            signal_threshold_bps=0.0,
            signal_scale_bps=1.0,
            stop_loss_floor_bps=50.0,
            profit_take_exit_multiple=2.0,
            hold_until_stop_or_take_profit=True,
            max_position_change=2.0,
            max_daily_loss_fraction=1.0,
            max_portfolio_drawdown_fraction=1.0,
          ),
        )
      )

      self.assertEqual(summary.status, "completed")
      self.assertEqual(summary.trades, 2)

      with (summary.output_dir / "periods.csv").open("r", encoding="utf-8", newline="") as handle:
        periods = list(csv.DictReader(handle))
      self.assertIn("hold_until_exit", periods[1]["constraint_flags"])
      self.assertEqual(periods[1]["target_position"], "1.0")
      self.assertIn("profit_take_exit", periods[2]["constraint_flags"])
      self.assertEqual(periods[2]["target_position"], "0.0")

  @staticmethod
  def _write_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "model_name": "lightgbm_return_meta",
      "task_type": "regression",
      "target_column": "forward_return_1_bar",
      "provider": "binance",
      "symbol": "BTCUSDT",
      "feature_columns": ["close", "rolling_return_std_5_bar"],
      "train_rows": 24,
      "validation_rows": 4,
      "metrics": {},
      "extra": {
        "dataset_name": "market_stacked",
      },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

  @staticmethod
  def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(["provider", "symbol", "candle_start", "target", "prediction"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:00:00.000Z", "0.005", "0.010"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:01:00.000Z", "-0.002", "0.020"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:02:00.000Z", "-0.004", "-0.030"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:03:00.000Z", "0.001", "0.000"])

  @staticmethod
  def _write_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(
        [
          "provider",
          "symbol",
          "feature_date",
          "interval",
          "candle_start",
          "candle_end",
          "is_closed",
          "high",
          "low",
          "close",
          "rolling_return_std_5_bar",
          "forward_return_1_bar",
        ]
      )
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:00:00.000Z", "2026-04-12T19:01:00.000Z", "true", "100.2", "99.8", "100", "0.001", "0.005"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:01:00.000Z", "2026-04-12T19:02:00.000Z", "true", "101.2", "100.8", "101", "0.001", "-0.002"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:02:00.000Z", "2026-04-12T19:03:00.000Z", "true", "102.2", "101.8", "102", "0.001", "-0.004"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:03:00.000Z", "2026-04-12T19:04:00.000Z", "true", "103.2", "102.8", "103", "0.001", "0.001"])

  @staticmethod
  def _write_hold_until_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(["provider", "symbol", "candle_start", "target", "prediction"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:00:00.000Z", "0.004", "0.020"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:01:00.000Z", "0.004", "-0.030"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:02:00.000Z", "-0.002", "-0.030"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12T19:03:00.000Z", "0.000", "0.000"])

  @staticmethod
  def _write_hold_until_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(
        [
          "provider",
          "symbol",
          "feature_date",
          "interval",
          "candle_start",
          "candle_end",
          "is_closed",
          "high",
          "low",
          "close",
          "rolling_return_std_5_bar",
          "forward_return_1_bar",
        ]
      )
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:00:00.000Z", "2026-04-12T19:01:00.000Z", "true", "100.2", "99.8", "100.0", "0.001", "0.004"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:01:00.000Z", "2026-04-12T19:02:00.000Z", "true", "100.45", "100.1", "100.4", "0.001", "0.004"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:02:00.000Z", "2026-04-12T19:03:00.000Z", "true", "101.05", "100.7", "100.8", "0.001", "-0.002"])
      writer.writerow(["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T19:03:00.000Z", "2026-04-12T19:04:00.000Z", "true", "100.7", "100.5", "100.6", "0.001", "0.0"])


if __name__ == "__main__":
  unittest.main()

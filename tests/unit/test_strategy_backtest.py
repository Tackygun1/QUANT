from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.backtesting.strategy_backtest import StrategyBacktestConfig, run_strategy_backtest
from quant.risk.sizing import PositionSizingConfig


class StrategyBacktestTests(unittest.TestCase):
  def test_runs_against_reference_artifact_window(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      artifact_dir = root / "models" / "ensembles" / "lightgbm_hybrid_return_meta" / "coinbase" / "BTC-USD" / "20260415T000000Z"
      dataset_path = (
        root
        / "features"
        / "training_datasets"
        / "market_hybrid_stacked"
        / "coinbase"
        / "BTC-USD"
        / "2026"
        / "04"
        / "14.csv"
      )
      self._write_metadata(artifact_dir / "metadata.json")
      self._write_predictions(artifact_dir / "validation_predictions.csv")
      self._write_dataset(dataset_path)

      summary = run_strategy_backtest(
        StrategyBacktestConfig(
          strategy_name="buy_and_hold_long",
          input_root=root / "features",
          output_root=root / "reports",
          reference_artifact_dir=artifact_dir,
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
          strategy_params={"signal_bps": 25.0},
        )
      )

      self.assertEqual(summary.status, "completed")
      self.assertEqual(summary.joined_rows, 3)
      self.assertEqual(summary.trades, 1)
      self.assertAlmostEqual(summary.total_net_return, 0.004000994, places=9)

      summary_path = summary.output_dir / "summary.json"
      payload = json.loads(summary_path.read_text(encoding="utf-8"))
      self.assertEqual(payload["reference_model_name"], "lightgbm_hybrid_return_meta")
      self.assertEqual(payload["joined_rows"], 3)

  @staticmethod
  def _write_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
      json.dumps(
        {
          "model_name": "lightgbm_hybrid_return_meta",
          "task_type": "regression",
          "target_column": "forward_return_1_bar",
          "provider": "coinbase",
          "symbol": "BTC-USD",
          "feature_columns": ["close", "close_return_5_bar"],
          "train_rows": 8,
          "validation_rows": 3,
          "metrics": {},
          "extra": {
            "dataset_name": "market_hybrid_stacked",
          },
        },
        indent=2,
        sort_keys=True,
      ),
      encoding="utf-8",
    )

  @staticmethod
  def _write_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(["provider", "symbol", "candle_start", "target", "prediction"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14T00:00:00.000Z", "0.002", "0.001"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14T00:01:00.000Z", "-0.001", "0.0005"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14T00:02:00.000Z", "0.003", "0.002"])

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
          "close",
          "rolling_return_std_5_bar",
          "close_return_1_bar",
          "close_return_5_bar",
          "trades_window_imbalance",
          "trades_window_vwap",
          "forward_return_1_bar",
        ]
      )
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14", "1m", "2026-04-14T00:00:00.000Z", "2026-04-14T00:01:00.000Z", "true", "100", "0.001", "0.001", "0.003", "0.2", "100.1", "0.002"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14", "1m", "2026-04-14T00:01:00.000Z", "2026-04-14T00:02:00.000Z", "true", "100.2", "0.001", "-0.001", "0.002", "0.1", "100.1", "-0.001"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14", "1m", "2026-04-14T00:02:00.000Z", "2026-04-14T00:03:00.000Z", "true", "100.1", "0.001", "0.002", "0.001", "-0.2", "100.0", "0.003"])
      writer.writerow(["coinbase", "BTC-USD", "2026-04-14", "1m", "2026-04-14T00:03:00.000Z", "2026-04-14T00:04:00.000Z", "true", "100.4", "0.001", "0.001", "0.001", "0.3", "100.3", "0.001"])


if __name__ == "__main__":
  unittest.main()

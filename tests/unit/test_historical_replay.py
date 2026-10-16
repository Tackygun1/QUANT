from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from quant.execution.historical_replay import (
  HistoricalReplayConfig,
  replay_control_path,
  replay_status_path,
  run_historical_replay,
)
from quant.execution.paper_trading import PaperTradingConfig
from quant.models.baselines import RidgeReturnRegressor
from quant.models.base import TrainingArtifactMetadata
from quant.risk.sizing import PositionSizingConfig


class HistoricalReplayTests(unittest.TestCase):
  def test_replays_supervised_rows_into_feature_and_paper_outputs(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "artifact"
      artifact_dir.mkdir(parents=True)

      model = RidgeReturnRegressor(alpha=1.0)
      model.fit(
        [
          [100.0, 101.0, 99.0, 100.5, 0.001],
          [100.5, 101.5, 100.0, 101.0, 0.001],
          [101.0, 102.0, 100.5, 101.7, 0.001],
        ],
        [0.0003, 0.0008, 0.0011],
      )
      model.save(artifact_dir / "model.pkl")
      TrainingArtifactMetadata(
        model_name="ridge_return",
        task_type="regression",
        target_column="forward_return_1_bar",
        provider="coinbase",
        symbol="BTC-USD",
        feature_columns=("open", "high", "low", "close", "rolling_return_std_5_bar"),
        train_rows=3,
        validation_rows=1,
      ).write_json(artifact_dir / "metadata.json")

      source_root = root / "source"
      self._write_supervised_rows(
        source_root / "training_datasets" / "market_supervised" / "coinbase" / "BTC-USD" / "2026" / "04" / "12.csv"
      )

      summary = run_historical_replay(
        HistoricalReplayConfig(
          artifact_dir=artifact_dir,
          source_data_root=source_root,
          provider="coinbase",
          symbol="BTC-USD",
          feature_date="2026-04-12",
          replay_root=root / "replay",
          step_delay_seconds=0.0,
          max_rows=4,
          paper=PaperTradingConfig(
            artifact_dir=artifact_dir,
            product="BTC-USD",
            output_root=root / "replay" / "paper",
            fee_bps=1.0,
            slippage_bps=0.5,
            take_profit_multiple=0.5,
            sizing=PositionSizingConfig(
              signal_threshold_bps=0.0,
              signal_scale_bps=1.0,
              stop_loss_floor_bps=20.0,
              max_position_change=1.0,
            ),
          ),
        )
      )

      self.assertEqual(summary.rows_streamed, 4)
      self.assertEqual(summary.signals_written, 4)
      self.assertTrue((root / "replay" / "replay_summary.json").exists())
      self.assertTrue(replay_control_path(root / "replay").exists())
      self.assertTrue(replay_status_path(root / "replay").exists())

      replay_dataset = root / "replay" / "features" / "training_datasets" / "market_supervised" / "coinbase" / "BTC-USD" / "2026" / "04" / "12.csv"
      with replay_dataset.open("r", encoding="utf-8", newline="") as handle:
        replay_rows = list(csv.DictReader(handle))
      self.assertEqual(len(replay_rows), 4)

      signal_path = root / "replay" / "paper" / "signals" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "19.jsonl"
      self.assertTrue(signal_path.exists())
      last_signal = json.loads(signal_path.read_text(encoding="utf-8").strip().splitlines()[-1])
      self.assertIn("take_profit_price", last_signal)
      self.assertIn("stop_loss_price", last_signal)

      replay_status = json.loads(replay_status_path(root / "replay").read_text(encoding="utf-8"))
      self.assertTrue(replay_status["completed"])
      self.assertEqual(replay_status["rows_streamed"], 4)
      self.assertEqual(replay_status["total_rows"], 4)

  def test_replay_passes_tradeability_artifact_to_signal_creation(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "artifact"
      artifact_dir.mkdir(parents=True)
      tradeability_artifact_dir = root / "tradeability"
      tradeability_artifact_dir.mkdir(parents=True)

      model = RidgeReturnRegressor(alpha=1.0)
      model.fit(
        [
          [100.0, 101.0, 99.0, 100.5, 0.001],
          [100.5, 101.5, 100.0, 101.0, 0.001],
          [101.0, 102.0, 100.5, 101.7, 0.001],
        ],
        [0.0003, 0.0008, 0.0011],
      )
      model.save(artifact_dir / "model.pkl")
      TrainingArtifactMetadata(
        model_name="ridge_return",
        task_type="regression",
        target_column="forward_return_1_bar",
        provider="coinbase",
        symbol="BTC-USD",
        feature_columns=("open", "high", "low", "close", "rolling_return_std_5_bar"),
        train_rows=3,
        validation_rows=1,
      ).write_json(artifact_dir / "metadata.json")

      source_root = root / "source"
      self._write_supervised_rows(
        source_root / "training_datasets" / "market_supervised" / "coinbase" / "BTC-USD" / "2026" / "04" / "12.csv"
      )

      sentinel_tradeability = object()

      def fake_create_paper_signal(**kwargs: object) -> tuple[dict[str, object], dict[str, object] | None]:
        config = kwargs["config"]
        self.assertEqual(config.tradeability_artifact_dir, tradeability_artifact_dir)
        self.assertIs(kwargs["tradeability_artifact"], sentinel_tradeability)
        return (
          {
            "generated_at": "2026-04-12T19:05:00Z",
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "model_name": "ridge_return",
            "task_type": "regression",
            "target_column": "forward_return_1_bar",
            "candle_start": "2026-04-12T19:00:00.000Z",
            "candle_end": "2026-04-12T19:05:00.000Z",
            "prediction": 0.001,
            "target_position": 0.0,
            "position_delta": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "active_position_bars": 0,
            "constraint_flags": [],
          },
          None,
        )

      with (
        mock.patch(
          "quant.execution.historical_replay.load_optional_live_scoring_artifact",
          return_value=(sentinel_tradeability, {"support": object()}),
        ),
        mock.patch("quant.execution.historical_replay.create_paper_signal", side_effect=fake_create_paper_signal),
        mock.patch(
          "quant.execution.historical_replay.create_paper_ledger",
          return_value={
            "generated_at": "2026-04-12T19:05:00Z",
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "model_name": "ridge_return",
            "task_type": "regression",
            "candle_start": "2026-04-12T19:00:00.000Z",
            "candle_end": "2026-04-12T19:05:00.000Z",
            "prediction": 0.001,
            "previous_position": 0.0,
            "target_position": 0.0,
            "active_position_bars": 0,
            "position_delta": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "realized_bar_return": 0.0,
            "gross_return": 0.0,
            "net_return": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "session_start_equity": 1.0,
            "session_date": "2026-04-12",
            "session_drawdown": 0.0,
            "portfolio_drawdown": 0.0,
            "entry_price": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "exit_price": None,
            "exit_reason": "",
            "take_profit_hit": False,
            "stop_loss_hit": False,
            "bar_open": 100.0,
            "bar_high": 101.0,
            "bar_low": 99.0,
            "bar_close": 100.5,
          },
        ),
      ):
        summary = run_historical_replay(
          HistoricalReplayConfig(
            artifact_dir=artifact_dir,
            source_data_root=source_root,
            provider="coinbase",
            symbol="BTC-USD",
            feature_date="2026-04-12",
            replay_root=root / "replay",
            step_delay_seconds=0.0,
            max_rows=1,
            paper=PaperTradingConfig(
              artifact_dir=artifact_dir,
              tradeability_artifact_dir=tradeability_artifact_dir,
              product="BTC-USD",
              output_root=root / "replay" / "paper",
              sizing=PositionSizingConfig(
                tradeability_gate_enabled=True,
              ),
            ),
          )
        )

      self.assertEqual(summary.rows_streamed, 1)

  def test_replay_preloads_history_rows_before_start_row_for_signal_context(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      artifact_dir = root / "artifact"
      artifact_dir.mkdir(parents=True)

      model = RidgeReturnRegressor(alpha=1.0)
      model.fit(
        [
          [100.0, 101.0, 99.0, 100.5, 0.001],
          [100.5, 101.5, 100.0, 101.0, 0.001],
          [101.0, 102.0, 100.5, 101.7, 0.001],
        ],
        [0.0003, 0.0008, 0.0011],
      )
      model.save(artifact_dir / "model.pkl")
      TrainingArtifactMetadata(
        model_name="ridge_return",
        task_type="regression",
        target_column="forward_return_1_bar",
        provider="coinbase",
        symbol="BTC-USD",
        feature_columns=("open", "high", "low", "close", "rolling_return_std_5_bar"),
        train_rows=3,
        validation_rows=1,
      ).write_json(artifact_dir / "metadata.json")

      source_root = root / "source"
      self._write_supervised_rows(
        source_root / "training_datasets" / "market_supervised" / "coinbase" / "BTC-USD" / "2026" / "04" / "12.csv"
      )

      captured_lengths: list[int] = []

      def fake_create_paper_signal(**kwargs: object) -> tuple[dict[str, object], dict[str, object] | None]:
        captured_lengths.append(len(kwargs["feature_rows"]))  # type: ignore[arg-type]
        row = kwargs["row"]
        return (
          {
            "generated_at": row["candle_end"],
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "model_name": "ridge_return",
            "task_type": "regression",
            "target_column": "forward_return_1_bar",
            "candle_start": row["candle_start"],
            "candle_end": row["candle_end"],
            "prediction": 0.001,
            "target_position": 0.0,
            "position_delta": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "active_position_bars": 0,
            "constraint_flags": [],
          },
          None,
        )

      with (
        mock.patch("quant.execution.historical_replay.create_paper_signal", side_effect=fake_create_paper_signal),
        mock.patch(
          "quant.execution.historical_replay.create_paper_ledger",
          return_value={
            "generated_at": "2026-04-12T19:15:00Z",
            "provider": "coinbase",
            "symbol": "BTC-USD",
            "model_name": "ridge_return",
            "task_type": "regression",
            "candle_start": "2026-04-12T19:10:00.000Z",
            "candle_end": "2026-04-12T19:15:00.000Z",
            "prediction": 0.001,
            "previous_position": 0.0,
            "target_position": 0.0,
            "active_position_bars": 0,
            "position_delta": 0.0,
            "turnover": 0.0,
            "transaction_cost": 0.0,
            "realized_bar_return": 0.0,
            "gross_return": 0.0,
            "net_return": 0.0,
            "equity": 1.0,
            "peak_equity": 1.0,
            "session_start_equity": 1.0,
            "session_date": "2026-04-12",
            "session_drawdown": 0.0,
            "portfolio_drawdown": 0.0,
            "entry_price": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "exit_price": None,
            "exit_reason": "",
            "take_profit_hit": False,
            "stop_loss_hit": False,
            "bar_open": 100.0,
            "bar_high": 101.0,
            "bar_low": 99.0,
            "bar_close": 100.5,
          },
        ),
      ):
        summary = run_historical_replay(
          HistoricalReplayConfig(
            artifact_dir=artifact_dir,
            source_data_root=source_root,
            provider="coinbase",
            symbol="BTC-USD",
            feature_date="2026-04-12",
            replay_root=root / "replay",
            step_delay_seconds=0.0,
            start_row=2,
            max_rows=1,
            paper=PaperTradingConfig(
              artifact_dir=artifact_dir,
              product="BTC-USD",
              output_root=root / "replay" / "paper",
              sizing=PositionSizingConfig(),
            ),
          )
        )

      self.assertEqual(summary.rows_streamed, 1)
      self.assertEqual(captured_lengths, [3])

  @staticmethod
  def _write_supervised_rows(path: Path) -> None:
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
          "open",
          "high",
          "low",
          "close",
          "rolling_return_std_5_bar",
          "close_return_1_bar",
          "forward_return_1_bar",
          "has_any_forward_label",
          "has_1_bar_label",
        ]
      )
      for index in range(6):
        writer.writerow(
          [
            "coinbase",
            "BTC-USD",
            "2026-04-12",
            f"2026-04-12T19:{index * 5:02d}:00.000Z",
            f"2026-04-12T19:{index * 5 + 5:02d}:00.000Z",
            "true",
            str(100 + index),
            str(101 + index),
            str(99 + index),
            str(100.5 + index),
            "0.001",
            "0.0008",
            "0.0009",
            "true",
            "true",
          ]
        )


if __name__ == "__main__":
  unittest.main()

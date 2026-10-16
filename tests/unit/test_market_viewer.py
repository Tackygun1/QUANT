from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.monitoring.market_viewer import build_series_index, load_series_payload


class MarketViewerTests(unittest.TestCase):
  def test_loads_joined_series_payload(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_csv(
        root / "training_datasets" / "market_supervised" / "binance" / "BTCUSDT" / "2026" / "04" / "12.csv",
        [
          "provider",
          "symbol",
          "feature_date",
          "candle_start",
          "open",
          "high",
          "low",
          "close",
          "volume",
          "trades_window_count",
          "top_window_mid_last",
          "top_window_spread_mean",
          "forward_return_1_bar_bps",
          "direction_label_1_bar",
          "tradeable_label_1_bar",
          "has_any_forward_label",
        ],
        [
          ["binance", "BTCUSDT", "2026-04-12", "2026-04-12T19:29:00.000Z", "100", "101", "99", "100.5", "10", "4", "100.4", "0.2", "12.5", "1", "1", "true"],
        ],
      )
      self._write_jsonl(
        root / "paper" / "signals" / "binance" / "BTCUSDT" / "2026" / "04" / "12" / "19.jsonl",
        [
          {
            "generated_at": "2026-04-12T19:30:00Z",
            "provider": "binance",
            "symbol": "BTCUSDT",
            "model_name": "histgb_return",
            "candle_start": "2026-04-12T19:29:00.000Z",
            "target_position": 0.4,
            "prediction": 0.0008,
            "stop_loss_price": 99.0,
            "take_profit_price": 103.0,
            "trade_plan_status": "active",
          }
        ],
      )
      self._write_jsonl(
        root / "paper" / "ledger" / "binance" / "BTCUSDT" / "2026" / "04" / "12" / "19.jsonl",
        [
          {
            "generated_at": "2026-04-12T19:30:00Z",
            "provider": "binance",
            "symbol": "BTCUSDT",
            "model_name": "histgb_return",
            "candle_start": "2026-04-12T19:29:00.000Z",
            "target_position": 0.4,
            "net_return": 0.0012,
            "equity": 1.0012,
            "exit_reason": "take_profit",
            "exit_price": 103.0,
          }
        ],
      )
      (root / "replay_control.json").write_text(
        json.dumps({"step_delay_seconds": 0.03, "paused": False, "updated_at": "2026-04-12T19:30:00Z"}),
        encoding="utf-8",
      )
      (root / "replay_status.json").write_text(
        json.dumps({"rows_streamed": 1, "total_rows": 10, "completed": False, "paused": False}),
        encoding="utf-8",
      )

      index = build_series_index(root)
      payload = load_series_payload(root, root / "paper", "binance", "BTCUSDT", "2026-04-12")

      self.assertEqual(len(index), 1)
      self.assertEqual(payload["summary"]["row_count"], 1)
      self.assertEqual(payload["rows"][0]["close"], 100.5)
      self.assertEqual(payload["rows"][0]["direction_label_1_bar"], "1")
      self.assertEqual(payload["paper"]["summary"]["signal_count"], 1)
      self.assertEqual(payload["paper"]["summary"]["latest_equity"], 1.0012)
      self.assertEqual(payload["paper"]["summary"]["take_profit_hit_count"], 1)
      self.assertEqual(payload["paper"]["summary"]["latest_take_profit_price"], 103.0)
      self.assertTrue(payload["replay"]["available"])
      self.assertEqual(payload["replay"]["control"]["step_delay_seconds"], 0.03)
      self.assertEqual(payload["replay"]["status"]["rows_streamed"], 1)

  @staticmethod
  def _write_csv(path: Path, fieldnames: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
      writer = csv.writer(handle)
      writer.writerow(fieldnames)
      writer.writerows(rows)

  @staticmethod
  def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
      for row in rows:
        json.dump(row, handle)
        handle.write("\n")


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from quant.labels.market_labels import LabelBuildConfig, build_market_labels


class MarketLabelBuilderTests(unittest.TestCase):
  def test_builds_forward_returns_and_labels(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_csv(
        root / "features" / "market_features" / "binance" / "BTCUSDT" / "2026" / "04" / "12.csv",
        [
          "provider",
          "symbol",
          "feature_date",
          "interval",
          "candle_start",
          "candle_end",
          "window_seconds",
          "event_time",
          "source_channel",
          "is_closed",
          "open",
          "high",
          "low",
          "close",
          "volume",
          "quote_volume",
          "candle_trade_count",
        ],
        [
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:00:00.000Z", "2026-04-12T18:01:00.000Z", "60", "2026-04-12T18:00:59.999Z", "kline", "true", "100", "101", "99", "100", "10", "1000", "5"],
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:01:00.000Z", "2026-04-12T18:02:00.000Z", "60", "2026-04-12T18:01:59.999Z", "kline", "true", "100", "102", "99", "101", "11", "1100", "6"],
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:02:00.000Z", "2026-04-12T18:03:00.000Z", "60", "2026-04-12T18:02:59.999Z", "kline", "true", "101", "103", "100", "103", "12", "1200", "7"],
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:03:00.000Z", "2026-04-12T18:04:00.000Z", "60", "2026-04-12T18:03:59.999Z", "kline", "true", "103", "104", "101", "102", "13", "1300", "8"],
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:04:00.000Z", "2026-04-12T18:05:00.000Z", "60", "2026-04-12T18:04:59.999Z", "kline", "true", "102", "105", "101", "104", "14", "1400", "9"],
          ["binance", "BTCUSDT", "2026-04-12", "1m", "2026-04-12T18:05:00.000Z", "2026-04-12T18:06:00.000Z", "60", "2026-04-12T18:05:59.999Z", "kline", "true", "104", "106", "103", "105", "15", "1500", "10"],
        ],
      )

      summary = build_market_labels(
        LabelBuildConfig(
          input_root=root / "features",
          output_root=root / "features",
          horizons=(1, 3),
          direction_dead_zone_bps=1,
          trade_cost_bps=50,
          realized_volatility_window=3,
          triple_barrier_profit_bps=50,
          triple_barrier_stop_bps=300,
        )
      )

      self.assertEqual(summary.series_processed, 1)
      output_path = (
        root
        / "features"
        / "market_labels"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12.csv"
      )
      rows = self._read_csv(output_path)

      self.assertEqual(len(rows), 6)
      self.assertEqual(rows[0]["forward_return_1_bar"], "0.01")
      self.assertEqual(rows[0]["direction_label_1_bar"], "1")
      self.assertEqual(rows[0]["tradeable_label_1_bar"], "1")
      self.assertEqual(rows[1]["direction_label_1_bar"], "1")
      self.assertEqual(rows[2]["direction_label_1_bar"], "-1")
      self.assertEqual(rows[0]["forward_return_3_bar"], "0.02")
      self.assertNotEqual(rows[0]["future_trend_slope_3_bar"], "")
      self.assertNotEqual(rows[0]["future_trend_slope_bps_per_bar_3_bar"], "")
      self.assertIn(rows[0]["future_trend_state_label_3_bar"], {"-1", "0", "1"})
      self.assertEqual(rows[0]["triple_barrier_label_1_bar"], "1")
      self.assertEqual(rows[0]["triple_barrier_time_1_bar"], "1")
      self.assertEqual(rows[0]["triple_barrier_return_1_bar"], "0.005")
      self.assertNotEqual(rows[0]["realized_volatility_3_bar"], "")
      self.assertEqual(rows[4]["forward_return_3_bar"], "")

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

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from quant.features.market_features import FeatureBuildConfig, build_market_features


class MarketFeatureBuilderTests(unittest.TestCase):
  def test_builds_candle_anchored_feature_rows(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      self._write_csv(
        root / "staging" / "candles" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "18.csv",
        [
          "provider",
          "symbol",
          "event_time",
          "received_at",
          "source_channel",
          "record_type",
          "interval",
          "candle_start",
          "candle_end",
          "is_closed",
          "open",
          "high",
          "low",
          "close",
          "volume",
          "quote_volume",
          "trade_count",
          "sequence_num",
          "update_type",
        ],
        [
          ["coinbase", "BTC-USD", "2026-04-12T18:00:59.999Z", "2026-04-12T18:01:00.000Z", "candles", "candle", "1m", "2026-04-12T18:00:00.000Z", "2026-04-12T18:00:59.999Z", "true", "100", "101", "99", "100", "10", "1000", "5", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:01:59.999Z", "2026-04-12T18:02:00.000Z", "candles", "candle", "1m", "2026-04-12T18:01:00.000Z", "2026-04-12T18:01:59.999Z", "true", "100", "102", "100", "101", "11", "1111", "6", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:02:59.999Z", "2026-04-12T18:03:00.000Z", "candles", "candle", "1m", "2026-04-12T18:02:00.000Z", "2026-04-12T18:02:59.999Z", "true", "101", "103", "101", "102", "12", "1224", "7", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:03:59.999Z", "2026-04-12T18:04:00.000Z", "candles", "candle", "1m", "2026-04-12T18:03:00.000Z", "2026-04-12T18:03:59.999Z", "true", "102", "104", "102", "103", "13", "1339", "8", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:04:59.999Z", "2026-04-12T18:05:00.000Z", "candles", "candle", "1m", "2026-04-12T18:04:00.000Z", "2026-04-12T18:04:59.999Z", "true", "103", "105", "103", "104", "14", "1456", "9", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:05:59.999Z", "2026-04-12T18:06:00.000Z", "candles", "candle", "1m", "2026-04-12T18:05:00.000Z", "2026-04-12T18:05:59.999Z", "true", "104", "106", "104", "105", "15", "1575", "10", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:06:59.999Z", "2026-04-12T18:07:00.000Z", "candles", "candle", "1m", "2026-04-12T18:06:00.000Z", "2026-04-12T18:06:59.999Z", "true", "105", "108", "105", "107", "16", "1712", "11", "", ""],
        ],
      )
      self._write_csv(
        root / "staging" / "trades" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "18.csv",
        [
          "provider",
          "symbol",
          "event_time",
          "received_at",
          "source_channel",
          "record_type",
          "trade_id",
          "price",
          "size",
          "notional_quote",
          "aggressor_side",
          "is_buyer_maker",
          "sequence_num",
          "update_type",
        ],
        [
          ["coinbase", "BTC-USD", "2026-04-12T18:00:10.000Z", "2026-04-12T18:00:10.100Z", "market_trades", "trade", "1", "100", "1", "100", "BUY", "", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:00:50.000Z", "2026-04-12T18:00:50.100Z", "market_trades", "trade", "2", "100", "2", "200", "SELL", "", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:01:10.000Z", "2026-04-12T18:01:10.100Z", "market_trades", "trade", "3", "101", "1.5", "151.5", "BUY", "", "", ""],
        ],
      )
      self._write_csv(
        root / "staging" / "order_book_top" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "18.csv",
        [
          "provider",
          "symbol",
          "event_time",
          "received_at",
          "source_channel",
          "record_type",
          "best_bid",
          "best_bid_size",
          "best_ask",
          "best_ask_size",
          "mid_price",
          "spread",
          "sequence_num",
          "update_type",
        ],
        [
          ["coinbase", "BTC-USD", "2026-04-12T18:00:20.000Z", "2026-04-12T18:00:20.100Z", "ticker", "ticker", "99.9", "1.2", "100.1", "1.4", "100", "0.2", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:00:40.000Z", "2026-04-12T18:00:40.100Z", "ticker", "ticker", "100.0", "1.0", "100.2", "1.6", "100.1", "0.2", "", ""],
          ["coinbase", "BTC-USD", "2026-04-12T18:01:20.000Z", "2026-04-12T18:01:20.100Z", "ticker", "ticker", "100.9", "1.1", "101.1", "1.2", "101", "0.2", "", ""],
        ],
      )

      summary = build_market_features(
        FeatureBuildConfig(
          input_root=root / "staging",
          output_root=root / "features",
        )
      )

      self.assertEqual(summary.datasets_processed, 1)
      self.assertEqual(summary.feature_rows_written, 7)

      output_path = (
        root
        / "features"
        / "market_features"
        / "coinbase"
        / "BTC-USD"
        / "2026"
        / "04"
        / "12.csv"
      )
      rows = self._read_csv(output_path)

      self.assertEqual(len(rows), 7)
      self.assertEqual(rows[0]["trades_window_count"], "2")
      self.assertEqual(rows[0]["trades_window_imbalance"], "-0.3333333333333333333333333333")
      self.assertEqual(rows[0]["top_window_mid_mean"], "100.05")
      self.assertEqual(rows[0]["top_window_spread_mean"], "0.2")
      self.assertEqual(rows[1]["close_return_1_bar"], "0.01")
      self.assertEqual(rows[1]["trades_window_vwap"], "101")
      self.assertEqual(rows[5]["close_return_5_bar"], "0.05")
      self.assertNotEqual(rows[5]["rolling_return_std_5_bar"], "")
      self.assertEqual(rows[6]["trend_return_6_bar"], "0.07")
      self.assertEqual(rows[6]["structure_prev_high_6_bar"], "106")
      self.assertEqual(rows[6]["structure_prev_low_6_bar"], "99")
      self.assertEqual(rows[6]["structure_breakout_up_6_bar"], "1")
      self.assertEqual(rows[6]["structure_breakout_down_6_bar"], "0")
      self.assertNotEqual(rows[6]["trend_ema_6"], "")
      self.assertNotEqual(rows[6]["trend_log_slope_6_bar"], "")
      self.assertNotEqual(rows[6]["trend_log_slope_tstat_6_bar"], "")
      self.assertNotEqual(rows[6]["trend_efficiency_6_bar"], "")
      self.assertEqual(rows[6]["trend_chop_score_6_bar"], "0")
      self.assertNotEqual(rows[6]["structure_session_vwap"], "")
      self.assertNotEqual(rows[6]["regime_hmm_p_bull"], "")
      self.assertNotEqual(rows[6]["regime_hmm_entropy"], "")
      self.assertNotEqual(rows[6]["regime_bocpd_change_prob"], "")

  def test_builds_supply_and_demand_zone_features_from_repeated_rejections(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      start = datetime(2026, 4, 12, 0, 0, tzinfo=UTC)
      pattern = (
        ("100", "102", "99", "101"),
        ("101", "104", "100", "103"),
        ("103", "106", "101", "102"),
        ("102", "103", "98", "100"),
        ("100", "101", "94", "98"),
        ("98", "100", "94", "100"),
        ("100", "102", "99", "101"),
        ("101", "103", "100", "102"),
      )
      candle_rows: list[list[str]] = []
      for index in range(104):
        open_price, high_price, low_price, close_price = pattern[index % len(pattern)]
        candle_start = start + timedelta(minutes=index)
        candle_end = candle_start + timedelta(minutes=1) - timedelta(milliseconds=1)
        volume = str(10 + (index % 3))
        quote_volume = str((10 + (index % 3)) * int(close_price))
        candle_rows.append(
          [
            "coinbase",
            "BTC-USD",
            candle_end.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            (candle_end + timedelta(milliseconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "candles",
            "candle",
            "1m",
            candle_start.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            candle_end.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "true",
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            quote_volume,
            str(5 + (index % 4)),
            "",
            "",
          ]
        )

      self._write_csv(
        root / "staging" / "candles" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "00.csv",
        [
          "provider",
          "symbol",
          "event_time",
          "received_at",
          "source_channel",
          "record_type",
          "interval",
          "candle_start",
          "candle_end",
          "is_closed",
          "open",
          "high",
          "low",
          "close",
          "volume",
          "quote_volume",
          "trade_count",
          "sequence_num",
          "update_type",
        ],
        candle_rows,
      )

      summary = build_market_features(
        FeatureBuildConfig(
          input_root=root / "staging",
          output_root=root / "features",
        )
      )

      self.assertEqual(summary.datasets_processed, 1)
      self.assertEqual(summary.feature_rows_written, 104)

      output_path = (
        root
        / "features"
        / "market_features"
        / "coinbase"
        / "BTC-USD"
        / "2026"
        / "04"
        / "12.csv"
      )
      rows = self._read_csv(output_path)
      last_row = rows[-1]

      required_fields = (
        "structure_supply_zone_low_24_bar",
        "structure_supply_zone_high_24_bar",
        "structure_supply_zone_distance_24_bar",
        "structure_supply_zone_age_24_bar",
        "structure_supply_zone_volume_share_24_bar",
        "structure_demand_zone_low_24_bar",
        "structure_demand_zone_high_24_bar",
        "structure_demand_zone_distance_24_bar",
        "structure_demand_zone_age_24_bar",
        "structure_demand_zone_volume_share_24_bar",
        "structure_supply_zone_alignment_24_96_bar",
        "structure_demand_zone_alignment_24_96_bar",
      )
      for field in required_fields:
        self.assertNotEqual(last_row[field], "")

      self.assertGreater(int(last_row["structure_supply_zone_touch_count_24_bar"]), 0)
      self.assertGreater(int(last_row["structure_supply_zone_rejection_count_24_bar"]), 0)
      self.assertGreater(int(last_row["structure_demand_zone_touch_count_24_bar"]), 0)
      self.assertGreater(int(last_row["structure_demand_zone_rejection_count_24_bar"]), 0)
      self.assertGreater(float(last_row["structure_supply_zone_alignment_24_96_bar"]), 0.5)
      self.assertGreater(float(last_row["structure_demand_zone_alignment_24_96_bar"]), 0.5)

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

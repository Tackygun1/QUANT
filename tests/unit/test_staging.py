from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from quant.ingestion.staging import StageConfig, run_stager


class StagingPipelineTests(unittest.TestCase):
  def test_stages_trade_rows(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      raw_path = root / "raw" / "coinbase" / "trades" / "BTC-USD" / "2026" / "04" / "12"
      raw_path.mkdir(parents=True)
      (raw_path / "18.jsonl").write_text(
        "\n".join(
          [
            json.dumps(
              {
                "provider": "coinbase",
                "channel": "trades",
                "source_channel": "market_trades",
                "symbol": "BTC-USD",
                "record_type": "trade",
                "received_at": "2026-04-12T18:00:00.100Z",
                "event_time": "2026-04-12T18:00:00.000Z",
                "trade_id": "1",
                "price": "100.5",
                "size": "0.25",
                "side": "BUY",
                "payload": {},
              }
            )
          ]
        ),
        encoding="utf-8",
      )

      summary = run_stager(
        StageConfig(input_root=root / "raw", output_root=root / "staging")
      )

      self.assertEqual(summary.files_processed, 1)
      staged_path = root / "staging" / "trades" / "coinbase" / "BTC-USD" / "2026" / "04" / "12" / "18.csv"
      rows = _read_csv(staged_path)
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["trade_id"], "1")
      self.assertEqual(rows[0]["notional_quote"], "25.125")
      self.assertEqual(rows[0]["aggressor_side"], "BUY")

  def test_stages_candle_rows(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      raw_path = root / "raw" / "binance" / "candles" / "BTCUSDT" / "2026" / "04" / "12"
      raw_path.mkdir(parents=True)
      (raw_path / "18.jsonl").write_text(
        "\n".join(
          [
            json.dumps(
              {
                "provider": "binance",
                "channel": "candles",
                "source_channel": "btcusdt@kline_1m",
                "symbol": "BTCUSDT",
                "record_type": "candle",
                "received_at": "2026-04-12T18:01:00.000Z",
                "event_time": "2026-04-12T18:01:00.000Z",
                "interval": "1m",
                "is_closed": True,
                "candle_start": "2026-04-12T18:01:00.000Z",
                "candle_end": "2026-04-12T18:01:59.999Z",
                "open": "100",
                "high": "105",
                "low": "95",
                "close": "102",
                "volume": "50",
                "payload": {
                  "k": {
                    "q": "5000",
                    "n": 12,
                  }
                },
              }
            )
          ]
        ),
        encoding="utf-8",
      )

      run_stager(StageConfig(input_root=root / "raw", output_root=root / "staging"))

      staged_path = root / "staging" / "candles" / "binance" / "BTCUSDT" / "2026" / "04" / "12" / "18.csv"
      rows = _read_csv(staged_path)
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["interval"], "1m")
      self.assertEqual(rows[0]["is_closed"], "true")
      self.assertEqual(rows[0]["quote_volume"], "5000")
      self.assertEqual(rows[0]["trade_count"], "12")

  def test_stages_order_book_tables(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)

      coinbase_level2_dir = root / "raw" / "coinbase" / "level2" / "BTC-USD" / "2026" / "04" / "12"
      coinbase_level2_dir.mkdir(parents=True)
      (coinbase_level2_dir / "18.jsonl").write_text(
        json.dumps(
          {
            "provider": "coinbase",
            "channel": "level2",
            "source_channel": "level2",
            "symbol": "BTC-USD",
            "record_type": "order_book_update",
            "received_at": "2026-04-12T18:02:00.500Z",
            "event_time": "2026-04-12T18:02:00.000Z",
            "sequence_num": 44,
            "update_type": "update",
            "payload": {
              "product_id": "BTC-USD",
              "type": "update",
              "updates": [
                {
                  "side": "bid",
                  "event_time": "2026-04-12T18:02:00.000Z",
                  "price_level": "100.0",
                  "new_quantity": "2.5",
                },
                {
                  "side": "offer",
                  "event_time": "2026-04-12T18:02:00.100Z",
                  "price_level": "100.2",
                  "new_quantity": "0",
                },
              ],
            },
            "update_count": 2,
          }
        ),
        encoding="utf-8",
      )

      binance_book_ticker_dir = root / "raw" / "binance" / "book_ticker" / "BTCUSDT" / "2026" / "04" / "12"
      binance_book_ticker_dir.mkdir(parents=True)
      (binance_book_ticker_dir / "18.jsonl").write_text(
        json.dumps(
          {
            "provider": "binance",
            "channel": "book_ticker",
            "source_channel": "btcusdt@bookTicker",
            "symbol": "BTCUSDT",
            "record_type": "book_ticker",
            "received_at": "2026-04-12T18:03:00.000Z",
            "event_time": "2026-04-12T18:03:00.000Z",
            "best_bid": "100.0",
            "best_bid_size": "1.1",
            "best_ask": "100.4",
            "best_ask_size": "1.3",
            "payload": {},
          }
        ),
        encoding="utf-8",
      )

      run_stager(StageConfig(input_root=root / "raw", output_root=root / "staging"))

      event_rows = _read_csv(
        root
        / "staging"
        / "order_book_events"
        / "coinbase"
        / "BTC-USD"
        / "2026"
        / "04"
        / "12"
        / "18.csv"
      )
      level_rows = _read_csv(
        root
        / "staging"
        / "order_book_levels"
        / "coinbase"
        / "BTC-USD"
        / "2026"
        / "04"
        / "12"
        / "18.csv"
      )
      top_rows = _read_csv(
        root
        / "staging"
        / "order_book_top"
        / "binance"
        / "BTCUSDT"
        / "2026"
        / "04"
        / "12"
        / "18.csv"
      )

      self.assertEqual(event_rows[0]["update_kind"], "delta")
      self.assertEqual(event_rows[0]["update_count"], "2")
      self.assertEqual(level_rows[0]["update_action"], "set")
      self.assertEqual(level_rows[1]["update_action"], "remove")
      self.assertEqual(top_rows[0]["spread"], "0.4")
      self.assertEqual(top_rows[0]["mid_price"], "100.2")


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8", newline="") as handle:
    return list(csv.DictReader(handle))


if __name__ == "__main__":
  unittest.main()

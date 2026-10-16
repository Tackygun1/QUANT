from __future__ import annotations

import unittest

from quant.ingestion.binance_historical import (
  _interval_to_milliseconds,
  agg_trade_to_record,
  kline_to_record,
)


class BinanceHistoricalTests(unittest.TestCase):
  def test_normalizes_kline_row_into_raw_record(self) -> None:
    record = kline_to_record(
      symbol="BTCUSDT",
      interval="1m",
      row=[
        1712880000000,
        "70000.00",
        "70010.00",
        "69990.00",
        "70005.00",
        "12.5",
        1712880059999,
        "875062.50",
        321,
        "6.2",
        "434031.00",
        "0",
      ],
    )

    self.assertEqual(record["provider"], "binance")
    self.assertEqual(record["channel"], "candles")
    self.assertEqual(record["symbol"], "BTCUSDT")
    self.assertEqual(record["interval"], "1m")
    self.assertEqual(record["is_closed"], True)
    self.assertEqual(record["trade_count"], 321)
    self.assertEqual(record["update_type"], "historical_backfill")

  def test_normalizes_agg_trade_into_raw_record(self) -> None:
    record = agg_trade_to_record(
      symbol="BTCUSDT",
      row={
        "a": 123,
        "p": "70000.50",
        "q": "0.25",
        "f": 100,
        "l": 101,
        "T": 1712880000123,
        "m": False,
        "M": True,
      },
    )

    self.assertEqual(record["provider"], "binance")
    self.assertEqual(record["channel"], "trades")
    self.assertEqual(record["trade_id"], 123)
    self.assertEqual(record["price"], "70000.50")
    self.assertEqual(record["size"], "0.25")
    self.assertEqual(record["taker_side"], "BUY")
    self.assertEqual(record["update_type"], "historical_backfill")

  def test_parses_interval_to_milliseconds(self) -> None:
    self.assertEqual(_interval_to_milliseconds("1m"), 60_000)
    self.assertEqual(_interval_to_milliseconds("2h"), 7_200_000)


if __name__ == "__main__":
  unittest.main()

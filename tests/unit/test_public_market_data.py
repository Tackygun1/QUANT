from __future__ import annotations

import unittest

from quant.connectors.binance import BinancePublicFeed
from quant.connectors.coinbase import CoinbasePublicFeed


class CoinbasePublicFeedTests(unittest.TestCase):
  def test_normalizes_trade_message(self) -> None:
    feed = CoinbasePublicFeed()
    payload = {
      "channel": "market_trades",
      "timestamp": "2023-02-09T20:19:35.39625135Z",
      "sequence_num": 12,
      "events": [
        {
          "type": "snapshot",
          "trades": [
            {
              "trade_id": "100",
              "product_id": "BTC-USD",
              "price": "30000.12",
              "size": "0.1",
              "side": "BUY",
              "time": "2023-02-09T20:19:35.100Z",
            }
          ],
        }
      ],
    }

    records = feed._normalize_message(payload, "2026-04-12T19:00:00.000Z")

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["channel"], "trades")
    self.assertEqual(records[0]["symbol"], "BTC-USD")
    self.assertEqual(records[0]["trade_id"], "100")
    self.assertEqual(records[0]["price"], "30000.12")

  def test_ignores_heartbeats(self) -> None:
    feed = CoinbasePublicFeed()
    payload = {
      "channel": "heartbeats",
      "timestamp": "2023-06-23T20:31:26.122969572Z",
      "sequence_num": 0,
      "events": [{"heartbeat_counter": "3049"}],
    }

    records = feed._normalize_message(payload, "2026-04-12T19:00:00.000Z")

    self.assertEqual(records, [])


class BinancePublicFeedTests(unittest.TestCase):
  def test_normalizes_agg_trade_message(self) -> None:
    feed = BinancePublicFeed()
    payload = {
      "stream": "btcusdt@aggTrade",
      "data": {
        "e": "aggTrade",
        "E": 1672515782136,
        "s": "BTCUSDT",
        "a": 12345,
        "p": "30000.01",
        "q": "0.25",
        "T": 1672515782136,
        "m": False,
      },
    }

    records = feed._normalize_message(payload, "2026-04-12T19:00:00.000Z")

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["channel"], "trades")
    self.assertEqual(records[0]["symbol"], "BTCUSDT")
    self.assertEqual(records[0]["taker_side"], "BUY")

  def test_normalizes_partial_depth_message(self) -> None:
    feed = BinancePublicFeed()
    payload = {
      "stream": "btcusdt@depth20@100ms",
      "data": {
        "lastUpdateId": 160,
        "bids": [["30000.00", "1.0"]],
        "asks": [["30000.10", "2.0"]],
      },
    }

    records = feed._normalize_message(payload, "2026-04-12T19:00:00.000Z")

    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]["channel"], "level2")
    self.assertEqual(records[0]["symbol"], "BTCUSDT")
    self.assertEqual(records[0]["bid_levels"], 1)
    self.assertEqual(records[0]["ask_levels"], 1)


if __name__ == "__main__":
  unittest.main()

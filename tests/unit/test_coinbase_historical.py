from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from quant.ingestion.coinbase_historical import (
  _http_get_json,
  coinbase_candle_to_record,
  coinbase_trade_to_record,
  interval_to_seconds,
)


class _StubResponse:
  def __init__(self, payload: object, headers: dict[str, str] | None = None) -> None:
    self._payload = payload
    self.headers = headers or {}

  def __enter__(self) -> "_StubResponse":
    return self

  def __exit__(self, exc_type, exc, tb) -> bool:
    return False

  def read(self) -> bytes:
    return json.dumps(self._payload).encode("utf-8")


class CoinbaseHistoricalTests(unittest.TestCase):
  def test_normalizes_candle_row_into_raw_record(self) -> None:
    record = coinbase_candle_to_record(
      product="BTC-USD",
      interval="5m",
      granularity_seconds=300,
      row=[1775433600, "82450.12", "82600.00", "82510.10", "82580.33", "12.34567890"],
    )

    self.assertEqual(record["provider"], "coinbase")
    self.assertEqual(record["channel"], "candles")
    self.assertEqual(record["symbol"], "BTC-USD")
    self.assertEqual(record["interval"], "5m")
    self.assertEqual(record["candle_start"], "2026-04-06T00:00:00.000Z")
    self.assertEqual(record["candle_end"], "2026-04-06T00:05:00.000Z")
    self.assertEqual(record["open"], "82510.10")
    self.assertEqual(record["high"], "82600.00")
    self.assertEqual(record["low"], "82450.12")
    self.assertEqual(record["close"], "82580.33")
    self.assertEqual(record["volume"], "12.34567890")
    self.assertEqual(record["update_type"], "historical_backfill")

  def test_parses_interval_to_seconds(self) -> None:
    self.assertEqual(interval_to_seconds("1m"), 60)
    self.assertEqual(interval_to_seconds("5m"), 300)
    self.assertEqual(interval_to_seconds("15m"), 900)
    self.assertEqual(interval_to_seconds("1h"), 3600)
    self.assertEqual(interval_to_seconds("6h"), 21600)
    self.assertEqual(interval_to_seconds("1d"), 86400)

    with self.assertRaises(ValueError):
      interval_to_seconds("2h")

  def test_normalizes_trade_row_into_raw_record(self) -> None:
    record = coinbase_trade_to_record(
      product="BTC-USD",
      row={
        "trade_id": 123,
        "time": "2026-04-12T00:00:00.123456Z",
        "price": "82580.33",
        "size": "0.015",
        "side": "sell",
      },
    )

    self.assertEqual(record["provider"], "coinbase")
    self.assertEqual(record["channel"], "trades")
    self.assertEqual(record["symbol"], "BTC-USD")
    self.assertEqual(record["trade_id"], 123)
    self.assertEqual(record["price"], "82580.33")
    self.assertEqual(record["size"], "0.015")
    self.assertEqual(record["maker_side"], "SELL")
    self.assertEqual(record["taker_side"], "BUY")
    self.assertEqual(record["update_type"], "historical_backfill")

  def test_http_get_json_retries_timeout_and_then_succeeds(self) -> None:
    with (
      patch("quant.ingestion.coinbase_historical.urlopen") as mock_urlopen,
      patch("quant.ingestion.coinbase_historical.time.sleep") as mock_sleep,
    ):
      mock_urlopen.side_effect = [
        TimeoutError("timed out"),
        _StubResponse([{"ok": True}], {"CB-AFTER": "123"}),
      ]

      payload, headers = _http_get_json(
        "/products/BTC-USD/trades",
        {"limit": 1},
        timeout_seconds=1.0,
        retry_attempts=2,
        retry_backoff_seconds=0.0,
      )

    self.assertEqual(payload, [{"ok": True}])
    self.assertEqual(headers, {"CB-AFTER": "123"})
    self.assertEqual(mock_urlopen.call_count, 2)
    mock_sleep.assert_not_called()

  def test_http_get_json_does_not_retry_non_retryable_http_error(self) -> None:
    with patch("quant.ingestion.coinbase_historical.urlopen") as mock_urlopen:
      mock_urlopen.side_effect = HTTPError(
        url="https://api.exchange.coinbase.com/products/BTC-USD/trades",
        code=400,
        msg="bad request",
        hdrs=None,
        fp=None,
      )

      with self.assertRaises(HTTPError):
        _http_get_json(
          "/products/BTC-USD/trades",
          {"limit": 1},
          timeout_seconds=1.0,
          retry_attempts=3,
          retry_backoff_seconds=0.0,
        )

    self.assertEqual(mock_urlopen.call_count, 1)


if __name__ == "__main__":
  unittest.main()

from __future__ import annotations

import json
from typing import Any, AsyncIterator
from urllib.parse import urlencode

from quant.connectors.public_market_data import (
  PublicMarketDataFeed,
  StreamRequest,
  build_record,
  iso_from_unix_millis,
  utc_now_iso,
)


class BinancePublicFeed(PublicMarketDataFeed):
  name = "binance"
  base_ws_url = "wss://data-stream.binance.vision/stream"
  supported_channels = frozenset({"trades", "candles", "book_ticker", "level2"})

  async def iter_records(self, request: StreamRequest) -> AsyncIterator[dict[str, Any]]:
    self.validate_request(request)

    import websockets

    streams = [
      self._stream_name(product, channel, request.candle_interval)
      for product in request.products
      for channel in request.channels
    ]
    query = urlencode({"streams": "/".join(streams)})
    ws_url = f"{self.base_ws_url}?{query}"

    async with websockets.connect(
      ws_url,
      open_timeout=10,
      close_timeout=10,
      max_queue=4096,
    ) as websocket:
      async for message in websocket:
        received_at = utc_now_iso()
        payload = json.loads(message)
        for record in self._normalize_message(payload, received_at):
          yield record

  @staticmethod
  def _stream_name(product: str, channel: str, candle_interval: str) -> str:
    symbol = product.lower()
    if channel == "trades":
      return f"{symbol}@aggTrade"
    if channel == "candles":
      return f"{symbol}@kline_{candle_interval}"
    if channel == "book_ticker":
      return f"{symbol}@bookTicker"
    if channel == "level2":
      return f"{symbol}@depth20@100ms"
    raise ValueError(f"Unsupported Binance channel: {channel}")

  def _normalize_message(
    self,
    payload: dict[str, Any],
    received_at: str,
  ) -> list[dict[str, Any]]:
    data = payload.get("data", payload)
    stream_name = payload.get("stream", "")
    event_type = data.get("e")

    if event_type == "aggTrade":
      return [self._normalize_trade(data, received_at, stream_name)]
    if event_type == "kline":
      return [self._normalize_candle(data, received_at, stream_name)]
    if {"u", "b", "B", "a", "A"} <= data.keys():
      return [self._normalize_book_ticker(data, received_at, stream_name)]
    if "lastUpdateId" in data:
      symbol = stream_name.split("@", 1)[0].upper()
      return [self._normalize_partial_depth(data, symbol, received_at, stream_name)]
    return []

  def _normalize_trade(
    self,
    data: dict[str, Any],
    received_at: str,
    stream_name: str,
  ) -> dict[str, Any]:
    taker_side = "SELL" if data.get("m") else "BUY"
    return build_record(
      provider=self.name,
      channel="trades",
      source_channel=stream_name,
      symbol=data["s"],
      record_type="trade",
      received_at=received_at,
      event_time=iso_from_unix_millis(data["T"]),
      payload=data,
      extra={
        "trade_id": data.get("a"),
        "price": data.get("p"),
        "size": data.get("q"),
        "taker_side": taker_side,
        "is_buyer_maker": data.get("m"),
      },
    )

  def _normalize_candle(
    self,
    data: dict[str, Any],
    received_at: str,
    stream_name: str,
  ) -> dict[str, Any]:
    candle = data["k"]
    return build_record(
      provider=self.name,
      channel="candles",
      source_channel=stream_name,
      symbol=data["s"],
      record_type="candle",
      received_at=received_at,
      event_time=iso_from_unix_millis(candle["t"]),
      payload=data,
      extra={
        "interval": candle.get("i"),
        "is_closed": candle.get("x"),
        "candle_start": iso_from_unix_millis(candle["t"]),
        "candle_end": iso_from_unix_millis(candle["T"]),
        "open": candle.get("o"),
        "high": candle.get("h"),
        "low": candle.get("l"),
        "close": candle.get("c"),
        "volume": candle.get("v"),
      },
    )

  def _normalize_book_ticker(
    self,
    data: dict[str, Any],
    received_at: str,
    stream_name: str,
  ) -> dict[str, Any]:
    return build_record(
      provider=self.name,
      channel="book_ticker",
      source_channel=stream_name,
      symbol=data["s"],
      record_type="book_ticker",
      received_at=received_at,
      event_time=received_at,
      payload=data,
      extra={
        "order_book_update_id": data.get("u"),
        "best_bid": data.get("b"),
        "best_bid_size": data.get("B"),
        "best_ask": data.get("a"),
        "best_ask_size": data.get("A"),
      },
    )

  def _normalize_partial_depth(
    self,
    data: dict[str, Any],
    symbol: str,
    received_at: str,
    stream_name: str,
  ) -> dict[str, Any]:
    return build_record(
      provider=self.name,
      channel="level2",
      source_channel=stream_name,
      symbol=symbol,
      record_type="order_book_snapshot",
      received_at=received_at,
      event_time=received_at,
      payload=data,
      extra={
        "last_update_id": data.get("lastUpdateId"),
        "bid_levels": len(data.get("bids", [])),
        "ask_levels": len(data.get("asks", [])),
      },
    )

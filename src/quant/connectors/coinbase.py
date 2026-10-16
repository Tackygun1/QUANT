from __future__ import annotations

import json
from typing import Any, AsyncIterator

from quant.connectors.public_market_data import (
  PublicMarketDataFeed,
  StreamRequest,
  build_record,
  iso_from_unix_seconds,
  utc_now_iso,
)

CHANNEL_MAP = {
  "trades": "market_trades",
  "candles": "candles",
  "ticker": "ticker",
  "level2": "level2",
  "heartbeats": "heartbeats",
}

REVERSE_CHANNEL_MAP = {value: key for key, value in CHANNEL_MAP.items()}
REVERSE_CHANNEL_MAP["l2_data"] = "level2"


class CoinbasePublicFeed(PublicMarketDataFeed):
  name = "coinbase"
  ws_url = "wss://advanced-trade-ws.coinbase.com"
  supported_channels = frozenset(CHANNEL_MAP)

  async def iter_records(self, request: StreamRequest) -> AsyncIterator[dict[str, Any]]:
    self.validate_request(request)

    import websockets

    subscribed_channels = list(dict.fromkeys([*request.channels, "heartbeats"]))

    async with websockets.connect(
      self.ws_url,
      open_timeout=10,
      close_timeout=10,
      max_queue=4096,
    ) as websocket:
      for channel_alias in subscribed_channels:
        native_channel = CHANNEL_MAP[channel_alias]
        message: dict[str, Any] = {
          "type": "subscribe",
          "channel": native_channel,
        }
        if native_channel != "heartbeats":
          message["product_ids"] = list(request.products)
        await websocket.send(json.dumps(message))

      async for message in websocket:
        received_at = utc_now_iso()
        payload = json.loads(message)
        for record in self._normalize_message(payload, received_at):
          yield record

  def _normalize_message(
    self,
    payload: dict[str, Any],
    received_at: str,
  ) -> list[dict[str, Any]]:
    channel_name = payload.get("channel")
    if channel_name == "subscriptions":
      return []

    channel_alias = REVERSE_CHANNEL_MAP.get(channel_name)
    if channel_alias is None or channel_alias == "heartbeats":
      return []

    sequence_num = payload.get("sequence_num")
    message_timestamp = payload.get("timestamp")
    events = payload.get("events", [])

    if channel_alias == "trades":
      return self._normalize_trades(
        events,
        sequence_num=sequence_num,
        received_at=received_at,
        source_channel=channel_name,
      )

    if channel_alias == "candles":
      return self._normalize_candles(
        events,
        sequence_num=sequence_num,
        received_at=received_at,
        source_channel=channel_name,
      )

    if channel_alias == "ticker":
      return self._normalize_ticker(
        events,
        sequence_num=sequence_num,
        received_at=received_at,
        source_channel=channel_name,
        message_timestamp=message_timestamp,
      )

    if channel_alias == "level2":
      return self._normalize_level2(
        events,
        sequence_num=sequence_num,
        received_at=received_at,
        source_channel=channel_name,
        message_timestamp=message_timestamp,
      )

    return []

  def _normalize_trades(
    self,
    events: list[dict[str, Any]],
    *,
    sequence_num: int | None,
    received_at: str,
    source_channel: str,
  ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
      update_type = event.get("type")
      for trade in event.get("trades", []):
        records.append(
          build_record(
            provider=self.name,
            channel="trades",
            source_channel=source_channel,
            symbol=trade["product_id"],
            record_type="trade",
            received_at=received_at,
            event_time=trade.get("time"),
            payload=trade,
            extra={
              "sequence_num": sequence_num,
              "update_type": update_type,
              "trade_id": trade.get("trade_id"),
              "price": trade.get("price"),
              "size": trade.get("size"),
              "side": trade.get("side"),
            },
          )
        )
    return records

  def _normalize_candles(
    self,
    events: list[dict[str, Any]],
    *,
    sequence_num: int | None,
    received_at: str,
    source_channel: str,
  ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
      update_type = event.get("type")
      for candle in event.get("candles", []):
        records.append(
          build_record(
            provider=self.name,
            channel="candles",
            source_channel=source_channel,
            symbol=candle["product_id"],
            record_type="candle",
            received_at=received_at,
            event_time=iso_from_unix_seconds(candle["start"]),
            payload=candle,
            extra={
              "sequence_num": sequence_num,
              "update_type": update_type,
              "candle_start": iso_from_unix_seconds(candle["start"]),
              "open": candle.get("open"),
              "high": candle.get("high"),
              "low": candle.get("low"),
              "close": candle.get("close"),
              "volume": candle.get("volume"),
            },
          )
        )
    return records

  def _normalize_ticker(
    self,
    events: list[dict[str, Any]],
    *,
    sequence_num: int | None,
    received_at: str,
    source_channel: str,
    message_timestamp: str | None,
  ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
      update_type = event.get("type")
      for ticker in event.get("tickers", []):
        records.append(
          build_record(
            provider=self.name,
            channel="ticker",
            source_channel=source_channel,
            symbol=ticker["product_id"],
            record_type="ticker",
            received_at=received_at,
            event_time=message_timestamp,
            payload=ticker,
            extra={
              "sequence_num": sequence_num,
              "update_type": update_type,
              "price": ticker.get("price"),
              "best_bid": ticker.get("best_bid"),
              "best_ask": ticker.get("best_ask"),
              "volume_24_h": ticker.get("volume_24_h"),
            },
          )
        )
    return records

  def _normalize_level2(
    self,
    events: list[dict[str, Any]],
    *,
    sequence_num: int | None,
    received_at: str,
    source_channel: str,
    message_timestamp: str | None,
  ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
      updates = event.get("updates", [])
      update_type = event.get("type")
      event_time = self._latest_event_time(updates) or message_timestamp
      records.append(
        build_record(
          provider=self.name,
          channel="level2",
          source_channel=source_channel,
          symbol=event["product_id"],
          record_type="order_book_update",
          received_at=received_at,
          event_time=event_time,
          payload=event,
          extra={
            "sequence_num": sequence_num,
            "update_type": update_type,
            "update_count": len(updates),
          },
        )
      )
    return records

  @staticmethod
  def _latest_event_time(updates: list[dict[str, Any]]) -> str | None:
    times = [update.get("event_time") for update in updates if update.get("event_time")]
    if not times:
      return None
    return max(times)

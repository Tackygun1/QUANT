from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from quant.connectors.binance import BinancePublicFeed
from quant.connectors.coinbase import CoinbasePublicFeed
from quant.connectors.public_market_data import PublicMarketDataFeed, StreamRequest
from quant.storage.jsonl import PartitionedJsonlWriter

LOGGER = logging.getLogger(__name__)

PROVIDERS: dict[str, PublicMarketDataFeed] = {
  "binance": BinancePublicFeed(),
  "coinbase": CoinbasePublicFeed(),
}


@dataclass(frozen=True, slots=True)
class CollectorConfig:
  provider: str
  products: tuple[str, ...]
  channels: tuple[str, ...]
  output_root: Path = Path("data/raw")
  candle_interval: str = "1m"
  max_backoff_seconds: float = 30.0
  max_records: int | None = None


async def run_collector(config: CollectorConfig) -> int:
  provider = PROVIDERS.get(config.provider)
  if provider is None:
    supported = ", ".join(sorted(PROVIDERS))
    raise ValueError(f"Unknown provider '{config.provider}'. Supported providers: {supported}.")

  request = StreamRequest(
    products=config.products,
    channels=config.channels,
    candle_interval=config.candle_interval,
  )
  provider.validate_request(request)

  writer = PartitionedJsonlWriter(config.output_root)
  record_count = 0
  backoff_seconds = 1.0

  try:
    while True:
      try:
        LOGGER.info(
          "Connecting provider=%s products=%s channels=%s",
          config.provider,
          ",".join(config.products),
          ",".join(config.channels),
        )

        async for record in provider.iter_records(request):
          path = writer.write(record)
          record_count += 1
          backoff_seconds = 1.0

          if record_count == 1 or record_count % 100 == 0:
            LOGGER.info(
              "Persisted %s records. Latest file: %s",
              record_count,
              path,
            )

          if config.max_records is not None and record_count >= config.max_records:
            LOGGER.info("Reached max_records=%s. Stopping collector.", config.max_records)
            return record_count

        LOGGER.warning(
          "Provider stream ended without an exception. Reconnecting in %.1f seconds.",
          backoff_seconds,
        )
      except asyncio.CancelledError:
        raise
      except KeyboardInterrupt:
        raise
      except Exception as exc:
        LOGGER.warning(
          "Collector error for provider=%s: %s. Reconnecting in %.1f seconds.",
          config.provider,
          exc,
          backoff_seconds,
        )

      await asyncio.sleep(backoff_seconds)
      backoff_seconds = min(backoff_seconds * 2, config.max_backoff_seconds)
  finally:
    writer.close()
